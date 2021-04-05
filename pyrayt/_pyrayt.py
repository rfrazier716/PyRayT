from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tinygfx import g3d as cg


class RaySet(np.ndarray):
    fields = (
        "generation",
        "intensity",
        "wavelength",
        "index",
        'id'
    )

    def __new__(cls, n_rays):
        obj = np.zeros((8 + len(RaySet.fields), n_rays), dtype=float)
        return obj.view(cls)

    def __init__(self, n_rays, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        # set default values
        self.rays = cg.bundle_of_rays(n_rays)
        self.wavelength = 0.633  # by default assume 633nm light
        self.index = 1
        self.generation = 0
        self.intensity = 100.  # default intensity is 100
        self.id = np.arange(n_rays)  # assign unique ids to each ray

    @property
    def n_rays(self) -> int:
        return self.rays.shape[-1]

    @property
    def metadata(self):
        return self[8:]

    @metadata.setter
    def metadata(self, update):
        self[8:] = update

    @property
    def rays(self) -> np.ndarray:
        return self[:8].reshape((2, 4, -1))

    @rays.setter
    def rays(self, update):
        self[:8] = update.reshape(8, -1)

    @property
    def generation(self):
        return self.metadata[0]

    @generation.setter
    def generation(self, update):
        self.metadata[0] = update

    @property
    def intensity(self):
        return self.metadata[1]

    @intensity.setter
    def intensity(self, update):
        self.metadata[1] = update

    @property
    def wavelength(self):
        return self.metadata[2]

    @wavelength.setter
    def wavelength(self, update):
        self.metadata[2] = update

    @property
    def index(self):
        return self.metadata[3]

    @index.setter
    def index(self, update):
        self.metadata[3] = update

    @property
    def id(self):
        return self.metadata[4]

    @id.setter
    def id(self, update):
        self.metadata[4] = update


class _RayTraceDataframe(object):
    def __init__(self):
        self.df_columns = RaySet.fields + ("surface", "x0", "y0", "z0", "x1", "y1", "z1", "x_tilt",
                                           "y_tilt", "z_tilt")
        self.data = pd.DataFrame(columns=self.df_columns, dtype='float32')

    def insert(self, ray_set: RaySet, next_ray_set: RaySet, surface_ids: np.ndarray) -> None:
        # create an array of generation numbers
        # trim the homogeneous coordinate
        trimmed_starts, tilts = ray_set.rays[:, :-1]  # should return a 2x3xn array
        trimmed_ends = next_ray_set.rays[0, :-1]  # returns a 1x3xn array

        # normalize the tilts
        tilts /= np.linalg.norm(tilts, axis=0)

        new_frame = pd.DataFrame(
            np.vstack((ray_set.metadata, surface_ids, trimmed_starts, trimmed_ends, tilts)).T,
            columns=self.df_columns)

        self.data = self.data.append(new_frame, ignore_index=True)


class RayTracer(object):
    ray_offset_value = 1E-6  # how far off the rays are moved from surfaces
    ray_intensity_threshold = 0.1  # kill rays whose instensity is below 0.1

    class States(Enum):
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6
        INTERACT = 7

    def __init__(self, sources: list, components: list, rays_per_source=10, generation_limit=10):
        self._frame = _RayTraceDataframe()  # make a new dataframe to hold results
        self._state = RayTracer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0
        self._simulation_complete = False

        # if single elements are passed for sources or components, pad them into a tuple
        if not hasattr(sources, "__iter__"):
            self._sources = (sources,)
        else:
            self._sources = sources

        if not hasattr(components, "__iter__"):  # returns True if type of iterable - same problem with strings
            self._components = (components,)
        else:
            self._components = components

        self._rays_per_source = rays_per_source
        self._generation_limit = generation_limit  # how many reflections/refractions a ray can encounter
        self._world_index = 1  # the refractive index of the world

        self._state_machine = {
            RayTracer.States.INITIALIZE: self._st_initialize,
            RayTracer.States.PROPAGATE: self._st_propagate,
            RayTracer.States.FINISH: self._st_finish,
            RayTracer.States.INTERACT: self._st_interact
        }

        self._ray_set = RaySet(0)
        self._next_ray_set = RaySet(0)

        # make a flattened list of all surface IDS
        self._surface_lut = tuple()
        for shape in self._components:
            self._surface_lut += shape.surface_ids

    def reset(self):
        self._simulation_complete = False
        self._frame = _RayTraceDataframe()  # reset the dataframe
        self._state = RayTracer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0

    def set_rays_per_source(self, n_rays: int) -> None:
        self._rays_per_source = n_rays

    def get_rays_per_source(self) -> int:
        return self._rays_per_source

    def set_generation_limit(self, limit):
        self._generation_limit = limit

    def get_generation_limit(self):
        return self._generation_limit

    def load_system(self, system):
        self._system = system

    def get_system(self):
        return self._system

    def trace(self):
        self._state = RayTracer.States.INITIALIZE  # kick off the state machine

        # run the state machine to completion
        while self._state != RayTracer.States.IDLE:
            self._state_machine[self._state]()  # execute the state function for that state

        # return the rendered data
        return self._frame.data

    def get_results(self):
        """
        Returns a dataframe of the Ray Trace Results

        :return: pandas dataframe
        """
        return self._frame.data

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number

        # make a ray set from the concatenated ray sets returned by the sources
        self._ray_set = np.hstack([source.generate_rays(self._rays_per_source) for source in self._sources]).view(
            RaySet)
        self._ray_set.id = np.arange(self._ray_set.n_rays)  # update the ID's to avoid duplicates
        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_propagate(self):
        # the hits matrix is an 1xn matrix to track the nearest hits
        hit_distances = np.full(self._ray_set.n_rays, np.inf)
        hit_surfaces = np.full(self._ray_set.n_rays, -1, dtype=np.int64)

        # calculate the intersection distances for every surface in the simulation
        ray_hit_index = np.arange(self._ray_set.n_rays)
        for n, shape in enumerate(self._components):
            shape_hits, shape_surfaces = shape.intersect(self._ray_set.rays)
            # eliminate any negative hits
            shape_hits = np.where(shape_hits > 0, shape_hits, np.inf)
            nearest_hit_arg = np.argmin(shape_hits, axis=0)
            nearest_hit = shape_hits[nearest_hit_arg, ray_hit_index]
            nearest_surface = shape_surfaces[nearest_hit_arg, ray_hit_index]
            new_minima = nearest_hit < hit_distances
            hit_distances = np.where(new_minima, nearest_hit, hit_distances)
            hit_surfaces = np.where(new_minima, nearest_surface, hit_surfaces)

        # assign the hit distances and surfaces to an instance variable so they can be called in the next state
        self._hit_distances = hit_distances
        self._hit_surfaces = hit_surfaces

        self._state = self.States.INTERACT

    def _st_interact(self):
        # iterate over the nearest surfaces and update the ray_set
        next_ray_set = self._ray_set.copy()

        # any ray that does not intersect a surface has it's direction vector killed
        next_ray_set.rays[1, ..., self._hit_surfaces == -1] = 0

        for surface_id, surface in self._surface_lut:
            surface_mask = self._hit_surfaces == surface_id
            if np.any(surface_mask):
                next_ray_set.rays[0, ..., surface_mask] += next_ray_set.rays[1, ..., surface_mask] * \
                                                           self._hit_distances[
                                                               surface_mask, np.newaxis]
                next_ray_set[..., surface_mask] = surface.material.trace(surface, next_ray_set[..., surface_mask])

        # update to elminate dead rays
        # dead rays are anywhere that the direction vector has been set to zero (absorbed) or the surface vector is -1
        # (no intersection)
        absorbed_rays = np.isclose(np.linalg.norm(self._ray_set.rays[1], axis=0), 0)
        powerless_rays = self._ray_set.intensity < self.ray_intensity_threshold
        dead_rays = np.logical_or(absorbed_rays, self._hit_surfaces == -1, powerless_rays)
        living_rays = np.logical_not(dead_rays)  # living rays aren't dead...
        # increment the generation number

        # if there's no more rays exit the sim without saving data
        if np.all(dead_rays):
            self._state = RayTracer.States.FINISH
        else:
            # otherwise save data
            next_ray_set = next_ray_set[..., living_rays]

            # update the dataframe of results
            self._frame.insert(self._ray_set[..., living_rays],
                               next_ray_set,
                               self._hit_surfaces[living_rays])
            # copy the new ray set over to the current
            self._ray_set = next_ray_set

            # increase the generation number
            self._generation_number += 1
            next_ray_set.generation = self._generation_number

            # if we hit the generation limit exit the sim
            if self._generation_number == self._generation_limit:
                self._state = RayTracer.States.FINISH

            else:
                # move the rays all off of the interesected surface by a small amount to avoid re-colliding
                self._ray_set.rays[0] += self.ray_offset_value * self._ray_set.rays[1]

                # continue the simulation
                self._state = RayTracer.States.PROPAGATE

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE

    def show(self, view='xy', axis=None, **kwargs):
        shaded = kwargs.pop('shaded', False)
        if axis is None:
            axis = plt.gca()
        cg.renderers.draw(self._components, view=view, axis=axis, shaded=shaded, **kwargs)

        # set the view projections
        if view == 'xy':
            ax0 = 'x'
            ax1 = 'y'
        elif view == 'xz':
            ax0 = 'x'
            ax1 = 'z'

        x1 = ax0 + '1'
        y1 = ax1 + '1'
        x0 = ax0 + '0'
        y0 = ax1 + '0'

        # if the simulation has been run plot the results
        if self._simulation_complete:
            u = self._frame.data[x1].sub(self._frame.data[x0])
            v = self._frame.data[y1].sub(self._frame.data[y0])
            axis.set_aspect('equal')
            axis.quiver(self._frame.data[x0], self._frame.data[y0], u, v, scale=1, units='x', width=0.01,
                        color='C0')
