from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from tinygfx import g3d as cg


def flatten(list_to_flatten):
    def _flatten_helper(iterable, flattened_list):
        """
        a helper method to walk through a nested list and return the flattened list in order

        :param iterable:
        :param flattened_list:
        :return:
        """
        for item in iterable:
            # if the item is itself iterable, recursively enter the function
            if not isinstance(item, str) and hasattr(item, '__iter__'):
                _flatten_helper(item, flattened_list)
            else:
                flattened_list.append(item)

    # reset the flattened_system
    flattened_list = []
    _flatten_helper(list_to_flatten, flattened_list)
    return flattened_list


class RaySet(object):
    fields = (
        "generation",
        "intensity",
        "wavelength",
        "index",
        'id'
    )

    def __init__(self, n_rays, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self.rays = cg.bundle_of_rays(n_rays)
        self.metadata = np.zeros((len(RaySet.fields), n_rays))

        # set default values
        self.wavelength = 0.633  # by default assume 633nm light
        self.index = 1
        self.generation = 0
        self.intensity = 100.  # default intensity is 100
        self.id = np.arange(n_rays)  # assign unique ids to each ray

    @classmethod
    def concat(cls, *ray_sets: "RaySet") -> "RaySet":
        """
        Creates a new RaySet by concatenating n number of existing sets
        """
        new_set = cls(0)
        new_set.rays = np.dstack([this_set.rays for this_set in ray_sets])
        new_set.metadata = np.hstack([this_set.metadata for this_set in ray_sets])
        new_set.id = np.arange(new_set.rays.shape[-1])  # re allocate ids
        return new_set

    @property
    def n_rays(self) -> int:
        return self.rays.shape[-1]

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


@dataclass(unsafe_hash=True)
class OpticalSystem(object):
    sources: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
    components: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
    detectors: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
    boundary: cg.TracerSurface = field(
        default_factory=lambda: cg.Cuboid.from_corners((-100, -100, -100), (100, 100, 100)))


class _AnalyticDataFrame(object):
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


class AnalyticRenderer(object):
    ray_offset_value = 1E-6  # how far off the rays are moved from surfaces

    class States(Enum):
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6

    def __init__(self, system=OpticalSystem(), rays_per_source=10, generation_limit=10):
        self._frame = _AnalyticDataFrame()  # make a new dataframe to hold results
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0
        self._simulation_complete = False

        self._system = system  # reference to the Analytical System
        self._sources = flatten(self._system.sources)
        self._surfaces = flatten((self._system.components, self._system.detectors))
        self._rays_per_source = rays_per_source
        self._generation_limit = generation_limit  # how many reflections/refractions a ray can encounter
        self._world_index = 1  # the refractive index of the world

        self._state_machine = {
            AnalyticRenderer.States.INITIALIZE: self._st_initialize,
            AnalyticRenderer.States.PROPAGATE: self._st_propagate,
            AnalyticRenderer.States.RECORD: self._st_record,
            AnalyticRenderer.States.TRIM: self._st_trim,
            AnalyticRenderer.States.FINISH: self._st_finish,
        }

        self._ray_set = RaySet(0)
        self._next_ray_set = RaySet(0)

    def reset(self):
        self._simulation_complete = False
        self._frame = _AnalyticDataFrame()  # reset the dataframe
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
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

    def render(self):
        self._state = AnalyticRenderer.States.INITIALIZE  # kick off the state machine

        # run the state machine to completion
        while self._state != AnalyticRenderer.States.IDLE:
            self._state_machine[self._state]()  # execute the state function for that state

        # return the rendered data
        return self._frame.data

    def get_results(self):
        """
        Returns a dataframe of the Ray Trace Results

        :return: pandas dataframe
        """
        return self._frame.data

    def _generate_flattened_structures(self):
        self._sources = tuple(flatten(self._system.sources))
        self._surfaces = tuple(flatten((self._system.components, self._system.detectors)))

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number
        self._generate_flattened_structures()

        # make a ray set from the concatenated ray sets returned by the sources
        self._ray_set = RaySet.concat(*[source.generate_rays(self._rays_per_source) for source in self._sources])
        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_propagate(self):
        # the hits matrix is an mxn matrix where you have m surfaces in the simulation and n rays being propagated
        hits_matrix = np.zeros((len(self._surfaces), self._ray_set.n_rays))
        hit_distances = np.full(self._ray_set.n_rays, np.inf)
        hit_surfaces = np.full(self._ray_set.n_rays, -1)

        # calculate the intersection distances for every surface in the simulation
        for n, surface in enumerate(self._surfaces):
            surface_hits = surface.intersect(self._ray_set.rays)
            new_minima = np.logical_and(surface_hits >= 0, surface_hits <= hit_distances)
            hit_distances = np.where(new_minima, surface_hits, hit_distances)
            hit_surfaces = np.where(new_minima, n, hit_surfaces)

        # recasting np.inf to -1 to avoid multiplication errors when calculating distance
        hit_distances = np.where(np.isinf(hit_distances), -1, hit_distances)

        self._next_ray_set = RaySet(self._ray_set.n_rays)  # make a new rayset to hold the updated rays
        # next need to call the shader function for each surface and trim the dead rays
        intersection_points = self._ray_set.rays[0] + np.where(hit_distances > 0,
                                                               hit_distances * self._ray_set.rays[1], 0)
        # make a list of rays where they intersect
        self._next_ray_set.rays = np.array((intersection_points, self._ray_set.rays[1]))

        # copy the metadata from the original ray set
        self._generation_number += 1  # increment the generation number
        self._next_ray_set.metadata = self._ray_set.metadata.copy()
        self._next_ray_set.generation = self._generation_number

        # call the surface shader function to get the new rays
        for n, surface in enumerate(self._surfaces):
            ray_mask = (hit_surfaces == n)
            new_rays, new_indices = surface.shade(
                self._next_ray_set.rays[:, :, ray_mask],
                self._next_ray_set.wavelength[ray_mask],
                self._next_ray_set.index[ray_mask])

            self._next_ray_set.rays[:, :, ray_mask] = new_rays
            self._next_ray_set.index[ray_mask] = new_indices

        # insert data into the frame
        self._frame.insert(self._ray_set,
                           self._next_ray_set,
                           hit_surfaces)

        # trim dead rays

        unabsorbed_rays = np.linalg.norm(self._next_ray_set.rays[1], axis=0) != 0
        intersected_rays = hit_surfaces != -1
        living_rays = np.logical_and(unabsorbed_rays, intersected_rays)

        if np.any(living_rays):
            self._ray_set.rays = self._next_ray_set.rays[:, :, living_rays]
            self._ray_set.metadata = self._next_ray_set.metadata[:, living_rays]

            # move rays slightly off the surface or they'll intersect at t=0
            step_size = 1E-8
            self._ray_set.rays[0] += step_size * self._ray_set.rays[1]

            # update the index and wavelength values
            # if the generation limit has not been reached continue propagating
            if self._generation_number < self._generation_limit:
                self._state = self.States.PROPAGATE
            else:
                self._state = self.States.FINISH

        else:
            # if there's no rays left finish the trace
            self._state = self.States.FINISH

    def _st_record(self):
        pass

    def _st_trim(self):
        pass

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE