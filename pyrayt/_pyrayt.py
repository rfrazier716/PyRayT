from dataclasses import dataclass, field
from .utils import wavelength_to_rgb
from enum import Enum
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tinygfx import g3d as cg
from typing import List


class RaySet(np.ndarray):

    fields = ("generation", "intensity", "wavelength", "index", "id")
    """
    The metadata fields that can be accessed from the rayset
    """

    def __new__(cls, n_rays):
        obj = np.zeros((8 + len(RaySet.fields), n_rays), dtype=float)
        return obj.view(cls)

    def __init__(self, n_rays, *args, **kwargs):
        """
        A chunk of continuous memory that stores all rays in a simulation and their metatdata. :class:`RayTracer`
        objects generate a RaySet during during simulations and incrementally stores results in a Pandas dataframe.
        Raysets should not be created by the user. A slice of metadata for call rays in the set can be accessed through
        class properties.

        :param n_rays: How many rays are in the set
        :param args: Additional arguments to be passed to the next class in the MRO
        :param kwargs: Additional Keyword arguments to be passed to the next class in the MRO
        """

        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        # set default values
        self.rays = cg.bundle_of_rays(n_rays)
        self.wavelength = 0.633  # by default assume 633nm light
        self.index = 1
        self.generation = 0
        self.intensity = 100.0  # default intensity is 100
        self.id = np.arange(n_rays)  # assign unique ids to each ray

    @property
    def n_rays(self) -> int:
        """
        How many rays are stored in the set

        :rtype: int
        """
        return self.rays.shape[-1]

    @property
    def metadata(self):
        """
        A slice of all metadata for the set.

        :rtype: np.ndarray
        """
        return self[8:]

    @metadata.setter
    def metadata(self, update):
        self[8:] = update

    @property
    def rays(self) -> np.ndarray:
        """
        A 2x4xn view of the set holding the ray's spatial data. View[0,:] is the position of all rays as 4D
        homogeneous coordinates, and view[1,:] is the normalized direction.

        :rtype: np.ndarray view
        """
        return self[:8].reshape((2, 4, -1))

    @rays.setter
    def rays(self, update):
        self[:8] = update.reshape(8, -1)

    @property
    def generation(self):
        """
        A view of the set holding the generation number for all rays.

        :return: np.ndarray
        """
        return self.metadata[0]

    @generation.setter
    def generation(self, update):
        self.metadata[0] = update

    @property
    def intensity(self):
        """
        A view of the set holding the intensity for all rays.

        :return: np.ndarray
        """
        return self.metadata[1]

    @intensity.setter
    def intensity(self, update):
        self.metadata[1] = update

    @property
    def wavelength(self):
        """
        A view of the set holding the wavelength of all rays, in um.

        :return: np.ndarray
        """
        return self.metadata[2]

    @wavelength.setter
    def wavelength(self, update):
        self.metadata[2] = update

    @property
    def index(self):
        """
        A view of the set holding the current refractive index of all rays.

        :return: np.ndarray
        """
        return self.metadata[3]

    @index.setter
    def index(self, update):
        self.metadata[3] = update

    @property
    def id(self):
        """
        A view of the set holding the unique ray ID of all rays.

        :return: np.ndarray
        """
        return self.metadata[4]

    @id.setter
    def id(self, update):
        self.metadata[4] = update


class _RayTraceDataframe(object):
    """
    Manages the DataFrame that stores ray metadata. the set itself is stored in :attr:`data`.

    """

    def __init__(self):
        self.df_columns = RaySet.fields + (
            "surface",
            "x0",
            "y0",
            "z0",
            "x1",
            "y1",
            "z1",
            "x_tilt",
            "y_tilt",
            "z_tilt",
        )
        self.data = pd.DataFrame(columns=self.df_columns, dtype="float32")

    def insert(
        self, ray_set: RaySet, next_ray_set: RaySet, surface_ids: np.ndarray
    ) -> None:
        # create an array of generation numbers
        # trim the homogeneous coordinate
        trimmed_starts, tilts = ray_set.rays[:, :-1]  # should return a 2x3xn array
        trimmed_ends = next_ray_set.rays[0, :-1]  # returns a 1x3xn array

        # normalize the tilts
        tilts /= np.linalg.norm(tilts, axis=0)

        new_frame = pd.DataFrame(
            np.vstack(
                (ray_set.metadata, surface_ids, trimmed_starts, trimmed_ends, tilts)
            ).T,
            columns=self.df_columns,
        )

        self.data = self.data.append(new_frame, ignore_index=True)


class RayTracer(object):
    ray_offset_value = 1e-6  # how far off the rays are moved from surfaces
    """
    How far to offset rays from their intersected surface between successive generations. prevents rays from immediately
    intersecting with the previously intersected surface.
    """

    ray_intensity_threshold = 0.1  # kill rays whose instensity is below 0.1
    """
    The intensity threshold for rays in the simulation. After each intersection, any rays with intensity values below 
    the threshold are removed from the simulation. 
    """

    class _States(Enum):
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6
        INTERACT = 7

    def __init__(
        self, sources: list, components: list, rays_per_source=10, generation_limit=10
    ):
        """
        A Simulator that traces rays generated by a set of sources through a set of components.

        :param sources: either a single source or an iterable of sources to use for the simulation
        :param components: either a single component or an interable of components to use for the simulation
        :param rays_per_source: How many rays should be generated for each source in the simulation
        :param generation_limit: The maximum generation allowed for a ray before it is terminated by the ray-tracer
        """

        self._frame = _RayTraceDataframe()  # make a new dataframe to hold results
        self._state = RayTracer._States.IDLE  # by default the renderer is idling
        self._generation_number = 0
        self._simulation_complete = False

        # if single elements are passed for sources or components, pad them into a tuple
        if not hasattr(sources, "__iter__"):
            self._sources = (sources,)
        else:
            self._sources = sources

        if not hasattr(
            components, "__iter__"
        ):  # returns True if type of iterable - same problem with strings
            self._components = (components,)
        else:
            self._components = components

        self._rays_per_source = rays_per_source
        self._generation_limit = (
            generation_limit  # how many reflections/refractions a ray can encounter
        )
        self._world_index = 1  # the refractive index of the world

        self._state_machine = {
            RayTracer._States.INITIALIZE: self._st_initialize,
            RayTracer._States.PROPAGATE: self._st_propagate,
            RayTracer._States.FINISH: self._st_finish,
            RayTracer._States.INTERACT: self._st_interact,
        }

        self._ray_set = RaySet(0)
        self._next_ray_set = RaySet(0)

        # make a flattened list of all surface IDS
        self._surface_lut = tuple()
        for shape in self._components:
            self._surface_lut += shape.surface_ids

    def reset(self):
        """
        Reset the simulation, destroying the current results dataframe.

        :return:
        """
        self._simulation_complete = False
        self._frame = _RayTraceDataframe()  # reset the dataframe
        self._state = RayTracer._States.IDLE  # by default the renderer is idling
        self._generation_number = 0

    def set_rays_per_source(self, n_rays: int) -> None:
        """
        Set how many rays each source generates in a simulation

        :param n_rays: the number of rays to generate per source.
        """
        self._rays_per_source = n_rays

    def get_rays_per_source(self) -> int:
        """
        Getter function for the rays per source attribute

        :return: how many rays each source generates
        """
        return self._rays_per_source

    def set_generation_limit(self, limit):
        """
        Setter function for the generation limit attribute

        :param limit: the new generation limit
        """
        self._generation_limit = limit

    def get_generation_limit(self):
        """
        Getter function for the generation limit attribute.

        :return: the current generation limit
        """
        return self._generation_limit

    def load_components(self, components: List[cg.Intersectable]) -> None:
        """
        Overwrites the current component list with a new set of components to trace.

        :param components: a single component or iterable set of components.
        """

        if not hasattr(
            components, "__iter__"
        ):  # returns True if type of iterable - same problem with strings
            self._components = (components,)
        else:
            self._components = components

    def get_system(self):
        """
        Returns the list of current components. This is a view into the current components list, not a shallow copy.
        Use caution when updating the results of this function.

        :return: List[tinygfx.g3d.world_objects.Intersectable]
        """

        return self._system

    def trace(self):
        self._state = RayTracer._States.INITIALIZE  # kick off the state machine

        # run the state machine to completion
        while self._state != RayTracer._States.IDLE:
            self._state_machine[
                self._state
            ]()  # execute the state function for that state

        # return the rendered data
        return self._frame.data

    def get_results(self):
        """
        Returns a dataframe of the Ray Trace Results.

        :return: pandas dataframe
        """
        return self._frame.data

    def calculate_source_ids(self):
        """
        Calculates the Source ID for every ray in the dataframe, adding it as a column
        """
        ids = (self._frame.data["id"] / self._rays_per_source).astype(int)
        self._frame.data["source_id"] = ids

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number

        # make a ray set from the concatenated ray sets returned by the sources
        self._ray_set = np.hstack(
            [source.generate_rays(self._rays_per_source) for source in self._sources]
        ).view(RaySet)
        self._ray_set.id = np.arange(
            self._ray_set.n_rays
        )  # update the ID's to avoid duplicates
        self._state = (
            self._States.PROPAGATE
        )  # update the state machine to propagate through the system

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

        self._state = self._States.INTERACT

    def _st_interact(self):
        # iterate over the nearest surfaces and update the ray_set
        next_ray_set = self._ray_set.copy()

        # any ray that does not intersect a surface has it's direction vector killed
        next_ray_set.rays[1, ..., self._hit_surfaces == -1] = 0

        for surface_id, surface in self._surface_lut:
            surface_mask = self._hit_surfaces == surface_id
            if np.any(surface_mask):
                next_ray_set.rays[0, ..., surface_mask] += (
                    next_ray_set.rays[1, ..., surface_mask]
                    * self._hit_distances[surface_mask, np.newaxis]
                )
                next_ray_set[..., surface_mask] = surface.material.trace(
                    surface, next_ray_set[..., surface_mask]
                )

        # update to elminate dead rays
        # dead rays are anywhere that the direction vector has been set to zero (absorbed) or the surface vector is -1
        # (no intersection)
        absorbed_rays = np.isclose(np.linalg.norm(self._ray_set.rays[1], axis=0), 0)
        powerless_rays = self._ray_set.intensity < self.ray_intensity_threshold
        dead_rays = np.logical_or(
            absorbed_rays, self._hit_surfaces == -1, powerless_rays
        )
        living_rays = np.logical_not(dead_rays)  # living rays aren't dead...
        # increment the generation number

        # if there's no more rays exit the sim without saving data
        if np.all(dead_rays):
            self._state = RayTracer._States.FINISH
        else:
            # otherwise save data
            next_ray_set = next_ray_set[..., living_rays]

            # update the dataframe of results
            self._frame.insert(
                self._ray_set[..., living_rays],
                next_ray_set,
                self._hit_surfaces[living_rays],
            )
            # copy the new ray set over to the current
            self._ray_set = next_ray_set

            # increase the generation number
            self._generation_number += 1
            next_ray_set.generation = self._generation_number

            # if we hit the generation limit exit the sim
            if self._generation_number == self._generation_limit:
                self._state = RayTracer._States.FINISH

            else:
                # move the rays all off of the interesected surface by a small amount to avoid re-colliding
                self._ray_set.rays[0] += self.ray_offset_value * self._ray_set.rays[1]

                # continue the simulation
                self._state = RayTracer._States.PROPAGATE

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self._States.IDLE

    def show(
        self, view="xy", axis=None, color_function=None, ray_width=0.01, **kwargs
    ) -> None:
        """
        Plot the ray trace results in a MatPlotLib figure with orthographic projection.
        If no trace has been run, the componets are rendered and displayed instead.

        :param view: the projected axis of the results, options are 'xy' or 'xz'
        :param axis: the matplotlib axis to plot the results in, if no axis is provided the current axis is resolved
            using plt.gca()
        :param color_function: Color function to use when drawing rays, options are 'wavelength', or 'source'. By
            default will color all rays a uniform color.
        :param ray_width: Width of the rays to draw. This is passed to pyplot.quiver() as 'width'.
        :param kwargs: additional keyword arguments to pass to :func:`~tinygfx.g3d.renderers.draw`, which renders the
            components
        """

        # figure out what color to use based on the color function argument
        color = "C0"
        if color_function == "wavelength":
            color = wavelength_to_rgb(self._frame.data["wavelength"])
        elif color_function == "source":
            n_colors = len(self._sources)
            colors = wavelength_to_rgb(
                np.linspace(0.45, 0.65, n_colors)
            )  # generate the colors we'll use
            color = np.empty((3, self._frame.data.shape[0]))
            for n, this_color in enumerate(colors):
                color = np.where(
                    np.logical_and(
                        self._frame.data["id"] >= n * self._rays_per_source,
                        self._frame.data["id"] < (n + 1) * self._rays_per_source,
                    ),
                    np.atleast_2d(this_color).T,
                    color,
                )
            color = color.T  # transpose so it can be treated as colors again

        shaded = kwargs.pop("shaded", False)
        show_at_end = False
        if axis is None:
            axis = plt.gca()
            show_at_end = True

        cg.renderers.draw(
            self._components, view=view, axis=axis, shaded=shaded, **kwargs
        )

        # set the view projections
        if view == "xy":
            ax0 = "x"
            ax1 = "y"
        elif view == "xz":
            ax0 = "x"
            ax1 = "z"

        x1 = ax0 + "1"
        y1 = ax1 + "1"
        x0 = ax0 + "0"
        y0 = ax1 + "0"

        # if the simulation has been run plot the results
        if self._simulation_complete:
            u = self._frame.data[x1].sub(self._frame.data[x0])
            v = self._frame.data[y1].sub(self._frame.data[y0])
            axis.set_aspect("equal")
            axis.quiver(
                self._frame.data[x0],
                self._frame.data[y0],
                u,
                v,
                color=color,
                scale=1,
                units="x",
                width=ray_width,
            )

        if show_at_end:
            plt.show()


class pin(object):
    """
    A context manager that pins a number of components at their given position and rotation. On exiting the context, all components are reset to their original states.

    e.g:

    .. code-block:: python

        lens = pyrayt.components.lens(10, -10, 1)
        lens.get_position() # [0,0,0,1]

        with pin(lens):
            # in the context manager you can freely manipulate the position of objects
            lens.move_x(100)
            lens.get_position() # [100, 0, 0, 1]

        # Once the context manager exits, any changes to position are reverted
        lens.get_position() # [0, 0, 0, 1]
    """

    _starting_matrices: List

    def __init__(self, *objects_to_pin):
        self._obj_set = objects_to_pin

    def __enter__(self):
        self._starting_matrices = [
            surface.get_world_transform() for surface in self._obj_set
        ]
        return self._obj_set

    def __exit__(self, exception_type, exception_value, traceback):

        for this_object, starting_matrix in zip(self._obj_set, self._starting_matrices):
            final_matrix = this_object.get_world_transform()
            matrix_change = np.matmul(final_matrix, np.linalg.inv(starting_matrix))
            this_object.transform(np.linalg.inv(matrix_change))
