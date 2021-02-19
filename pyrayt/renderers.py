from enum import Enum
import pyrayt.simple_cg as cg
import pyrayt.designer as designer
import pandas as pd
import numpy as np


def analytic_render(system):
    pass


class _AnalyticDataFrame(object):
    def __init__(self):
        self.df_columns = ["id", "generation", "wavelength", "index", "x0", "y0", "z0", "x1", "y1", "z1", "x_tilt",
                           "y_tilt", "z_tilt"]
        self.data = pd.DataFrame(columns=self.df_columns, dtype='float32')

    def insert(self, gen_number, ids, wavelengths, indices, ray_starts, ray_ends):
        # create an array of generation numbers
        gen_numbers = np.full(ids.shape, gen_number)
        # trim the homogeneous coordinate
        trimmed_starts = ray_starts[:-1]
        trimmed_ends = ray_ends[:-1]

        # calculate tilts and normalize
        tilts = trimmed_ends - trimmed_starts
        tilts /= np.linalg.norm(tilts, axis=0)

        new_frame = pd.DataFrame(
            np.vstack((ids, gen_numbers, wavelengths, indices, trimmed_starts, trimmed_ends, tilts)).T,
            columns=self.df_columns)

        self.data = self.data.append(new_frame, ignore_index=True)


class AnalyticRenderer(object):
    class States(Enum):
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6

    def __init__(self, system=designer.AnalyticSystem(), rays_per_source=10, generation_limit=10):
        self._frame = _AnalyticDataFrame()  # make a new dataframe to hold results
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0
        self._simulation_complete = False

        self._system = system  # reference to the Analytical System
        self._sources = designer.flatten(self._system['sources'])
        self._surfaces = designer.flatten([self._system[key] for key in ('components', 'detectors')])
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

        self._ids = None  # the ray IDs, which get logged as the propagate
        self._rays = None  # the set of rays
        self._next_ray_set = None  # the next rays to trace with
        self._index = None  # the refractive index of each ray
        self._wavelength = None  # the wavelength of each ray

    def reset(self):
        self._simulation_complete = False
        self._frame = _AnalyticDataFrame()  # reset the dataframe
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0

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

    def get_results(self):
        """
        Returns a dataframe of the Ray Trace Results

        :return: pandas dataframe
        """
        return self._frame.data

    def _generate_flattened_structures(self):
        self._sources = designer.flatten(self._system['sources'])
        self._surfaces = designer.flatten([self._system[key] for key in ('components', 'detectors')])

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number
        self._generate_flattened_structures()

        # iterate over components in system, if they have a function to generate rays, call it
        attr_string = 'generate_rays'

        # create the ray and associated parameter matrices
        n_rays = self._rays_per_source * len(self._surfaces)  # how many rays to create for the system
        self._ids = np.arange(n_rays)  # assign ids
        self._rays = cg.bundle_of_rays(n_rays)  # allocate the ray matrix
        self._index = np.full(n_rays, self._world_index)
        self._wavelength = np.zeros(n_rays)

        # fill the matrix with rays
        for n, source in enumerate(self._surfaces):
            self._rays[:, :, n * self._rays_per_source:(n + 1) * self._rays_per_source] = source.generate_rays(
                self._rays_per_source)
            self._wavelength[n * self._rays_per_source:(n + 1) * self._rays_per_source] = source.get_wavelength(
                self._rays_per_source)

        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_propagate(self):
        hits_matrix = np.zeros((len(self._system), self._rays.shape[-1]))
        for n, surface in enumerate(self._system):
            hits_matrix[n] = surface.intersect(self._rays)

        hit_distances = np.min(hits_matrix, axis=0)
        hit_surfaces = np.where(np.isfinite(hit_distances), np.argmin(hits_matrix, axis=0), -1)

        # next need to call the shader function for each surface and trim the dead rays
        intersection_points = self._rays[0] + np.where(np.isfinite(hit_distances), hit_distances * self._rays[1], 0)
        intersection_rays = np.array((intersection_points, self._rays[1]))  # make a list of rays where they intersect

        for n, surface in enumerate(self._system):
            new_rays, new_indices = surface.shade(
                intersection_rays[:, :, hit_surfaces == n],
                self._wavelength,
                self._index)

            intersection_rays[:, :, hit_surfaces == n] = new_rays

        # advance the generation number and move to the next state
        self._next_ray_set = intersection_rays  # update the next ray set
        self._generation_number += 1
        self._state = self.States.RECORD  # next the rays need to be recorded

    def _st_record(self):
        self._frame.insert(self._generation_number,
                           self._ids,
                           self._wavelength,
                           self._index,
                           self._rays[0],
                           self._next_ray_set[0])

        self._state = self.States.TRIM  # next the rays need to be trimmed

    def _st_trim(self):
        # trim dead rays
        living_rays = np.linalg.norm(self._next_ray_set[1], axis=0) != 0
        if np.any(living_rays):
            self._rays = self._next_ray_set[:, :, living_rays]  # update rays to keep the living rays

            # move rays slightly off the surface or they'll intersect at t=0
            step_size = 1E-5
            self._rays[0] = self._rays[0] + step_size * self._rays[1]

            # update the index and wavelength values
            self._ids = self._ids[living_rays]
            self._index = self._index[living_rays]
            self._wavelength = self._wavelength[living_rays]

            # if the generation limit has not been reached continue propagating
            if self._generation_number < self._generation_limit:
                self._state = self.States.PROPAGATE
            else:
                self._state = self.States.FINISH

        else:
            # if there's no rays left finish the trace
            self._state = self.States.FINISH
        pass

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE  # next the rays need to be trimmed
