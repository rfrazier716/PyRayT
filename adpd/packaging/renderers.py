from enum import Enum
import adpd.packaging.simple_cg as cg
import numpy as np


def analytic_render(system):
    pass


class AnalyticRenderer(object):
    class States(Enum):
        GENERATE = 0
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5

    def __init__(self, sources=[], system=[], rays_per_source=10, generation_limit=10):
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0

        self._sources = sources
        self._system = system  # list of surfaces in the optical system
        self._rays_per_source = rays_per_source
        self._generation_limit = generation_limit  # how many reflections/refractions a ray can encounter

        self._state_machine = {
            AnalyticRenderer.States.GENERATE: self._st_generate,
            AnalyticRenderer.States.INITIALIZE: self._st_initialize,
            AnalyticRenderer.States.PROPAGATE: self._st_propagate,
            AnalyticRenderer.States.RECORD: self._st_record,
            AnalyticRenderer.States.FINISH: self._st_finish,
        }

    def reset(self):
        self._state = AnalyticRenderer.States.IDLE  # by default the renderer is idling
        self._generation_number = 0

    def set_generation_limit(self, limit):
        self._generation_limit = limit

    def get_generation_limit(self):
        return self._generation_limit

    def load_system(self, system):
        self._system = system

    def render(self):
        self._state = AnalyticRenderer.States.INITIALIZE  # kick off the state machine

        # run the state machine to completion
        while self._state != AnalyticRenderer.States.IDLE:
            self._state_machine[self._state]()  # execute the state function for that state

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number

        # iterate over components in system, if they have a function to generate rays, call it
        attr_string = 'generate_rays'

        n_rays = self._rays_per_source * len(self._sources)  # how many rays to create for the system
        self._rays = cg.bundle_of_rays(n_rays)  # allocate the ray matrix

        # fill the matrix with rays
        for n, source in enumerate(self._sources):
            self._rays[:,:,n*self._rays_per_source:(n + 1) * self._rays_per_source] = source.generate_rays(self._rays_per_source)

        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_generate(self):
        pass

    def _st_propagate(self):
        hits_matrix = np.zeros((len(self._system), self._rays.shape[-1]))
        for n, surface in enumerate(self._system):
            hits_matrix[n] = surface.intersect(self._rays)

        hit_distances = np.min(hits_matrix, axis=0)
        hit_surfaces = np.where(np.isfinite(hit_distances), np.argmin(hits_matrix, axis=0), -1)
        # next need to call the shader function for each surface and trim the dead rays
        intersection_points = self._rays[0] + hit_distances*self._rays[1]
        intersection_rays = np.array((intersection_points, self._rays[1])) # make a list of rays where they intersect

        for n, surface in enumerate(self._system):
            new_rays, new_indices = surface.shade(intersection_rays[:,:,hit_surfaces == n], 1, 0)
            intersection_rays[:, :, hit_surfaces == n] = new_rays

        self._generation_number+=1
        self._state = self.States.PROPAGATE if self._generation_number < self._generation_limit else self.States.IDLE


    def _st_record(self):
        pass

    def _st_finish(self):
        pass
