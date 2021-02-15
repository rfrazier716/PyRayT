from enum import Enum
import adpd.packaging.simple_cg as cg

def analytic_render(system):


# randers a system of surfaces and writes results to an SQLITE file

class AnalyticRenderer(object):
    class States(Enum):
        GENERATE = 0
        PROPAGATE = 1
        RECORD = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5


    def __init__(self, system=[], rays_per_source = 10, generation_limit = 10):
        self._state = AnalyticRenderer.States.IDLE # by default the renderer is idling
        self._generation_number = 0

        self._system = system # list of surfaces in the optical system
        self._rays_per_source = rays_per_source
        self._generation_limit = generation_limit # how many reflections/refractions a ray can encounter

        self._state_functions = {
            AnalyticRenderer.States.GENERATE : self._st_generate,
            AnalyticRenderer.States.INITIALIZE : self._st_initialize,
            AnalyticRenderer.States.PROPOGATE : self._st_propogate,
            AnalyticRenderer.States.RECORD : self._st_record,
            AnalyticRenderer.States.FINISH : self._st_finish,
        }

    def reset(self):
        self._state = AnalyticRenderer.States.IDLE # by default the renderer is idling
        self._generation_number = 0

    def set_generation_limit(self, limit):
        self._generation_limit = limit

    def get_generation_limit(self):
        return self._generation_limit

    def load_system(self, system):
        self._system = system

    def render(self):
        self._state = AnalyticRenderer.States.INITIALIZE # kick off the state machine

        # run the state machine to completion
        while self._state!= AnalyticRenderer.States.IDLE:
            self._state_machine[self._state]() # execute the state function for that state

    def _st_initialize(self):
        self.reset() # reset the renderer states/generation number

        # iterate over components in system, if they have a function to generate rays, call it
        attr_string = 'generate_rays'

        generate_fns = [getattr(item, attr_string) for item in self._system if hasattr(item, attr_string)]
        n_rays = self._rays_per_source*len(generate_fns) # how many rays to create for the system
        self._rays = cg.bundle_of_rays(n_rays) # allocate the ray matrix

        # fill the matrix with rays
        for n,fn in enumerate(generate_fns):
            self._rays[:(n+1)*n_rays] = fn(self._rays_per_source)

        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_generate(self):
        pass

    def _st_propagate(self):
        pass

    def _st_record(self):
        pass

    def _st_finish(self):
        pass



