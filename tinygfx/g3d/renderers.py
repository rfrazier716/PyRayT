from enum import Enum

import numpy as np
from scipy import ndimage as ndimage
from tinygfx.g3d.world_objects import OrthographicCamera


class EdgeRender(object):
    ray_offset_value = 1E-6  # how far off the rays are moved from surfaces

    class States(Enum):
        PROPAGATE = 1
        INTERACT = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6

    def __init__(self, camera: OrthographicCamera, surfaces: list):
        self._state = self.States.IDLE  # by default the renderer is idling
        self._simulation_complete = False
        self._results = None

        self._camera = camera  # make a new dataframe to hold results
        self._surfaces = surfaces

        self._state_machine = {
            self.States.INITIALIZE: self._st_initialize,
            self.States.PROPAGATE: self._st_propagate,
            self.States.INTERACT: self._st_interact,
            self.States.TRIM: self._st_trim,
            self.States.FINISH: self._st_finish,
        }

    def reset(self):
        self._simulation_complete = False
        self._results = None
        self._state = self.States.IDLE  # by default the renderer is idling

    def render(self):
        self._state = self.States.INITIALIZE  # kick off the state machine

        # run the state machine to completion
        while self._state != self.States.IDLE:
            self._state_machine[self._state]()  # execute the state function for that state

        # return the rendered data
        return self._results

    def get_results(self):
        """
        Returns a dataframe of the Ray Trace Results

        :return: pandas dataframe
        """
        return self._frame.data

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number

        # make a ray set from the concatenated ray sets returned by the sources
        self._rays = self._camera.generate_rays()
        self._state = self.States.PROPAGATE  # update the state machine to propagate through the system

    def _st_propagate(self):
        # the hits matrix is an mxn matrix where you have m surfaces in the simulation and n rays being propagated
        hit_distances = np.full(self._rays.shape[-1], np.inf)
        hit_surfaces = np.full(self._rays.shape[-1], -1)

        # calculate the intersection distances for every surface in the simulation
        for n, surface in enumerate(self._surfaces):
            surface_hits = surface.intersect(self._rays)
            new_minima = np.logical_and(surface_hits >= 0, surface_hits <= hit_distances)
            hit_distances = np.where(new_minima, surface_hits, hit_distances)
            hit_surfaces = np.where(new_minima, n, hit_surfaces)

        # assign the hit distances and surfaces to an instance variable so they can be called in the next state
        self._hit_distances = hit_distances
        self._hit_surfaces = hit_surfaces

        self._state = self.States.INTERACT

    def _st_interact(self):
        # find the edges of the surfaces using np diff functions
        # convert the hit_surfaces array into a matrix
        hit_matrix = self._hit_surfaces.reshape(self._camera.get_resolution()[-1], -1)
        h_diffs = np.abs(np.diff(hit_matrix, axis=-1, prepend=0))
        v_diffs = np.abs(np.diff(hit_matrix, axis=0, prepend=0))

        # now do a binary dilation to make the lines a bit thicker
        edges = ndimage.binary_dilation(h_diffs + v_diffs, ndimage.generate_binary_structure(2, 2))
        self._results = edges
        self._state = self.States.FINISH

    def _st_record(self):
        pass

    def _st_trim(self):
        pass

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE