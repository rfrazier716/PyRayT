from enum import Enum
from typing import List

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tinygfx.g3d.world_objects import OrthographicCamera, TracerSurface
from tinygfx.g3d import Point


class EdgeRender(object):
    ray_offset_value = 1e-6  # how far off the rays are moved from surfaces

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
        self._shapes = surfaces

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
            self._state_machine[
                self._state
            ]()  # execute the state function for that state

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
        self._state = (
            self.States.PROPAGATE
        )  # update the state machine to propagate through the system

    def _st_propagate(self):
        # the hits matrix is an 1xn matrix to track the nearest hits
        hit_distances = np.full(self._rays.shape[-1], np.inf)
        hit_surfaces = np.full(self._rays.shape[-1], -1)

        # calculate the intersection distances for every surface in the simulation
        ray_hit_index = np.arange(self._rays.shape[-1])
        for n, shape in enumerate(self._shapes):
            shape_hits, shape_surfaces = shape.intersect(self._rays)
            nearest_hit_arg = np.argmin(
                np.where(shape_hits > 0, shape_hits, np.inf), axis=0
            )
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
        # find the edges of the surfaces using np diff functions
        # convert the hit_surfaces array into a matrix
        hit_matrix = self._hit_surfaces.reshape(self._camera.get_resolution()[-1], -1)
        h_diffs = np.abs(np.diff(hit_matrix, axis=-1, prepend=-1))
        v_diffs = np.abs(np.diff(hit_matrix, axis=0, prepend=-1))

        # now do a binary dilation to make the lines a bit thicker
        edges = ndimage.binary_dilation(
            h_diffs + v_diffs,
            ndimage.generate_binary_structure(2, 2),
            iterations=np.maximum(1, int(np.max(hit_matrix.shape) / 300)),
        )
        # edges = (h_diffs + v_diffs) > 0

        # put the result into an image canvas
        canvas = np.zeros((*hit_matrix.shape, 4), dtype=float)
        canvas[..., :] = np.logical_not(edges)[..., np.newaxis]
        canvas[..., 3] = edges
        self._results = canvas
        self._state = self.States.FINISH

    def _st_record(self):
        pass

    def _st_trim(self):
        pass

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE


class ShadedRenderer:
    class States(Enum):
        PROPAGATE = 1
        INTERACT = 2
        FINISH = 3
        IDLE = 4
        INITIALIZE = 5
        TRIM = 6

    def __init__(
        self, camera: OrthographicCamera, shapes: list, light_position: np.ndarray
    ):
        self._state = self.States.IDLE  # by default the renderer is idling
        self._simulation_complete = False
        self._results = None

        self._light = np.asarray(light_position)
        self._camera = camera
        self._shapes = shapes
        self._surface_lut = tuple()

        # make a flattened list of all surface IDS
        for shape in self._shapes:
            self._surface_lut += shape.surface_ids

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
            self._state_machine[
                self._state
            ]()  # execute the state function for that state

        # return the rendered data
        return self._results

    def _st_initialize(self):
        self.reset()  # reset the renderer states/generation number

        # Create an outgoing ray set from the camera
        self._rays = self._camera.generate_rays()
        self._state = (
            self.States.PROPAGATE
        )  # update the state machine to propagate through the system

    def _st_propagate(self):
        # the hits matrix is an 1xn matrix to track the nearest hits
        hit_distances = np.full(self._rays.shape[-1], np.inf)
        hit_surfaces = np.full(self._rays.shape[-1], -1)

        # calculate the intersection distances for every surface in the simulation
        ray_hit_index = np.arange(self._rays.shape[-1])
        for n, shape in enumerate(self._shapes):
            shape_hits, shape_surfaces = shape.intersect(self._rays)
            nearest_hit_arg = np.argmin(
                np.where(shape_hits > 0, shape_hits, np.inf), axis=0
            )
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
        # create a canvas to fill with the shaded values
        canvas = np.zeros((4, self._rays.shape[-1]))

        # iterate over the surfaces and call their shader functions
        for id, surface in self._surface_lut:
            surface_mask = self._hit_surfaces == id
            if np.any(surface_mask):
                canvas[:, surface_mask] = surface.shade(
                    self._rays[..., surface_mask],
                    self._hit_distances[surface_mask],
                    light_positions=self._light,
                )

        # # now draw the material outline
        # hit_matrix = self._hit_surfaces.reshape(self._camera.get_resolution()[-1], -1)
        # h_diffs = np.abs(np.diff(hit_matrix, axis=-1, prepend=0))
        # v_diffs = np.abs(np.diff(hit_matrix, axis=0, prepend=0))
        #
        # # now do a binary dilation to make the lines a bit thicker
        # edges = ndimage.binary_dilation(h_diffs + v_diffs, ndimage.generate_binary_structure(2, 2))
        # canvas=np.where(edges,0, canvas) # set edges to black

        # reshape the canvas so it's mxnx4 rgba values
        canvas = canvas.T.reshape(*self._camera.get_resolution()[::-1], 4)
        self._results = canvas
        self._state = self.States.FINISH

    def _st_record(self):
        pass

    def _st_trim(self):
        pass

    def _st_finish(self):
        self._simulation_complete = True
        self._state = self.States.IDLE


def draw(
    surfaces: List[TracerSurface],
    view="xy",
    axis=None,
    shaded=True,
    bounds=None,
    resolution=640,
):
    if not hasattr(surfaces, "__iter__"):
        surfaces = (surfaces,)

    # draw a surface for a given projection
    bounding_box = np.hstack(
        [surface.bounding_volume.bounding_points[:3] for surface in surfaces]
    )
    if bounds is not None:
        mins = np.asarray(bounds[0])
        maxes = np.asarray(bounds[1])
    else:
        mins = np.min(bounding_box, axis=1)
        maxes = np.max(bounding_box, axis=1)

    if axis is None:
        axis = plt.gca()

    # this case is for a "top" projection in the xy plane
    # the camera origin should be above the object centered over it
    if view == "xy":
        _draw_xy(surfaces, axis, shaded, resolution, maxes, mins)

    elif view == "xz":
        _draw_xz(surfaces, axis, shaded, resolution, maxes, mins)


def _draw_xy(surfaces: List[TracerSurface], axis, shaded, resolution, maxes, mins):
    camera_origin = (maxes + mins) / 2
    camera_origin[2] = 1.5 * maxes[2]
    h_span, v_span = 1.5 * (maxes[:2] - mins[:2])  # the camera spans
    resolution = resolution if h_span > v_span else int(resolution * h_span / v_span)
    # make the camera and move it into position
    camera = OrthographicCamera(resolution, h_span, v_span / h_span)
    camera.rotate_y(90).rotate_z(90).move(*camera_origin[:3])

    light_position = Point(*maxes)
    light_position[2] *= 3

    if shaded:
        renderer = ShadedRenderer(camera, surfaces, light_position=light_position)

    else:
        renderer = EdgeRender(camera, surfaces)

    image = renderer.render()
    if axis is None:
        axis = plt.gca()

    axis.imshow(
        image,
        extent=[
            camera_origin[0] - h_span / 2,
            camera_origin[0] + h_span / 2,
            camera_origin[1] - v_span / 2,
            camera_origin[1] + v_span / 2,
        ],
    )
    axis.set_axisbelow(True)


def _draw_xz(surfaces: List[TracerSurface], axis, shaded, resolution, maxes, mins):
    camera_origin = (maxes + mins) / 2
    camera_origin[1] = 1.5 * maxes[1]
    h_span, v_span = 1.5 * (maxes[[0, 2]] - mins[[0, 2]])  # the camera spans
    resolution = resolution if h_span > v_span else int(resolution * h_span / v_span)
    # make the camera and move it into position
    camera = OrthographicCamera(resolution, h_span, v_span / h_span)
    camera.rotate_z(90).move(*camera_origin[:3])

    light_position = Point(*maxes)
    light_position[1] *= -3

    if shaded:
        renderer = ShadedRenderer(camera, surfaces, light_position=light_position)

    else:
        renderer = EdgeRender(camera, surfaces)
    image = renderer.render()
    if axis is None:
        axis = plt.gca()

    axis.imshow(
        image,
        extent=[
            camera_origin[0] - h_span / 2,
            camera_origin[0] + h_span / 2,
            camera_origin[2] - v_span / 2,
            camera_origin[2] + v_span / 2,
        ],
    )
    axis.set_axisbelow(True)
