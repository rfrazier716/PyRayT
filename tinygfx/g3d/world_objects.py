import collections
import copy

from itertools import count
from numpy import linalg as linalg
from scipy.spatial import transform as transform

import abc
import numpy as np

import tinygfx.g3d.primitives as primitives
from tinygfx.g3d.materials.gooch import BLACK


def bounding_box(point_set):
    """
    Returns an axis oriented bounding box that fully contains the point set
    :param point_set:
    :return:
    """
    point_set_min = np.min(point_set[:3], axis=1)
    point_set_max = np.max(point_set[:3], axis=1)
    return primitives.Cube(point_set_min, point_set_max)


class CountedObject(object):
    _ids = count(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._id = next(self._ids)

    def get_id(self):
        """
        Gets the class specific object ID number for this instance.

        :return: the unique class id of the object instance
        :rtype: int
        """
        return self._id


class WorldObject(CountedObject):
    """
    a world object represents an object in 3D space, it has an origin and a direction, as well as a transform
    matrices to convert to/from world space
    """

    @staticmethod
    def _transfer_matrix():
        """
        Create and return a 4x4 identity matrix

        :return:
        """
        return np.identity(4)

    @staticmethod
    def _sin_cos(angle, format="deg"):
        """
        returns the sine and cosine of the input angle

        :param angle:
        :param format:
        :return:
        """
        if format == "deg":
            cos_a = np.cos(angle * np.pi / 180.0)
            sin_a = np.sin(angle * np.pi / 180.0)

        elif format == "rad":
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

        else:
            raise ValueError(f"{format} is not a valid option for angle units")

        return sin_a, cos_a

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj_origin = primitives.Point(0, 0, 0)  # position in object space
        self._obj_direction = primitives.Vector(0, 0, 1)  # direction in object space

        self._world_origin = primitives.Point(
            0, 0, 0
        )  # the objects position in world space
        self._world_direction = primitives.Vector(
            0, 0, 1
        )  # the objects direction in world space

        # Flags that get set to false whenever the transform matrix has been updated
        self._dir_valid = True
        self._pos_valid = True
        self._obj_transform_valid = True

        self._world_coordinate_transform = np.identity(
            4, dtype=float
        )  # transform matrix from object to world space
        self._object_coordinate_transform = np.identity(4, dtype=float)

        # var watch list is a list of function to call whenever the world transform matrix is updated
        # the function must accept no arguments
        self.var_watchlist = [self._world_matrix_update_handler]

    def _world_matrix_update_handler(self):
        # update the world origin and direction

        self._world_origin = np.matmul(
            self._world_coordinate_transform, self._obj_origin
        )
        world_dir = np.matmul(self._world_coordinate_transform, self._obj_direction)
        norm = linalg.norm(world_dir)
        if norm < 1e-7:
            raise ValueError(
                f"Measured Norm of World primitives.Vector below tolerance: {norm}"
            )
        else:
            self._world_direction = world_dir / norm

        # update the object transform matrix
        self._object_coordinate_transform = np.linalg.inv(
            self._world_coordinate_transform
        )

    def _append_world_transform(self, new_transform):
        self._world_coordinate_transform = np.matmul(
            new_transform, self._world_coordinate_transform
        )
        # update the functions in the watch list
        [fn() for fn in self.var_watchlist]

    def get_position(self):
        # check if the position is valid, if not update it and return
        if not self._pos_valid:
            self._world_origin = np.matmul(
                self._world_coordinate_transform, self._obj_origin
            )
            self._pos_valid = True
        return self._world_origin

    def get_orientation(self):
        # check if we need to update the direction vector
        if not self._dir_valid:
            world_dir = np.matmul(self._world_coordinate_transform, self._obj_direction)
            norm = linalg.norm(world_dir)
            if norm < 1e-7:
                raise ValueError(
                    f"Measured Norm of World primitives.Vector below tolerance: {norm}"
                )
            else:
                self._world_direction = world_dir / norm

        return self._world_direction

    def get_quaternion(self):
        # make a rotation object
        r = transform.Rotation.from_matrix(self._world_coordinate_transform[:-1, :-1])
        # return the quaternion
        return r.as_quat()

    def get_world_transform(self):
        """
        returns the 4x4 matrix that translates the object into world coordinate space. the returned matrix is a copy
        of the internal object and can be modified without changing the object's state.

        :return: a 4x4 matrix of type float representing the object transform
        """
        return copy.copy(self._world_coordinate_transform)

    def _get_object_transform(self):
        # if the object transform matrix is out of date, update it
        if not self._obj_transform_valid:
            self._object_coordinate_transform = np.linalg.inv(
                self._world_coordinate_transform
            )
            self._obj_transform_valid = True
        return self._object_coordinate_transform

    def get_object_transform(self):
        """
        returns the 4x4 matrix that translates the world coordinates into object space. the returned matrix is a copy
        of the internal object and can be modified without changing the object's state.

        :return: a 4x4 numpy array of float
        """
        return copy.copy(self._get_object_transform())

    def to_object_coordinates(self, coordinates):
        return np.matmul(self._get_object_transform(), coordinates)

    def to_world_coordinates(self, coordinates):
        return np.matmul(self._world_coordinate_transform, coordinates)

    # Movement operations
    def move(self, x=0, y=0, z=0):
        tx = self._transfer_matrix()
        tx[:-1, -1] = (x, y, z)
        # update the transform matrix
        self._append_world_transform(tx)
        return self

    def move_x(self, movement):
        self.move(x=movement)
        return self

    def move_y(self, movement):
        self.move(y=movement)
        return self

    def move_z(self, movement):
        self.move(z=movement)
        return self

    # Scale operations
    def scale(self, x=1, y=1, z=1):
        # for now we're only going to allow positive scaling
        if x < 0 or y < 0 or z < 0:
            raise ValueError("Negative values for scale operations are prohibited")

        tx = np.diag((x, y, z, 1))
        self._append_world_transform(tx)
        return self

    def scale_x(self, scale_val):
        return self.scale(x=scale_val)

    def scale_y(self, scale_val):
        return self.scale(y=scale_val)

    def scale_z(self, scale_val):
        return self.scale(z=scale_val)

    def scale_all(self, scale_val):
        return self.scale(scale_val, scale_val, scale_val)

    # Rotation Operations
    def rotate_x(self, angle, units="deg"):
        sin_a, cos_a = self._sin_cos(angle, units)
        tx = self._transfer_matrix()
        tx[1, 1] = cos_a
        tx[2, 2] = cos_a
        tx[1, 2] = -sin_a
        tx[2, 1] = sin_a

        self._append_world_transform(tx)
        return self

    def rotate_y(self, angle, units="deg"):
        sin_a, cos_a = self._sin_cos(angle, units)
        tx = self._transfer_matrix()
        tx[0, 0] = cos_a
        tx[2, 2] = cos_a
        tx[2, 0] = -sin_a
        tx[0, 2] = sin_a

        self._append_world_transform(tx)
        return self

    def rotate_z(self, angle, units="deg"):
        sin_a, cos_a = self._sin_cos(angle, units)
        tx = self._transfer_matrix()
        tx[0, 0] = cos_a
        tx[1, 1] = cos_a
        tx[0, 1] = -sin_a
        tx[1, 0] = sin_a

        self._append_world_transform(tx)
        return self

    def transform(self, transform_matrix):
        """
        applies the transform matrix to the object. Can contain rotation, translation, scale, and sheer operations. Can
        be chained with other transform operations.

        :param transform_matrix:
        :return: self
        """
        self._append_world_transform(transform_matrix)
        return self


class ObjectGroup(WorldObject, collections.UserList):
    def __init__(self, *args, **kwargs):
        # inheriting from UserList will make a public variable called data in our group
        super().__init__(*args, **kwargs)

    def _append_world_transform(self, new_transform):
        super()._append_world_transform(
            new_transform
        )  # update the Groups world transform matrix

        # iterate over all surfaces in the group and update their transform matrices
        for surface in self.data:
            surface.transform(new_transform)


class Intersectable(WorldObject, abc.ABC):
    """
    Base class for a renderable object
    """

    _aobb: primitives.Cube
    _parent: WorldObject  # the parent is an additional world object that this object is tied to
    _normal_scale = 1  # a multiplier used when normals are inverted

    @abc.abstractmethod
    def intersect(self, rays):
        pass

    @property
    def bounding_box(self):
        return self._aobb

    def attach_to(self, parent_object: WorldObject) -> None:
        self._parent = parent_object
        self.var_watchlist += parent_object.var_watchlist

    def invert_normals(self):
        self._normal_scale = -1

    def reset_normals(self):
        self._normal_scale = 1

    @property
    def surface_ids(self) -> tuple:
        """
        Returns a tuple of surface ids and surface that make up the intersectable object, used for calculating
        intersections.
        """
        return ((self.get_id(), self),)

    @property
    def bounding_volume(self):
        return self._aobb


class TracerSurface(Intersectable, abc.ABC):
    surface: primitives.SurfacePrimitive

    def __init__(self, surface_args, material=BLACK, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._surface_primitive = type(self).surface(
            *surface_args
        )  # create a surface primitive from the provided args
        self.material = material

        # make a bounding volume to enclose the shape
        self._aobb = bounding_box(self._surface_primitive.bounding_points)
        self.var_watchlist.append(self._boundary_box_update_fn)

    def _boundary_box_update_fn(self):
        self._aobb = bounding_box(
            np.matmul(
                self._world_coordinate_transform,
                self._surface_primitive.bounding_points,
            )
        )

    def intersect(self, rays):
        """
        Intersect the set of rays with the surface, returns a 1d array of euclidean distances between the ray and
        surface. if the surface does not intersect -1 is returned instead
        :param rays:
        :return:
        """
        local_ray_set = np.matmul(
            self._get_object_transform(), np.atleast_3d(rays)
        )  # translate the rays into object space

        # get the hits matrix, which is mxn where n is the number of rays propagated
        hits = self._surface_primitive.intersect(local_ray_set)

        # if there is an aperture the hits have to be filtered
        # any hit that is not in the aperture is eliminated

        # filter out any negative hits, and then return the smallest, replacing np.inf with -1
        # hits = np.min(np.where(hits >= 0, hits, np.inf), axis=0)  # reduce the hits to a 1xn array of minimum hits
        # hits = np.where(np.isfinite(hits), hits, -1)  # mask the hits so that anywhere it's np.inf it's cast as -1

        return np.sort(hits, axis=0), np.full(
            hits.shape, self.get_id()
        )  # return the hits matrix and the intersecting surface

    def shade(self, rays: np.ndarray, distances: np.ndarray, **kwargs) -> np.ndarray:
        """
        propagates a set of rays incident on the surface. returns a new set of rays and refractive indices representing
        the rays after transmitting/reflecting off the surface.
        """

        # get the normals for the surface at the coordinate
        coordinates = rays[0] + distances * rays[1]
        normals = self.get_world_normals(
            coordinates
        )  # get the world normals of the surface
        pixel_values = self.material.shade(
            np.stack((coordinates, rays[1]), axis=0), normals, **kwargs
        )
        return pixel_values

    def get_world_normals(self, positions):
        """
        returns the normal vectors of the surface for the given positions

        :param positions: a 4xn array of homogeneous point coordinates.
        :return: a 4xn array of unit vectors representing the surface's normal vector at the corresponding input
            position
        """
        local_ray_set = self.to_object_coordinates(positions)
        local_normals = self._surface_primitive.normal(local_ray_set)
        world_normals = np.matmul(self._get_object_transform().T, local_normals)
        world_normals[
            -1
        ] = 0  # wipe out any w fields caused by the transpose of the transform
        world_normals /= np.linalg.norm(world_normals, axis=0)
        return (
            world_normals * self._normal_scale
        )  # return the normals, flipped if the object has them inverted

    @property
    def primitive(self) -> primitives.SurfacePrimitive:
        return self._surface_primitive


class Sphere(TracerSurface):
    surface = primitives.Sphere

    def __init__(self, radius, material=BLACK, *args, **kwargs):
        super().__init__(surface_args=(radius,), material=material, *args, **kwargs)


class Cylinder(TracerSurface):
    surface = primitives.Cylinder

    def __init__(
        self, radius, min_height=-1, max_height=-1, material=BLACK, *args, **kwargs
    ):
        super().__init__(
            surface_args=(radius, min_height, max_height),
            material=material,
            *args,
            **kwargs,
        )


class Paraboloid(TracerSurface):
    surface = primitives.Paraboloid

    def __init__(self, focus=1, height=1, material=BLACK, *args, **kwargs):
        super().__init__(
            surface_args=(focus, height), material=material, *args, **kwargs
        )


class XYPlane(TracerSurface):
    surface = primitives.Plane

    def __init__(self, width=2, length=2, material=BLACK, *args, **kwargs):
        super().__init__(
            surface_args=(width, length), material=material, *args, **kwargs
        )


class Cuboid(TracerSurface):
    surface = primitives.Cube

    def __init__(
        self, l_corner=(-1, -1, -1), r_corner=(1, 1, 1), material=BLACK, *args, **kwargs
    ):
        super().__init__(
            surface_args=(l_corner, r_corner), material=material, *args, **kwargs
        )

    @classmethod
    def from_sides(cls, x=1, y=1, z=1, **kwargs):
        corners = np.tile((-0.5, 0.5), (3, 1)).T * np.array((x, y, z))
        return cls(*corners, **kwargs)

    @classmethod
    def from_length(cls, length, **kwargs):
        corners = length * np.tile((-0.5, 0.5), (3, 1)).T
        return cls(*corners, **kwargs)


class Cylinder(TracerSurface):
    surface = primitives.Cylinder

    def __init__(
        self, radius=1, min_height=-1, max_height=1, material=BLACK, *args, **kwargs
    ):
        super().__init__(
            surface_args=(radius, min_height, max_height),
            material=material,
            *args,
            **kwargs,
        )


class OrthographicCamera(WorldObject):
    """
    A camera oriented along the z-axis pointing along the x-axis
    """

    def __init__(
        self, h_pixel_count: int, h_width: float, aspect_ratio: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._h_pixels = h_pixel_count
        self._h_width = h_width
        self._v_width = aspect_ratio * h_width
        self._v_pixels = int(aspect_ratio * self._h_pixels)

    def get_resolution(self):
        return (self._h_pixels, self._v_pixels)

    def get_span(self):
        return (self._h_width, self._v_width)

    def generate_rays(self) -> np.ndarray:
        rays = self._local_ray_generation()
        rays = np.matmul(
            self._world_coordinate_transform, rays
        )  # transform rays to world space
        rays[1] /= np.linalg.norm(rays[1], axis=0)  # normalize the direction vector
        return rays

    def _local_ray_generation(self) -> np.ndarray:
        h_steps = np.linspace(self._h_width / 2, -self._h_width / 2, self._h_pixels)
        v_steps = np.linspace(self._v_width / 2, -self._v_width / 2, self._v_pixels)

        rays = primitives.bundle_of_rays(self._h_pixels * self._v_pixels)
        ys, zs = np.meshgrid(h_steps, v_steps)
        rays[0, 1] = ys.reshape(-1)
        rays[0, 2] = zs.reshape(-1)
        rays[1, 0] = 1  # orient rays to face the positive x-axis

        return rays
