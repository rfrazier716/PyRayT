import numpy as np
import numpy.linalg as linalg
import copy
import scipy.spatial.transform as transform
import collections


class CountedObject(object):
    _object_count = 0  # keeps track of how many objects of each type exist
    _next_object_id = 0  # the next id number to use when a new object is initialized

    @classmethod
    def _increment(cls):
        # increments the classes object count and returns the object number of the instance that called it
        this_id = cls._next_object_id
        cls._object_count += 1
        cls._next_object_id += 1
        return this_id

    @classmethod
    def _decrement(cls):
        cls._object_count -= 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._id = self._increment()  # object specific id number for this instance

    def get_id(self):
        """
        Gets the class specific object ID number for this instance.

        :return: the unique class id of the object instance
        :rtype: int
        """
        return self._id

    @classmethod
    def get_count(cls):
        """
        Gets the current count of objects
        :return: the current number of objects in memory
        :rtype: int
        """
        return cls._object_count

    def __del__(self):
        type(self)._decrement()  # reduce the object count by 1


class HomogeneousCoordinate(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # creates an array with the homogeneous coordinates
        obj = np.zeros(4, dtype=np.float32).view(cls)
        return obj

    def __init__(self, x=0, y=0, z=0, w=0):
        # assign initialization
        self[0] = x
        self[1] = y
        self[2] = z
        self[3] = w

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, x):
        self[0] = x

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, y):
        self[1] = y

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, z):
        self[2] = z

    @property
    def w(self):
        return self[3]

    @w.setter
    def w(self, w):
        self[3] = w


class Point(HomogeneousCoordinate):
    def __init__(self, x=0, y=0, z=0, *args, **kwargs):
        super().__init__(x, y, z, 1)  # call the homogeneous coordinate constructor, points have a coord of 1


class Vector(HomogeneousCoordinate):
    def __init__(self, x=0, y=0, z=0, *args, **kwargs):
        super().__init__(x, y, z, 0)


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
            cos_a = np.cos(angle * np.pi / 180.)
            sin_a = np.sin(angle * np.pi / 180.)

        elif format == "rad":
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

        else:
            raise ValueError(f"{format} is not a valid option for angle units")

        return sin_a, cos_a

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj_origin = Point(0, 0, 0)  # position in object space
        self._obj_direction = Vector(0, 0, 1)  # direction in object space

        self._world_position = Point(0, 0, 0)  # the objects position in world space
        self._world_direction = Vector(0, 0, 1)  # the objects direction in world space

        # Flags that get set to false whenever the transform matrix has been updated
        self._dir_valid = True
        self._pos_valid = True
        self._obj_transform_valid = True

        self._world_coordinate_transform = np.identity(4,
                                                       dtype=np.float32)  # transform matrix from object to world space
        self._object_coordinate_transform = np.identity(4, dtype=np.float32)

    def _append_world_transform(self, new_transform):
        self._world_coordinate_transform = np.matmul(new_transform, self._world_coordinate_transform)
        self._dir_valid = False
        self._pos_valid = False
        self._obj_transform_valid = False

    def get_position(self):
        # check if the position is valid, if not update it and return
        if not self._pos_valid:
            self._world_position = np.matmul(self._world_coordinate_transform, self._obj_origin)
            self._pos_valid = True
        return self._world_position

    def get_orientation(self):
        # check if we need to update the direction vector
        if not self._dir_valid:
            world_dir = np.matmul(self._world_coordinate_transform, self._obj_direction)
            norm = linalg.norm(world_dir)
            if norm < 1E-7:
                raise ValueError(f"Measured Norm of World Vector below tolerance: {norm}")
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
            self._object_coordinate_transform = np.linalg.inv(self._world_coordinate_transform)
            self._obj_transform_valid = True
        return self._object_coordinate_transform

    def get_object_transform(self):
        """
        returns the 4x4 matrix that translates the world coordinates into object space. the returned matrix is a copy
        of the internal object and can be modified without changing the object's state.

        :return: a 4x4 numpy array of float
        """
        return copy.copy(self._get_object_transform())

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
        super()._append_world_transform(new_transform)  # update the Groups world transform matrix

        # iterate over all surfaces in the group and update their transform matrices
        for surface in self.data:
            surface.transform(new_transform)
