import numpy as np
import numpy.linalg as linalg
import copy
import scipy.spatial.transform as transform
import collections
import abc

def smallest_positive_root(a,b,c):
    """
    finds the 2nd order polynomial roots given the equations a*x^2 + b*x + c = 0, returns the smallest root > 0, and
    np.inf if a root does not satisfy that condition. Note that if a has a value of zero, it will be cast as 1 to avoid
    a divide by zero error. it's up the user to filter out these results before/after execution (using np.where etc.).

    :param a: coefficients for the second order of the polynomial
    :param b: coefficients for the first order of the polynomial
    :param c: coefficients for the zeroth order of the polynomial
    :return: an array of length n with the smallest positive root for the array
    """
    disc = b ** 2 - 4 * a * c  # find the discriminant
    root = np.sqrt(np.maximum(0, disc)) # the square root of the discriminant protected from being nan
    polyroots = np.array(((-b + root), (-b - root))) / (2 * a+np.isclose(a,0))  # the positive element of the polynomial root

    # want to keep the smallest hit that is positive, so if hits[1]<0, just keep the positive hit
    nearest_hit = np.where(polyroots[1] >= 0, np.amin(polyroots, axis=0), polyroots[0])
    return np.where(np.logical_and(disc >= 0, nearest_hit >= 0), nearest_hit, np.inf)

def element_wise_dot(mat_1, mat_2, axis=0):
    """
    calculates the row-wise/column-wise dot product two nxm matrices

    :param mat_1: the first matrix for the dot product
    :param mat_2: the second matrix for the dot product
    :param axis: axis to perform the dot product along, 0 or 1
    :return: a 1D array of length m for axis 0 and n for axis 1
    """

    einsum_string = "ij,ij->j"
    if axis == 1:
        einsum_string = 'ij,ij->i'

    return np.einsum(einsum_string, mat_1, mat_2)


def reflect(vectors, normals):
    """
    reflects an array of vectors by a normal vector.

    :param vectors: a mxn array of vectors, where each column corresponds to one vector
    :param normals: a mxn array of unit-normal vectors or a 4x0 single normal vector. If only one normal is provided,
        every vector is reflected across that normal
    :return: an mxn array of reflected vector
    """
    # if we got 2x 1x4 arrays it's a basic case
    if vectors.ndim == 1 and normals.ndim == 1:
        return vectors - normals * 2 * vectors.dot(normals)

    # if only one normal was provided reflect every vector off of it
    elif normals.ndim == 1:
        dots = np.einsum('ij,i->j', vectors, normals)
        return vectors - 2 * np.tile(normals, (vectors.shape[1], 1)).T * dots

    # otherwise it's full blown matrix math
    else:
        dots = element_wise_dot(vectors, normals, axis=0)
        return vectors - 2 * normals * dots

    return


def bundle_of_rays(n_rays):
    """
    returns a 2x4xn_rays array of ray objects where rays[0] are 4xn_ray homogoneous points(0,0,0), and rays[1] are
    4xn_ray homogeneous vectors(0,0,0)

    :param n_rays:
    :return:
    """
    rays = np.zeros((2,4,n_rays))
    rays[0,-1] = 1
    return rays
def bundle_rays(rays):
    return np.stack(rays, axis=2)


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


class Ray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # creates an array with the homogeneous coordinates
        obj = np.zeros((2, 4), dtype=np.float32).view(cls)
        return obj

    def __init__(self, origin=Point(), direction=Vector(1, 0, 0)):
        # assign the magnitude and direction
        self[0] = origin
        self[1] = direction

    @property
    def origin(self):
        return self[0].view(HomogeneousCoordinate)

    @origin.setter
    def origin(self, new_origin):
        self[0] = new_origin

    @property
    def direction(self):
        return self[1].view(HomogeneousCoordinate)

    @direction.setter
    def direction(self, new_direction):
        self[1] = new_direction


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


class SurfacePrimitive(abc.ABC):
    """
    SurfacePrimitive classes have abstract functions to calculate ray-object intersections and normals
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

    @abc.abstractmethod
    def intersect(self, rays):
        """
        calculates the intersection of an array of rays with the surface, returning a 1-D array with the hit distance
            for each ray. Rays are represented by the vector equation x(t) = o + t*v, where o is the point-origin,
            and v is the vector direction.

        :param rays: a 2x4xn numpy array where n is the number of rays being projected. For each slice of rays the first
            row is the ray's origin, and the second row is the direction. Both should be represented as homogeneous
            coordinates
        :return: an array of the parameter t from the ray equation where the ray intersects the object.
        :rtype: 1-D numpy array of np.float32
        """
        pass

    @abc.abstractmethod
    def normal(self, intersections):
        """
        calculates the normal of a sphere at each point in an array of coordinates. It is assumed that the points lie
            on the surface of the object, as this is not verified during calculation.

        :param intersections: a 4xn array of homogeneous points representing a point on the sphere.
        :type intersections: 4xn numpy of np.float32
        :return: an array of vectors representing the unit normal at each point in intersection
        :rtype:  4xn numpy array of np.float32
        """


class Sphere(SurfacePrimitive):
    def __init__(self, radius=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._radius = radius  # this is the sphere's radius in object space, it can be manipulated in world space with
        # scale and transform operations

    def get_radius(self):
        """
        Get the sphere's radius in object space. The apparent radius in world space may be different due to object
            transforms

        :return: the sphere's object space radius
        :rtype: float
        """

        return self._radius

    def intersect(self, rays):
        # a sphere intersection requires the discriminant of the 2nd order roots equation is positive
        # if the discriminant is zero, the ray is tangent to the sphere
        # if it's negative, there's not intersection
        # otherwise it intersects the sphere at two points
        padded_rays = np.atleast_3d(rays)
        # step one is transform the rays into object space -- rays should always exist in world space!
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        # calculate the a,b, and c of the polynomial roots equation
        a = element_wise_dot(directions, directions, axis=0)  # a must be positive because it's the squared magnitude
        b = 2 * element_wise_dot(directions, origins, axis=0)  # be can be positive or negative
        c = element_wise_dot(origins, origins, axis=0) - self._radius ** 2  # c can be positive or negative

        # calculate the discriminant, but override the sqrt if it would result in a negative number
        disc = b ** 2 - 4 * a * c
        root = np.sqrt(np.maximum(0, disc))
        hits = np.array(((-b + root), (-b - root))) / (2 * a)  # the positive element of the polynomial root

        # want to keep the smallest hit that is positive, so if hits[1]<0, just keep the positive hit
        nearest_hit = np.where(hits[1] >= 0, np.amin(hits, axis=0), hits[0])
        hit_array = np.where(np.logical_and(disc >= 0, nearest_hit >= 0), nearest_hit, np.inf)

        return hit_array

    def normal(self, intersections):
        # calculates the normal of a transformed sphere at a point XYZ
        # the normal is the distance from the origin to the point (assuming they were on the sphere)
        # to get it back into world coordinates it has to be multiplied by the transpose of the world coord transform
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals

        # if only one point was provided, wrap it and transpose it
        padded_intersections = intersections
        dims = intersections.ndim
        if dims != 2:
            if dims == 1:
                padded_intersections = np.atleast_2d(intersections).T
            else:
                raise AttributeError(f"Argument intersections has too many dimensions, expect 1 or 2, got {dims}")

        # for a sphere the normal is pretty easy, just scale the coordinate
        world_normals = padded_intersections.copy()
        world_normals[-1] = 0  # wipe out the point coordinate aspect
        world_normals /= np.linalg.norm(world_normals, axis=0)

        # if a 1d array was passed, transpose it and strip a dimension
        return world_normals if dims == 2 else world_normals.T[0]


class Paraboloid(SurfacePrimitive):
    """
    A Spherical parabola with focus at point (0,0,f). The directrix plane is the YZ plane
    """

    def __init__(self, focus=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._focus = focus

    def get_focus(self):
        return self._focus

    def intersect(self, rays):
        # this is similar to the line sphere intersection, only you start with |x-2f|^2=x_z^2 for the line intersection
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        # prebuild the hits array with np.inf, which is the default if a hit does not exist
        hits = np.full(padded_rays.shape[-1], np.inf)

        # get the components of the polynomial root equation
        a = element_wise_dot(directions[1:], directions[1:], 0)
        b = 2*(element_wise_dot(directions[1:], origins[1:]) - 2 * directions[0] * self._focus)

        c = element_wise_dot(origins[1:], origins[1:], axis=0) - origins[0]*4*self._focus

        disc = b**2 - 4*a*c # calculate the discriminant for the polynomial roots equation

        # trivial cases are where c=0
        # these points are at the origin and intersect with the sphere at t=0
        trivial_cases = np.isclose(c, 0)

        # linear cases are where a=0, this cannot be solved by the polyroots equation since the denominator blows up
        # trivial cases get excluded from the linear case set
        linear_cases = np.logical_and(np.isclose(a, 0), np.logical_not(trivial_cases))

        # all other cases must have double roots
        dbl_root_cases = np.logical_not(np.logical_or(trivial_cases, linear_cases))

        # update the hits array based on the cases
        hits = np.where(trivial_cases, 0, hits)
        # so trivial_cases is in the denominator so there's not a divide by zero error if b==0, hacky but works
        hits = np.where(linear_cases, -c/(b+(b==0)), hits)

        hits = np.where(dbl_root_cases, smallest_positive_root(a, b, c), hits)

        # retun the hits array
        return hits


    def normal(self, intersections):
        # normals are pretty simple and are of the form <-1,y/2f,x/2f>
        # create the normals array
        normals = intersections.copy()
        normals[0] = -2*self._focus
        return normals/np.linalg.norm(normals, axis=0)
