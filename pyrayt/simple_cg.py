import itertools
import numpy as np
import numpy.linalg as linalg
import copy
import scipy.spatial.transform as transform
import collections
import abc


def smallest_positive_root(a, b, c):
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
    root = np.sqrt(np.maximum(0, disc))  # the square root of the discriminant protected from being nan
    polyroots = np.array(((-b + root), (-b - root))) / (
            2 * a + np.isclose(a, 0))  # the positive element of the polynomial root

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
    # if a regular array was passed just do the dot product
    if mat_1.ndim == 1:
        return mat_1.dot(mat_2)

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


def refract(vectors, normals, n1, n2, n_global=1):
    """
    calculates the refracted vector at an interface with index mismatch of n1 and n2. If the angle between the normal
        and the vector is  <90 degrees, the vector is exiting the medium, and the global refractive index value is used
        instead. this is analygous to a ray exiting a glass interface and entering air

    :param vectors: the ray transmission vectors in the current medium
    :param normals: the normals to the surface at the point of intersection with the ray
    :param n1: refractive index of the medium the ray is currently transmitting through
    :param n2: refractive index of the medium the ray is entering
    :param n_global: the world refractive index to use if the ray is exiting the medium

    :return: a 4xn array of homogeneous vectors representing the new transmission direction of the vectors after the
        medium
    """
    cos_theta1_p = element_wise_dot(vectors, normals, axis=0)  # the angle dotted with the normals
    cos_theta1_n = element_wise_dot(vectors, -normals, axis=0)  # the vector dotted with the negative normals

    # anywhere that cos_theta1_p>0, we're exiting the medium and the n2 value should be updated with the global index
    n2_local = np.where(cos_theta1_p > 0, n_global, n2)
    normals = np.where(cos_theta1_p > 0, -normals, normals)  # update normals so they always are along ray direction
    r = n1 / n2_local  # the ratio of refractive indices

    # we want to keep the positive values only, which is the angle between the norm and the vector
    cos_theta1 = np.where(cos_theta1_p > 0, cos_theta1_p, cos_theta1_n)

    # see https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form for more details on using this
    radicand = 1 - (r ** 2) * (1 - cos_theta1 ** 2)  # radicand of the square root function
    cos_theta2 = np.sqrt(
        np.maximum(0, radicand))  # pipe in a zero there so that we don't take the sqrt of a negative number

    # return the refracted or reflected ray depending on the radicand
    refracted = np.where(radicand > 0, r * vectors + (r * cos_theta1 - cos_theta2) * normals,
                         vectors + 2 * cos_theta1 * normals)
    refracted /= np.linalg.norm(refracted, axis=0)  # normalize the vectors

    n_refracted = np.where(radicand > 0, n2_local, n1)  # the refracted index is the original material if TIR
    return refracted, n_refracted


def bundle_of_rays(n_rays):
    """
    returns a 2x4xn_rays array of ray objects where rays[0] are 4xn_ray homogoneous points(0,0,0), and rays[1] are
    4xn_ray homogeneous vectors(0,0,0)

    :param n_rays:
    :return:
    """
    rays = np.zeros((2, 4, n_rays))
    rays[0, -1] = 1
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
        obj = np.zeros(4, dtype=float).view(cls)
        return obj

    def __init__(self, x=0, y=0, z=0, w=0):
        # assign initialization
        self[0] = x
        self[1] = y
        self[2] = z
        self[3] = w

    def normalize(self):
        self[:-1] /= np.linalg.norm(self[:-1])
        return self

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
        obj = np.zeros((2, 4), dtype=float).view(cls)
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
        self.rays = bundle_of_rays(n_rays)
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
                                                       dtype=float)  # transform matrix from object to world space
        self._object_coordinate_transform = np.identity(4, dtype=float)

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
        :rtype: 1-D numpy array of float
        """
        pass

    @abc.abstractmethod
    def normal(self, intersections):
        """
        calculates the normal of a sphere at each point in an array of coordinates. It is assumed that the points lie
            on the surface of the object, as this is not verified during calculation.

        :param intersections: a 4xn array of homogeneous points representing a point on the sphere.
        :type intersections: 4xn numpy of float
        :return: an array of vectors representing the unit normal at each point in intersection
        :rtype:  4xn numpy array of float
        """
        pass


class Shape2D(abc.ABC):
    """
    A 2D shape that exist in the XY Plane
    :param object:
    :return:
    """

    @abc.abstractmethod
    def point_in_shape(self, points: np.ndarray) -> np.ndarray:
        """
        Calculates if the points are in
        :param points:
        :return:
        """


class Disk(Shape2D):
    def __init__(self, radius=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._radius = radius

    @classmethod
    def from_diameter(cls, diameter: float) -> "Disk":
        """
        Creates a disk object from the diameter
        :param diameter:
        :return:
        """
        return cls(diameter / 2)

    def point_in_shape(self, points: np.ndarray) -> np.ndarray:
        # any point with a norm <1 is
        return np.linalg.norm(points, axis=0) <= self._radius


class Rectangle(Shape2D):
    def __init__(self, x_length=2, y_length=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x_length = x_length
        self._y_length = y_length

    def point_in_shape(self, points: np.ndarray) -> np.ndarray:
        single_dim = points.ndim == 1
        if single_dim:
            points = np.atleast_2d(points).T

        in_shape = np.all(np.abs(points) <= np.tile((self._x_length / 2, self._y_length / 2), (points.shape[-1], 1)).T,
                          axis=0)

        return in_shape[0] if single_dim else in_shape


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

        # shape should return entire hits array, but masked with np.inf for invalid discriminants
        valid_hits = np.where(disc >= 0, hits, np.inf)
        return valid_hits

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
        b = 2 * (element_wise_dot(directions[1:], origins[1:]) - 2 * directions[0] * self._focus)

        c = element_wise_dot(origins[1:], origins[1:], axis=0) - origins[0] * 4 * self._focus

        disc = b ** 2 - 4 * a * c  # calculate the discriminant for the polynomial roots equation

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
        hits = np.where(linear_cases, -c / (b + (b == 0)), hits)

        hits = np.where(dbl_root_cases, smallest_positive_root(a, b, c), hits)

        # retun the hits array
        return hits

    def normal(self, intersections):
        # normals are pretty simple and are of the form <-1,y/2f,x/2f>
        # create the normals array
        normals = intersections.copy()
        normals[3] = 0  # wipe out the point setting
        normals[0] = -2 * self._focus
        return normals / np.linalg.norm(normals, axis=0)


class Plane(SurfacePrimitive):
    """
    A YZ-Plane defined at x=0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def intersect(self, rays):
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        hits = np.where(np.logical_not(np.isclose(directions[0], 0)),
                        -origins[0] / (directions[0] + (directions[0] == 0)), np.inf)
        positive_hits = np.where(hits >= 0, hits, np.inf)

        return positive_hits

    def normal(self, intersections):
        # a plane has a trivial normal, it's the -x axis
        normals = np.zeros(intersections.shape)
        normals[0] = -1
        return normals


class Cube(SurfacePrimitive):
    """
    a cube with one corner at (-1,-1,-1) and the opposite corner at (1,1,1)
    """

    def intersect(self, rays):
        # a cube is pretty much 6 plane that intersect. steps for the intersection are then:
        # - Find the intersection distance and point for each plane that makes up the cube surface
        # - eliminate intersection points that are not within -<1,1,1> and <1,1,1>
        # - take the min along the axis of the 6 planes to find the nearest intersection hitting the plane
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        hits = np.full((6, origins.shape[-1]), -1, dtype=float)  # hit distance matrix
        # matrix tracking xyz where each ray hits each of the 6 planes making up cube

        # project into the xz,yz, and xy planes

        for axis in [0, 1, 2]:
            # if the vector does not travel in that direction they won't intersect
            is_zero = np.isclose(directions[axis], 0)

            # now update the intersection point for each plane
            hits[2 * axis] = np.where(np.logical_not(is_zero),
                                      -(origins[axis] + 1) / (directions[axis] + is_zero), -1)

            hits[2 * axis + 1] = np.where(np.logical_not(is_zero),
                                          -(origins[axis] - 1) / (directions[axis] + is_zero), -1)

        axis_intersections = np.tile(origins, (6, 1, 1)) + np.tile(hits, 3).reshape(6, 3,
                                                                                    padded_rays.shape[-1]) * directions

        # now we want to reduce it to a 2D array of points where all points lie on the unit cube,
        # this is where the abs of every point is <1
        abs_intersection = np.abs(axis_intersections)  # the absolute value of the intersection
        cube_hits = np.all(np.logical_or(
            abs_intersection <= 1.0,
            np.isclose(abs_intersection, 1.0)  # this last part is here for floating pointer errors being slightly >1
        ), axis=1)
        cube_hits = np.where(hits > 0, cube_hits, False)

        # next need to find the smallest positive valued hits
        nearest_hits = np.min(np.where(np.logical_and(hits > 0, cube_hits), hits, np.inf), axis=0)

        # return the smallest hits
        return nearest_hits

    def normal(self, intersections):
        # for normals you find the component of the point with the largest value, since the cube can only exist from
        # -1 to 1

        # the normal coord is the one that's closest to one
        normal_mask = np.isclose(np.abs(intersections), 1.0)
        normals = np.sign(intersections) * normal_mask
        normals[-1] = 0  # wipe out the point homogenous coordinate
        normals /= np.linalg.norm(normals, axis=0)

        # if a 1d array was passed, transpose it and strip a dimension
        return normals
