import numpy as np
import numpy.linalg as linalg
import itertools
import abc

from tinygfx.g3d.operations import binomial_root, element_wise_dot


def _corners_to_cube_points(min_corner, max_corner):
    axis_spans = np.sort(np.vstack((min_corner[:3], max_corner[:3])), axis=0).T
    # the corner points are 8 points that make up the span of the cube
    corner_points = np.vstack([Point(x, y, z) for x, y, z in itertools.product(*axis_spans)]).T
    return corner_points


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


class SurfacePrimitive(abc.ABC):
    """
    SurfacePrimitive classes have abstract functions to calculate ray-object intersections and normals
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self.bounding_points = None  # set of points that fully enclose the surface

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
        # bounding points make up a cube with a side length of 2*radius
        self.bounding_points = _corners_to_cube_points((-radius, -radius, -radius), (radius, radius, radius))
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
        self.bounding_points = _corners_to_cube_points((0, -np.inf, -np.inf), (np.inf, np.inf, np.inf))

    def get_focus(self):
        return self._focus

    def intersect(self, rays):
        # this is similar to the line sphere intersection, only you start with |x-2f|^2=x_z^2 for the line intersection
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        # get the components of the polynomial root equation
        a = element_wise_dot(directions[1:], directions[1:], axis=0)
        b = 2 * (element_wise_dot(directions[1:], origins[1:]) - 2 * directions[0] * self._focus)

        c = element_wise_dot(origins[1:], origins[1:], axis=0) - origins[0] * 4 * self._focus

        # these points are at the origin and intersect with the sphere at t=0
        trivial_cases = np.isclose(c, 0)

        hits = binomial_root(a, b, c)
        # hits = np.where(trivial_cases, 0, hits) # replace trivial cases with zeros

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
        self.bounding_points = _corners_to_cube_points((0, -np.inf, -np.inf), (0, np.inf, np.inf))

    def intersect(self, rays):
        """
        a plane can have at most one intersection with a ray

        :param rays:
        :return:
        """
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        hits = np.where(np.logical_not(np.isclose(directions[0], 0)),
                        -origins[0] / (directions[0] + (directions[0] == 0)), np.inf)

        return np.atleast_2d(hits)

    def normal(self, intersections):
        # a plane has a trivial normal, it's the -x axis
        normals = np.zeros(intersections.shape)
        normals[0] = -1
        return normals


class Cube(SurfacePrimitive):
    """
    a axis aligned cube fully defined by its corners
    """

    def __init__(self, min_corner=(-1, -1, -1), max_corner=(1, 1, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the parent constructor

        # a 3x2 matrix that tracks the min and max values for each axis,
        self._axis_spans = np.sort(np.vstack((min_corner[:3], max_corner[:3])), axis=0).T
        # the corner points are 8 points that make up the span of the cube
        self.bounding_points = np.vstack([Point(x, y, z) for x, y, z in itertools.product(*self._axis_spans)]).T

    def intersect(self, rays):
        # So, if the minimum intersection of an axis exceeds the maximum intersection of a separate axis, rays do not
        # intersect the cube. This is kinda odd, maybe draw it out in 2D to help
        # but I'm pretty sure teh math checks out
        # it's pretty much saying that if you project teh vector along one of the coordinate axes, you leave the line
        # of the cube before the other axis starts the intersection
        padded_rays = np.atleast_3d(rays)

        # get the origins and directions
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        hits = np.full((6, origins.shape[-1]), -1, dtype=float)  # hit distance matrix
        new_hits = np.zeros((2, origins.shape[-1]))  # a matrix to store the next hits

        for axis in [0, 1, 2]:
            # if the vector does not travel in that direction they won't intersect
            is_zero = np.isclose(directions[axis], 0)
            # need a special case if the position is skew to an axis, have to know if the point is in teh projected
            # square
            skew_case_min = np.where(np.abs(origins[axis]) <= 1.0, -np.inf, np.inf)

            # now update the intersection point for each plane
            new_hits[0] = np.where(np.logical_not(is_zero),
                                   -(origins[axis] - self._axis_spans[axis, 0]) / (directions[axis] + is_zero),
                                   skew_case_min)

            new_hits[1] = np.where(np.logical_not(is_zero),
                                   -(origins[axis] - self._axis_spans[axis, 1]) / (directions[axis] + is_zero), np.inf)

            # now we need to sort the new hits so 0 is the min and 1 is the max
            new_hits = np.sort(new_hits, axis=0)

            # update the hits matrix with these new hits
            hits[axis] = new_hits[0]
            hits[3 + axis] = new_hits[1]

        cube_hits = np.zeros(
            new_hits.shape)  # cube hits will be a 2xn array of where the points actually intersect the cube
        cube_hits[0] = np.max(hits[:3], axis=0)  # first hit is the max value of the minimums
        cube_hits[1] = np.min(hits[3:], axis=0)  # second hit is the min value of the maximums

        # if the min is larger than the max the ray missed the cube and should be replaced with inf
        cube_hits = np.where(cube_hits[0] < cube_hits[1], cube_hits, np.inf)

        # return the cube hits
        return cube_hits

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


class Cylinder(SurfacePrimitive):
    """
    A cylinder with a radius of 1 in the ZY plane, extending from +1 to -1
    """

    def __init__(self, radius=1, infinite=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._radius = radius  # this is the sphere's radius in object space, it can be manipulated in world space with
        # scale and transform operations
        self._infinite = infinite  # whether or not the cylinder has end caps

    def get_radius(self):
        """
        Get the sphere's radius in object space. The apparent radius in world space may be different due to object
            transforms

        :return: the sphere's object space radius
        :rtype: float
        """

        return self._radius

    def intersect(self, rays):
        """
        For Cylinder intersections first find the intersection point with an infinite cylinder, than check if
            the caps are intersected before or after the cylinder
        :param rays:
        :return:
        """
        padded_rays = np.atleast_3d(rays)
        origins = padded_rays[0, :-1]  # should be a 3xn array of points
        directions = padded_rays[1, :-1]  # should be a 3xn array of vectors

        # use the ray projected into the XY plane for the side of the cylinder intersection
        origins_2d = origins[:-1]
        directions_2d = directions[:-1]

        # calculate the a,b, and c of the polynomial roots equation
        a = element_wise_dot(directions_2d, directions_2d,
                             axis=0)  # a must be positive because it's the squared magnitude
        b = 2 * element_wise_dot(directions_2d, origins_2d, axis=0)  # be can be positive or negative
        c = element_wise_dot(origins_2d, origins_2d, axis=0) - self._radius ** 2  # c can be positive or negative

        # calculate the sidewall hits
        hits = np.zeros((4, directions.shape[-1]))
        sidewall_hits = np.sort(binomial_root(a, b, c), axis=0)  # have to sort the roots for intersections

        # calculate where the cap is hit
        if not self._infinite:
            cap_hits = np.tile((-np.inf, np.inf), (origins.shape[-1], 1)).T  # by default assume the cap hits as +/- inf

            # if the vector does not travel in the z-direction they'll never intersect
            is_zero = np.isclose(directions[2], 0)

            # need a special case if the position is skew to an axis, have to know if the point is in the projected
            # square
            skew_case_min = np.full((2, directions.shape[-1]), np.inf)
            skew_case_min[0] = np.where(np.abs(origins[2]) <= 1, -np.inf, np.inf)

            cap_hits = -np.vstack(((origins[2] + 1), origins[2] - 1)) / (directions[2] + is_zero)
            cap_hits = np.where(np.logical_not(is_zero), cap_hits, skew_case_min)
            cap_hits = np.sort(cap_hits, axis=0)

            all_hits = np.vstack((sidewall_hits[0], cap_hits[0], sidewall_hits[1], cap_hits[1]))
            # now calculate the max of the nearest hits, and min of the furthest hits to find where we intersect
            # the cylinder

            cylinder_hits = np.vstack((
                np.max(all_hits[:2], axis=0),
                np.min(all_hits[2:], axis=0))
            )
            cylinder_hits = np.where(cylinder_hits[0] < cylinder_hits[1], cylinder_hits, np.inf)
            return cylinder_hits
        else:
            return sidewall_hits

    def normal(self, intersections):
        pass