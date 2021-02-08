import abc
import adpd.packaging.simple_cg as cg
import numpy as np
import json
import copy

OPTICA_MATERIALS = {"Silicon"}


class _Aperture(object):
    def __init__(self, *args, **kwargs):
        self.shape = tuple()
        self.offset = cg.Point(0, 0, 0)


def to_mathematica_array(np_array):
    return np.array2string(np.asarray(np_array), precision=6, separator=',').replace("[", "{").replace("]",
                                                                                                       "}").replace(
        '\n', '')


def to_mathematica_rules(arg_dict):
    """
    Converts a dict of keyword arguments into a list of mathematica rules

    :param arg_dict:
    :return: str
    """
    options_associations = []
    for key, value in arg_dict.items():
        # if the value is a string we pass it like normal
        if isinstance(value, str):
            encoded_value = value

        # if the value has length, assume it's an array and pass the converted expression
        elif hasattr(value, '__len__'):
            encoded_value = to_mathematica_array(value)

        # otherwise assume it's a plain ole' float
        else:
            encoded_value = f"{value:.03f}"

        options_associations.append(key + " -> " + encoded_value)
    return "{" + ", ".join(options_associations) + "}"


class NamedObject(object):
    def __init__(self, name="NamedObject", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    def get_name(self):
        return self._name

    def __str__(self):
        return self._name


class RenderObject(cg.WorldObject, NamedObject, abc.ABC):
    """
    Render Objects have abstract functions to calculate ray-object intersections and normals
    """

    def __init__(self, name="RenderObject", *args, **kwargs):
        super().__init__(name, *args, **kwargs)  # call the next constructor in the MRO

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


class OpticaExportable(cg.WorldObject):
    _optica_function_call = "OpticaSurface"  # the label for the shape, class specific

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optica_options_dict={} # create a dict to hold optica options

    def create_optica_function(self):
        rules_command = self._create_rules_list()  # create the rules list
        obj_declaration = self._surface_id + " = " + self._get_optica_obj_string() + ';'  # create the object string
        move_array = to_mathematica_array(self.get_position()[:3])
        rotation_array = to_mathematica_array(self._world_coordinate_transform[:3, :3])
        move_command = self._name + " = Move[{label}, {move}, {rot}];".format(
            label=self._surface_id,
            move=move_array,
            rot=rotation_array
        )
        return "\n".join([rules_command, obj_declaration, move_command])

    def _get_optica_obj_string(self):
        func_name, func_arguments = self._create_optica_function_arguments()
        return func_name + "[" + func_arguments + ", " + self._opts_string + "]"

    @abc.abstractmethod
    def _create_optica_function_arguments(self):
        """
        Returns the name of the function and a string of arguments to define it
        :return:
        """
        return "OpticaShape", "{0,0}, 10"

    def _create_rules_list(self):
        rules = to_mathematica_rules(self._optica_options_dict)  # create a list of rules from the options dict
        rules_command = self._opts_string + " = " + rules + ";"
        return rules_command

    def _create_aperture_string(self):
        # if the aperture isn't a single number, turn it into a string
        if hasattr(self._aperture.shape, '__len__'):
            aperture_str = to_mathematica_array(self._aperture.shape)
        else:
            aperture_str = f"{self._aperture.shape:.03f}"
        return aperture_str


class Apertured(cg.WorldObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # call the next constructor in the MRO

        self._aperture = _Aperture() # the object's aperture
        self.set_aperture_shape(1)  # make the aperture a circle with radius of 1 by default

    def set_aperture_shape(self, new_shape):
        self._aperture.shape = new_shape
        return self

    def set_aperture_offset(self, x_offset, y_offset):
        self._aperture.offset.x = x_offset
        self._aperture.offset.y = y_offset


class TracerSurface(OpticaExportable, Apertured, cg.WorldObject, NamedObject, abc.ABC):

    def __init__(self, name="surface", *args, **kwargs):
        super().__init__(name, *args, **kwargs)  # call the next constructor in the MRO
        self._surface_id = type(self)._optica_function_call + f"{self.get_id():04d}"
        self._opts_string = self._surface_id + "OPTS"

        # a dictionary that holds important parameters that will be used to construct a json string of the object
        self._json_repr = dict()
        self._json_repr["name"] = self._name  # add name to be exported
        self._json_repr["type"] = self._optica_function_call

    def get_label(self):
        return self._surface_id

    def add_custom_parameter(self, parameter_key: str, parameter_value):
        self._optica_options_dict[str(parameter_key)] = parameter_value

    def collect_parameters(self):
        """
        Returns a dict of key parameters that define the surface
        :return:
        """
        self._json_repr["position"] = tuple(self.get_position()[:3].astype(float))  # get the non-homogenous position
        self._json_repr["rotation"] = tuple(
            self.get_quaternion().astype(float))  # get the rotation represented as a quaternion
        self._json_repr.update(self._optica_options_dict)
        return copy.copy(self._json_repr)

    def to_json_string(self):
        """
        return a JSON string representing the object
        """
        return json.dumps(self.collect_parameters(), indent=2)

    def set_aperture_shape(self, new_shape):
        self._json_repr["aperture"] = new_shape
        return super().set_aperture_shape(new_shape)

    def set_aperture_offset(self, x_offset, y_offset):
        self._optica_options_dict["OffAxis"] = tuple(x_offset, y_offset)
        super().set_aperture_offset(x_offset, y_offset)


class ThickSurface(TracerSurface, abc.ABC):
    def __init__(self, thickness, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)  # call the next constructor in the MRO
        self._thickness = thickness
        self._json_repr["thickness"] = self._thickness

    def set_thickness(self, new_thickness):
        self._thickness = new_thickness
        self._json_repr["thickness"] = new_thickness

    def get_thickness(self):
        return self._thickness


class RefractiveSurface(ThickSurface, abc.ABC):
    """
    a base class for any refractive surface, has functions to assign material and thickness is a required input
    """

    def __init__(self, thickness=1, name="my_window", material=None, *args, **kwargs):
        super().__init__(thickness, name, *args, **kwargs)  # call the parent constructor

        # if a material was provided add it to the dict
        self._material = None  # declare this, but it's going to be overwritten
        if material is not None:
            self.set_material(material)

    def set_material(self, new_material):
        # validate that the material exists in the optica surfaces
        if new_material in OPTICA_MATERIALS:
            self._optica_options_dict["ComponentMedium"] = new_material
            self._material = new_material
        else:
            raise ValueError(f"material {new_material} is not listed as an Optica compliant material")

    def get_material(self):
        return self._material


class ThinBaffle(TracerSurface):
    _optica_function_call = "ThinBaffle"

    def __init__(self, name="my_baffle", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _create_optica_function_arguments(self):
        func_name = "ThinBaffle"
        func_args = to_mathematica_array(self._aperture.shape)
        return func_name, func_args


class Baffle(ThickSurface):
    _optica_function_call = "Baffle"

    def __init__(self, thickness=1, name="my_baffle", *args, **kwargs):
        super().__init__(thickness, name, *args, **kwargs)

    def _create_optica_function_arguments(self):
        func_name = "Baffle"
        func_args = to_mathematica_array(self._aperture.shape) + ', ' + f"{self._thickness:.03f}"
        return func_name, func_args


class Window(RefractiveSurface):
    _optica_function_call = "Window"

    def __init__(self, thickness=1, material=None, name="my_window", *args, **kwargs):
        super().__init__(thickness, name, material, *args, **kwargs)  # call the parent constructor

    def _create_optica_function_arguments(self):
        func_name = "Window"
        func_args = to_mathematica_array(self._aperture.shape) + ',' + f"{self._thickness:.03f}"
        return func_name, func_args


class AperturedWindow(RefractiveSurface):
    _optica_function_call = "AperturedWindow"

    def _create_optica_function_arguments(self):
        pass

    def __init__(self, thickness=1, material=None, name="my_window", *args, **kwargs):
        super().__init__(thickness, name, material, *args, **kwargs)  # call the parent constructor
        self._sub_aperture = _Aperture()


class Sphere(RenderObject):
    def __init__(self, radius=1, name="sphere", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
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

        # step one is transform the rays into object space -- rays should always exist in world space!
        object_space_rays = np.matmul(self._get_object_transform(), np.atleast_3d(rays))
        origins = object_space_rays[0, :-1]  # should be a 4xn array of points
        directions = object_space_rays[1, :-1]  # should be a 4xn array of vectors

        # calculate the a,b, and c of the polynomial roots equation
        a = cg.element_wise_dot(directions, directions, axis=0)  # a must be positive because it's the squared magnitude
        b = 2*cg.element_wise_dot(directions, origins, axis=0)  # be can be positive or negative
        c = cg.element_wise_dot(origins, origins, axis=0) - self._radius ** 2  # c can be positive or negative

        # calculate the discriminant, but override the sqrt if it would result in a negative number
        disc = b ** 2 - 4 * a * c
        root = np.sqrt(np.maximum(0, disc))
        hits = np.array(((-b + root), (-b - root))) / (2 * a)  # the positive element of the polynomial root

        # want to keep the smallest hit that is positive, so if hits[1]<0, just keep the positive hit
        nearest_hit = np.where(hits[1] >= 0, np.amin(hits, axis=0), hits[0])
        return np.where(np.logical_and(disc >= 0, nearest_hit >= 0), nearest_hit, np.inf)

    def normal(self, intersections):
        pass
