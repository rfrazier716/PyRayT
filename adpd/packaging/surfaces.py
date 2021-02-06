import abc
import adpd.packaging.simple_cg as pycg
import numpy as np
import json
import copy

OPTICA_MATERIALS = {"Silicon"}


class _Aperture(object):
    def __init__(self, *args, **kwargs):
        self.shape = tuple()
        self.offset = pycg.Point(0, 0, 0)


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


class TracerSurface(pycg.WorldObject, abc.ABC):
    _shape_label = "GenericSurface"  # the label for the shape, class specific

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO
        self._surface_id = type(self)._shape_label + f"{self.get_id():04d}"
        self._opts_string = self._surface_id + "OPTS"

        self._name = name  # a local name of for the surface
        # a dictionary that holds important parameters that will be used to construct a json string of the object
        self._json_repr = dict()
        self._json_repr["name"] = self._name  # add name to be exported
        self._json_repr["type"] = self._shape_label

        self._aperture = _Aperture()
        self.set_aperture_shape(1)  # make the aperture a circle with radius of 1 by default
        self._optica_options_dict = {}  # a dictionary of additional keyword options

    def get_surface_id(self):
        return self._surface_id

    def get_name(self):
        return self._name

    def set_aperture_shape(self, new_shape):
        self._aperture.shape = new_shape
        self._json_repr["aperture"] = new_shape
        return self

    def set_aperture_offset(self, x_offset, y_offset):
        self._aperture.offset.x = x_offset
        self._aperture.offset.y = y_offset
        self._optica_options_dict["OffAxis"] = (x_offset, y_offset)

    def add_custom_parameter(self, parameter_key: str, parameter_value):
        self._optica_options_dict[str(parameter_key)] = parameter_value

    def collect_parameters(self):
        """
        Returns a dict of key parameters that define the surface
        :return:
        """
        self._json_repr["position"] = tuple(self.get_position()[:3].astype(float))  # get the non-homogenous position
        self._json_repr["rotation"] = tuple(self.get_quaternion().astype(float))  # get the rotation represented as a quaternion
        self._json_repr.update(self._optica_options_dict)
        return copy.copy(self._json_repr)

    def to_json_string(self):
        """
        return a JSON string representing the object
        """
        return json.dumps(self.collect_parameters(), indent=2)

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
        return func_name + "[" + func_arguments + "," + self._opts_string + "]"

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
    _shape_label = "ThinBaffle"

    def __init__(self, name="my_baffle", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _create_optica_function_arguments(self):
        func_name = "ThinBaffle"
        func_args = self._create_aperture_string()
        return func_name, func_args


class Baffle(ThickSurface):
    _shape_label = "Baffle"

    def __init__(self, thickness=1, name="my_baffle", *args, **kwargs):
        super().__init__(thickness, name, *args, **kwargs)

    def _create_optica_function_arguments(self):
        func_name = "Baffle"
        func_args = self._create_aperture_string() + ',' + f"{self._thickness:.03f}"
        return func_name, func_args


class Window(RefractiveSurface):
    _shape_label = "Window"

    def __init__(self, thickness=1, material=None, name="my_window", *args, **kwargs):
        super().__init__(thickness, name, material, *args, **kwargs)  # call the parent constructor

    def _create_optica_function_arguments(self):
        func_name = "Window"
        func_args = self._create_aperture_string() + ',' + f"{self._thickness:.03f}"
        return func_name, func_args


class AperturedWindow(RefractiveSurface):
    _shape_label = "AperturedWindow"

    def __init__(self, thickness=1, material=None, name="my_window", *args, **kwargs):
        super().__init__(thickness, name, material, *args, **kwargs)  # call the parent constructor
        self._sub_aperture = _Aperture()
