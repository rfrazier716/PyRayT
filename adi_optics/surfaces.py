import abc
import adi_optics.pycg.pycg as pycg
import numpy as np

OPTICA_MATERIALS = {"Silicon"}

class _Aperture(object):
    def __init__(self,*args, **kwargs):
        self.shape = tuple()
        self.offset = pycg.Point(0,0,0)

def to_mathematica_array(np_array):
    return np.array2string(np.asarray(np_array), precision=6, separator=',').replace("[","{").replace("]","}").replace('\n','')

class TracerSurface(pycg.WorldObject, abc.ABC):
    _object_count = 0 # keeps track of how many objects of each type exist
    _shape_label = "GenericSurface" # the label for the shape, class specific

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs) # call the next constructor in the MRO
        type(self)._object_count+=1 # increment the counter for how many classes of this type were made
        self._shape_id = type(self)._shape_label + f"{type(self)._object_count:04d}"

        self._name = name # a local name of for the surface
        self._aperture = _Aperture()

    def set_aperture_shape(self, new_shape):
        self._aperture.shape = new_shape
        return self

    def set_aperture_offset(self, x_offset, y_offset):
        self._aperture.offset.x = x_offset
        self._aperture.offset.y = y_offset

    @abc.abstractmethod
    def to_json(self):
        """
        return a JSON string representing the object
        """
        pass

    @abc.abstractmethod
    def _get_optica_obj_string(self):
        return "XXX"

    def create_optica_function(self):
        obj_declaration = self._shape_id+" = "+ self._get_optica_obj_string()+';' # create the object string
        move_array = to_mathematica_array(self.get_position()[:3])
        rotation_array = to_mathematica_array(self._world_coordinate_transform[:3, :3])
        move_command = self._name+"=Move[{label}, {move}, {rot}];".format(
            label=self._shape_id,
            move=move_array,
            rot=rotation_array
        )
        return "\n".join([obj_declaration, move_command])


class Window(TracerSurface):
    _shape_label = "window"

    def __init__(self, thickness=1, material=None, name="my_window", *args, **kwargs):
        print(kwargs)
        print(args)
        super().__init__(name, *args, **kwargs) # call the parent constructor
        self._thickness = thickness
        self._material = material

    def to_json(self):
        pass

    def _get_optica_obj_string(self):
        aperture_str = to_mathematica_array(self._aperture.shape)
        math_options = []
        if self._material is not None:
            if self._material in OPTICA_MATERIALS:
                math_options.append("ComponentMedium->"+self._material)

        math_opts=""
        if math_options:
            math_opts = ', '+', '.join(math_options)

        return "Window[{aperture}, {thickness:.02f}{options}]".format(aperture=aperture_str,
                                                                        thickness=self._thickness,
                                                                        options=math_opts)








