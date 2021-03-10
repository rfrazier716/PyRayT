from tinygfx.g3d.primitives import ObjectGroup
import tinygfx.g3d as cg
from dataclasses import dataclass, field


def flatten(list_to_flatten):
    def _flatten_helper(iterable, flattened_list):
        """
        a helper method to walk through a nested list and return the flattened list in order

        :param iterable:
        :param flattened_list:
        :return:
        """
        for item in iterable:
            # if the item is itself iterable, recursively enter the function
            if not isinstance(item, str) and hasattr(item, '__iter__'):
                _flatten_helper(item, flattened_list)
            else:
                flattened_list.append(item)

    # reset the flattened_system
    flattened_list = []
    _flatten_helper(list_to_flatten, flattened_list)
    return flattened_list


@dataclass(unsafe_hash=True)
class OpticalSystem(object):
    sources: ObjectGroup = field(default_factory=ObjectGroup)
    components: ObjectGroup = field(default_factory=ObjectGroup)
    detectors: ObjectGroup = field(default_factory=ObjectGroup)
    boundary: cg.TracerSurface = field(
        default_factory=lambda: cg.Cuboid.from_corners((-100, -100, -100), (100, 100, 100)))





