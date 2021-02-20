import pyrayt.simple_cg as cg
import collections
from dataclasses import dataclass, field
import re


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
class AnalyticSystem(object):
    sources: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
    components: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
    detectors: cg.ObjectGroup = field(default_factory=cg.ObjectGroup)
