import adpd.packaging.surfaces as surfaces
import adpd.packaging.simple_cg as cg
import collections
import re


def _multi_string_replace(string, replacements: dict):
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)

    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped))

    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[match.group(0)], string)


class AnalyticSystem(collections.UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

        # create default object groups that can be written to
        self.data["sources"] = cg.ObjectGroup()
        self.data["components"] = cg.ObjectGroup()
        self.data["detectors"] = cg.ObjectGroup()

        self._flattened_system = list() # a placeholder for the flattened optical system

    def flatten(self):
        # reset the flattened_system
        self._flattened_system = []
        self._flatten_helper(self.data.values(), self._flattened_system)
        return self._flattened_system

    @staticmethod
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
                AnalyticSystem._flatten_helper(item, flattened_list)
            else:
                flattened_list.append(item)



