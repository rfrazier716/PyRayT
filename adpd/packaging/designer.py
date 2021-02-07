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


class OpticalSystem(collections.UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

        # create default object groups that can be written to
        self.data["sources"] = cg.ObjectGroup()
        self.data["components"] = cg.ObjectGroup()
        self.data["detectors"] = cg.ObjectGroup()

    def to_optica(self, output_stream):
        # iterate over the data fields
        # for each
        group_writeout = ''
        for key, value in self.data.items():
            items_in_group = self._to_optica_helper(output_stream, value, [])
            replacement_map = {'\'': '', '[': '{', ']': '}'}
            group_set = _multi_string_replace(str(items_in_group), replacement_map)
            group_writeout+=key+' = '+group_set+';\n'

        output_stream.write(group_writeout)

    @staticmethod
    def _to_optica_helper(file_stream, iterable, nested_list):
        for item in iterable:
            # if the item is itself iterable, recursively enter the function
            if not isinstance(item, str) and hasattr(item, '__iter__'):
                nested_list.append(OpticalSystem._to_optica_helper(file_stream, item, []))
            else:
                file_stream.write(f"(*--{item.get_name()}--*)\n")
                file_stream.write(item.create_optica_function() + '\n\n')
                nested_list.append(item.get_name())
        return nested_list
