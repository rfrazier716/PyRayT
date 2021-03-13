from enum import Enum
import numpy as np


class Operation(Enum):
    UNION = 1
    INTERSECT = 2
    DIFFERENCE = 3


def array_csg(array1: np.ndarray, array2: np.ndarray, operation: Operation):
    """
    Given two arrays and an operation, returns a new array which is the CSG operation acting on the array.
    If the array is thought of as intersection points between a ray and a to objects being combined with a CSG
    operation, the result is the valid hits for the resulting object. Function assumes both arrays are sorted and have
    an even number of axis=0 elements

    :param array1:
    :param array2:
    :param operation:
    :return:
    """

    # we're going to stack both array into one that will be sorted

    if array1.ndim == 1:  # if a 1d array was passed, reshape the joined array appropriately
        merged_array = np.atleast_2d(np.hstack((array1, array2))).T
    else:
        merged_array = np.vstack((array1, array2))

    if operation == Operation.UNION:
        # make a mask that tracks when the ray enters and exits each surface
        merged_mask = np.ones(merged_array.shape)
        merged_mask[1::2] = -1

        # take the argsort along axis0 to sort the hits, this will be used to reshape the mask
        merged_argsort = np.argsort(merged_array, axis=0)
        merged_array = np.sort(merged_array, axis=0)  # sort the merged array since we don't need it anymore
        merged_mask = merged_mask[merged_argsort, np.arange(merged_mask.shape[-1])]
        ray_hit_path = np.cumsum(merged_mask, axis=0)
        print(ray_hit_path)

        if ray_hit_path[-1] != 0:
            raise ValueError("Invalid Ray Traversal, Shape was never exited?")
        # for a union operation you want the longest spans from 1-0, do this by casting all numbers >1 to zero
        ray_hit_path = np.where(ray_hit_path > 1, 1, ray_hit_path)
        print(ray_hit_path)
        ray_hit_path = np.diff(ray_hit_path, axis=0, prepend=0)
        return np.sort(np.where(ray_hit_path != 0, merged_array, np.inf), axis=0)

    if operation == Operation.INTERSECT:
        # make a mask that tracks when the ray enters and exits each surface
        merged_mask = np.ones(merged_array.shape)
        merged_mask[1::2] = -1

        # take the argsort along axis0 to sort the hits, this will be used to reshape the mask
        merged_argsort = np.argsort(merged_array, axis=0)
        merged_array = np.sort(merged_array, axis=0)  # sort the merged array since we don't need it anymore
        merged_mask = merged_mask[merged_argsort, np.arange(merged_mask.shape[-1])]
        ray_hit_path = np.cumsum(merged_mask, axis=0)

        if ray_hit_path[-1] != 0:
            raise ValueError("Invalid Ray Traversal, Shape was never exited?")

        # for an intersection operation we want to find where the ray hit path is 2 and take the next point

        ray_hit_path = np.where(ray_hit_path == 2, 1, 0)
        ray_hit_path += np.roll(ray_hit_path, 1, axis=0)
        return np.sort(np.where(ray_hit_path == 1, merged_array, np.inf), axis=0)

    if operation == Operation.DIFFERENCE:
        # A difference operation is the same as the union operate of A with the inverse of B.

        merged_mask = np.ones(merged_array.shape)
        merged_mask[1:array1.shape[0]:2] = -1
        merged_mask[array1.shape[0]::2] = -1

        # take the argsort along axis0 to sort the hits, this will be used to reshape the mask
        merged_argsort = np.argsort(merged_array, axis=0)
        merged_array = np.sort(merged_array, axis=0)  # sort the merged array since we don't need it anymore
        merged_mask = merged_mask[merged_argsort, np.arange(merged_mask.shape[-1])]
        ray_hit_path = np.cumsum(merged_mask, axis=0)

        # first, cast anything that isn't a 1 to a zero
        ray_hit_path = np.where(ray_hit_path != 1, 0, ray_hit_path)
        print(ray_hit_path)

        # now add together the ray hit path with itself, rolled one down
        ray_hit_path += np.roll(ray_hit_path, 1, axis=0)
        return np.sort(np.where(ray_hit_path == 1, merged_array, np.inf), axis=0)
