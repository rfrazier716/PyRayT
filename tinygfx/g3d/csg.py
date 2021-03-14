from enum import Enum
import numpy as np

from tinygfx.g3d import Intersectable, primitives as primitives


class Operation(Enum):
    UNION = 1
    INTERSECT = 2
    DIFFERENCE = 3


def array_csg(array1: np.ndarray, array2: np.ndarray, operation: Operation):
    """
    Given two arrays and an operation, returns a new array which is the CSG operation acting on the array.
    If the array is thought of as intersection points between a ray and a two objects being combined with a CSG
    operation, the returned is the valid hits for the resulting object. Function assumes both arrays are sorted and have
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

        if np.any(ray_hit_path[-1], 0):
            raise ValueError("Invalid Ray Traversal, Shape was never exited?")
        # for a union operation you want the longest spans from 1-0, do this by casting all numbers >1 to zero
        ray_hit_path = np.where(ray_hit_path > 1, 1, ray_hit_path)
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

        if np.any(ray_hit_path[-1]) != 0:
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

        # now add together the ray hit path with itself, rolled one down
        ray_hit_path += np.roll(ray_hit_path, 1, axis=0)
        return np.sort(np.where(ray_hit_path == 1, merged_array, np.inf), axis=0)


class CSGSurface(Intersectable):

    def __init__(self, l_child: Intersectable, r_child: Intersectable, operation: Operation, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

        # store the boolean type
        self._operation = operation

        # attach the child objects to the parent
        # make it so if a child is modified the bounding boxes are recalculated
        self._l_child = l_child
        self._l_child.attach_to(self)
        self._l_child.var_watchlist.append(self._update_bounding_box)

        self._r_child = r_child
        self._r_child.attach_to(self)
        self._r_child.var_watchlist.append(self._update_bounding_box)

        self._update_bounding_box()  # update the bounding box

    def _update_bounding_box(self):
        """
        the bounding box is only updated for add and intersection operations
        :return:
        """
        if self._operation != Operation.DIFFERENCE:
            new_spans = array_csg(self._l_child.bounding_box.axis_spans.T,
                                  self._r_child.bounding_box.axis_spans.T,
                                  self._operation)

            # check if the resulting spans are valid
            if (self._operation == Operation.UNION and np.any(np.isfinite(new_spans[2:]))) or \
                    (self._operation == Operation.INTERSECT and np.any(np.all(np.isinf(new_spans), axis=0))):
                raise ValueError(f"CSG Child Surfaces {self._l_child} and {self._r_child} no longer intersect")

            self._aobb = primitives.Cube(*new_spans[:2])

        else:
            # for a diff operation the bounding box must always be the bounding box of the left hand child
            self._aobb = self._l_child.bounding_box

    def intersect(self, rays):
        # steps for intersection
        # 1 check that the rays intersect the bounding box
        # for the rays that do, call the l_child and r_child intersect functions
        # depending on the CSG operation reject certain hits, and return

        # get a boolean mask of rays that intersect the surface
        rays = np.atleast_3d(rays)
        bounding_box_intersections = np.any(np.isfinite(self._aobb.intersect(rays)), axis=0)

        # calculate the csg hits for the array subset that intersects the object
        intersecting_rays = rays[:,:,bounding_box_intersections]  # make a subset of rays that intersect
        local_ray_set = np.matmul(self._get_object_transform(), intersecting_rays)
        l_hits = self._l_child.intersect(local_ray_set)
        r_hits = self._r_child.intersect(local_ray_set)
        csg_hits = array_csg(l_hits, r_hits, self._operation)

        # plug the csg_hits back into a main hit matrix
        all_hits = np.full((csg_hits.shape[0], rays.shape[-1]), np.inf)
        all_hits[:,bounding_box_intersections] = csg_hits  # plug the csg_hits into the hits matrix
        return all_hits  # return the hits matrix
