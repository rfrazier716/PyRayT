from enum import Enum
import numpy as np

from tinygfx.g3d import Intersectable, primitives, bounding_box


class Operation(Enum):
    UNION = 1
    INTERSECT = 2
    DIFFERENCE = 3


def array_csg(
    array1: np.ndarray, array2: np.ndarray, operation: Operation, sort_output=True
):
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

    if array1.ndim == 1:
        # if 1D arrays were passed, concatenate
        merged_array = np.concatenate((array1, array2))
        merged_argsort = np.argsort(merged_array, axis=0)
        merged_array = merged_array[merged_argsort]

    else:
        # otherwise stack them where each column represents a unique ray's hits
        merged_array = np.vstack((array1, array2))
        merged_argsort = np.argsort(merged_array, axis=0)
        merged_array = merged_array[merged_argsort, np.arange(merged_array.shape[-1])]

    if operation == Operation.UNION or operation == Operation.INTERSECT:
        merged_mask = np.where(merged_argsort & 1, -1, 1)
        surface_count = np.cumsum(merged_mask, axis=0)

    elif operation == Operation.DIFFERENCE:
        merged_mask = np.where(
            np.logical_xor(merged_argsort & 1, merged_argsort >= array1.shape[0]), -1, 1
        )
        surface_count = np.cumsum(merged_mask, axis=0) + 1
    else:
        raise ValueError(f"operation {operation} is invalid")

    if operation == Operation.UNION:
        surface_count = np.logical_xor(surface_count, np.roll(surface_count, 1, axis=0))
        csg_hits = np.where(surface_count != 0, merged_array, np.inf)

    elif operation == Operation.INTERSECT or operation == Operation.DIFFERENCE:
        is_two = surface_count == 2
        mask = np.logical_or(is_two, np.roll(is_two, 1, axis=0))
        csg_hits = np.where(mask, merged_array, np.inf)

    return np.sort(csg_hits, axis=0) if sort_output else csg_hits


class CSGSurface(Intersectable):
    def __init__(
        self,
        l_child: Intersectable,
        r_child: Intersectable,
        operation: Operation,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # call the next constructor in the MRO

        # store the boolean type
        self._operation = operation
        self.var_watchlist.append(self._update_bounding_box)

        # attach the child objects to the parent
        # make it so if a child is modified the bounding boxes are recalculated
        self._l_child = l_child
        self._l_child.attach_to(self)

        self._r_child = r_child
        self._r_child.attach_to(self)

        # invert the normals on a difference operation so colors process right
        if self._operation == Operation.DIFFERENCE:
            self._r_child.invert_normals()

        self._update_bounding_box()  # update the bounding box

    def _update_bounding_box(self):
        """
        the bounding box is only updated for add and intersection operations
        :return:
        """
        if self._operation != Operation.DIFFERENCE:
            new_spans = array_csg(
                self._l_child.bounding_box.axis_spans.T,
                self._r_child.bounding_box.axis_spans.T,
                self._operation,
            )

            # check if the resulting spans are valid
            # if (self._operation == Operation.UNION and np.any(np.isfinite(new_spans[2:]))) or \
            #         (self._operation == Operation.INTERSECT and np.any(np.all(np.isinf(new_spans), axis=0))):
            #     raise ValueError(f"CSG Child Surfaces {self._l_child} and {self._r_child} no longer intersect")

            local_aobb = primitives.Cube(*new_spans[:2])

        else:
            # for a diff operation the bounding box must always be the bounding box of the left hand child
            local_aobb = self._l_child.bounding_box

        self._aobb = local_aobb

    def intersect(self, rays):
        # steps for intersection
        # 1 check that the rays intersect the bounding box
        # for the rays that do, call the l_child and r_child intersect functions
        # depending on the CSG operation reject certain hits, and return

        # get a boolean mask of rays that intersect the surface
        rays = np.atleast_3d(rays)
        bounding_box_intersections = np.any(
            np.isfinite(self._aobb.intersect(rays)), axis=0
        )

        # calculate the csg hits for the array subset that intersects the object
        local_ray_set = rays[
            :, :, bounding_box_intersections
        ]  # make a subset of rays that intersect
        l_hits, l_surfaces = self._l_child.intersect(local_ray_set)
        r_hits, r_surfaces = self._r_child.intersect(local_ray_set)

        # build a mask to sort the surfaces
        csg_surfaces = np.vstack((l_surfaces, r_surfaces))
        surface_sort_mask = np.argsort(np.vstack((l_hits, r_hits)), axis=0)
        csg_surfaces = csg_surfaces[
            surface_sort_mask, np.arange(csg_surfaces.shape[-1])
        ]

        csg_hits = array_csg(l_hits, r_hits, self._operation, sort_output=False)

        # sort the hits and the surface mask
        hit_argsort = np.argsort(csg_hits, axis=0)
        csg_surfaces = csg_surfaces[hit_argsort, np.arange(csg_surfaces.shape[-1])]
        csg_hits = csg_hits[hit_argsort, np.arange(csg_hits.shape[-1])]

        # plug the csg_hits back into a main hit matrix
        all_hits = np.full((csg_hits.shape[0], rays.shape[-1]), np.inf)
        all_hits[
            :, bounding_box_intersections
        ] = csg_hits  # plug the csg_hits into the hits matrix

        # plug the csg surfaces back into the main surface matrix
        all_surfaces = np.full((csg_hits.shape[0], rays.shape[-1]), -1)
        all_surfaces[:, bounding_box_intersections] = csg_surfaces
        return all_hits, all_surfaces  # return the hits matrix

    def invert_normals(self):
        self._l_child.invert_normals()
        self._r_child.invert_normals()

    def reset_normals(self):
        self._l_child.reset_normals()
        self._r_child.reset_normals()

    @property
    def surface_ids(self) -> tuple:
        # returns the surface id's of both children
        return self._l_child.surface_ids + self._r_child.surface_ids

    def _append_world_transform(self, new_transform):
        # override the append world transform so children are moved too
        super()._append_world_transform(new_transform)
        self._l_child.transform(new_transform)
        self._r_child.transform(new_transform)


def union(s0: Intersectable, s1: Intersectable) -> CSGSurface:
    return CSGSurface(s0, s1, Operation.UNION)


def intersect(s0: Intersectable, s1: Intersectable) -> CSGSurface:
    return CSGSurface(s0, s1, Operation.INTERSECT)


def difference(s0: Intersectable, s1: Intersectable) -> CSGSurface:
    return CSGSurface(s0, s1, Operation.DIFFERENCE)
