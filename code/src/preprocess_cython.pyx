import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, atan2

cnp.import_array()

DTYPE = np.float32
cdef float MAX_ANGLE = 2 * np.pi

ctypedef cnp.float32_t DTYPE_t

cdef void get_contour_points_internal(cnp.ndarray[DTYPE_t, ndim=2] mask, int center_x, int center_y, int resolution, cnp.ndarray[DTYPE_t, ndim=1] max_points, float bbox_size):
    assert mask.dtype == DTYPE

    cdef int vmax = mask.shape[0]
    cdef int wmax = mask.shape[1]
    cdef int c_x = 0
    cdef int c_y = 0
    cdef double p = 0
    cdef double a = 0
    cdef int angle_index = 0
    cdef int x = 0
    cdef int y = 0

    for x in range(wmax):
        for y in range(vmax):
            if mask[y,x] == 0:
                continue
            
            c_x, c_y = x - center_x, y - center_y
            p, a = sqrt(c_x**2 + c_y**2) ,atan2(c_y, c_x)
            a = a if a >= 0 else (MAX_ANGLE + a)

            if bbox_size > 0:
                p = p / bbox_size

            angle_index =  int(a * resolution / MAX_ANGLE)

            if p > max_points[angle_index]:
                max_points[angle_index] = p


def get_contour_points_from_mask(cnp.ndarray[DTYPE_t, ndim=2] mask, int center_x, int center_y, int resolution, float bbox_size = 0):
    cdef cnp.ndarray[DTYPE_t, ndim=1] max_points = np.zeros(resolution, dtype=DTYPE)

    get_contour_points_internal(mask, center_x, center_y, resolution, max_points, bbox_size)

    return max_points
