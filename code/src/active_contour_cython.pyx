import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef cnp.ndarray[cnp.int8_t, ndim=2] N4_ENCODING = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

cdef struct s_ActiveContourStatus:
    cnp.ndarray[cnp.int8_t, ndim=2] level_set
    cnp.ndarray[cnp.float32_t, ndim=2] speed
    cnp.ndarray[cnp.int32_t, ndim=2] histogram
    cnp.ndarray[cnp.float32_t, ndim=1] bin_sizes
    bool initialized
    int max_iter

ctypedef s_ActiveContourStatus ActiveContourStatus

cdef bool is_within_coords(int i, int j, int max_i, int max_j):
    return i >= 0 and i < max_i and j >= 0 and j < max_j

cdef cnp.ndarray get_n4 (int i, int j, cnp.ndarray[cnp.float32_t, ndim=2] array):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] n4 = np.zeros(4)
    cdef int c = 0
    cdef int i2 = 0
    cdef int j2 = 0

    for c in range(4):
        i2 = i + N4_ENCODING[c, 0]
        j2 = j + N4_ENCODING[c, 1]
        n4[c] = array[i2, j2] if is_within_coords(i2, j2, array.shape[0], array.shape[1]) else 0
    
    return n4

cdef bool is_internal(float value, cnp.ndarray[cnp.float32_t, ndim=1] n4):
    return value == 1 and n4[n4 == 1].any()
    
cdef bool is_l_in(float value, cnp.ndarray[cnp.float32_t, ndim=1] n4):
    return value == 1 and n4[n4 == 0].any()
    
cdef bool is_l_out(float value, cnp.ndarray[cnp.float32_t, ndim=1] n4):
    return value == 0 and n4[n4 == 1].any()

cdef cnp.ndarray bin_array(cnp.ndarray[cnp.float32_t, ndim=1] array)

cdef float proba(ActiveContourStatus status, cnp.ndarray[cnp.float32_t, ndim=3] image, float y, float v, bool is_in):
    cdef char level = -3 if is_in else 3
    cdef cnp.ndarray[cnp.float32_t, ndim=2] level_set_points = image[status.level_set == level]
    cdef int total_cnt = len(level_set_points)
    cdef int values_cnt = len(level_set_points[(level_set_points[:, 0] // status.bin_size[0]) == (y // self.features_bin_size[0]) \
                                    & (level_set_points[:, 2] // self.features_bin_size[1]) == (v //self.features_bin_size[1])])
    return values_cnt / total_cnt
    
cdef _calculate_speed(self, image: np.ndarray, pixel):
        return np.log(self._prob(image, pixel[0], pixel[2], True) / self._prob(image, pixel[0], pixel[2], False))

cdef void initialize(ActiveContourStatus status, cnp.ndarray[cnp.float32_t, ndim=2] image, cnp.ndarray[cnp.int8_t, ndim=2] mask):
    status.level_set = np.zeros((image.shape[0], image.shape[1]), dtype=np.int8)
    status.speed = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    status.histogram = np.zeros((status.bin_sizes[0], status.bin_sizes[1]), dtype=np.int32)

    cdef int i = 0
    cdef int j = 0
    cdef float y = 0
    cdef float v = 0
    cdef int value = 0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] n4

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            y,v = image[i, j, 0], image[i, j, 2]
            status.histogram[int(y // status.bin_sizes[0]), int(v // status.bin_size[1])] += 1
            value = mask[i,j]
            n4 = get_n4(i, j, mask)
            if is_internal(value, n4):
                status.level_set[i, j] = -3
            elif is_l_in(value, n4):
                status.level_set[i, j] = -1
                status.speed[i, j] = calculate_speed(image, image[i,j])
            elif is_l_out(value, n4):
                status.level_set[i, j] = 1
                status.speed[i, j] = calculate_speed(image, image[i,j])
            else:
                status.level_set[i, j] = 3
    
    status.initialized = True
