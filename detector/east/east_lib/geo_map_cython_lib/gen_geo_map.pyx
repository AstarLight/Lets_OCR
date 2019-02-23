import numpy as np
cimport numpy as np
cimport cython


ctypedef fused DTYPE_t:
	np.float32_t
	np.float64_t


@cython.boundscheck(False)
cdef inline DTYPE_t tri_area(DTYPE_t a, DTYPE_t b, DTYPE_t c):
	cdef DTYPE_t s = (a + b + c) / 2.0
	return abs(s*(s-a)*(s-b)*(s-c)) ** 0.5


@cython.boundscheck(False)
cdef inline DTYPE_t norm(DTYPE_t dx, DTYPE_t dy):
	return (dx**2 + dy**2) ** 0.5	


def gen_geo_map(np.ndarray[DTYPE_t, ndim=3] geo_map, np.ndarray[np.int_t, ndim=2] xy_in_poly, np.ndarray[DTYPE_t, ndim=2] rectangle, DTYPE_t rot_angle):
	cdef int i, num_pt
	num_pt = xy_in_poly.shape[0]
	cdef np.int_t y, x
	cdef np.ndarray[DTYPE_t, ndim=1] p0, p1, p2, p3

	cdef DTYPE_t a, b, c

	p0 = rectangle[0, :]
	p1 = rectangle[1, :]
	p2 = rectangle[2, :]
	p3 = rectangle[3, :]

	for i in range(num_pt):
		y = xy_in_poly[i, 0]
		x = xy_in_poly[i, 1]

		# top, right, down, left, angle.
		a = norm(p1[0]-p0[0], p1[1]-p0[1])
		b = norm(p0[0]-x, p0[1]-y)
		c = norm(p1[0]-x, p1[1]-y)
		if a >= 1.0:
			geo_map[y, x, 0] = 2 * tri_area(a, b, c) / a
		else:
			geo_map[y, x, 0] = 2 * tri_area(a, b, c) / (a+1)

		a = norm(p2[0]-p1[0], p2[1]-p1[1])
		b = norm(p1[0]-x, p1[1]-y)
		c = norm(p2[0]-x, p2[1]-y)
		if a >= 1.0:
			geo_map[y, x, 1] = 2 * tri_area(a, b, c) / a
		else:
			geo_map[y, x, 1] = 2 * tri_area(a, b, c) / (a+1)

		a = norm(p3[0]-p2[0], p3[1]-p2[1])
		b = norm(p2[0]-x, p2[1]-y)
		c = norm(p3[0]-x, p3[1]-y)
		if a >= 1.0:
			geo_map[y, x, 2] = 2 * tri_area(a, b, c) / a
		else:
			geo_map[y, x, 2] = 2 * tri_area(a, b, c) / (a+1)

		a = norm(p0[0]-p3[0], p0[1]-p3[1])
		b = norm(p3[0]-x, p3[1]-y)
		c = norm(p0[0]-x, p0[1]-y)
		if a >= 1.0:
			geo_map[y, x, 3] = 2 * tri_area(a, b, c) / a
		else:
			geo_map[y, x, 3] = 2 * tri_area(a, b, c) / (a+1)

		geo_map[y, x, 4] = rot_angle
