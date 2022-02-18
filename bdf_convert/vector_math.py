import math


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vector_minus(a, b):
    return (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2])


def triangle_area(a, b, c):
    v = vector_minus(b, a)
    w = vector_minus(c, a)
    n = cross(v, w)
    return math.sqrt(dot(n, n)) / 2


def tetrahedron_volume(a, b, c, d):
    u = vector_minus(b, a)
    v = vector_minus(c, a)
    w = vector_minus(d, a)

    return 1. / 6. * math.fabs(dot(cross(u, v), w))
