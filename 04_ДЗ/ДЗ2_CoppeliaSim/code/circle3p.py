# -*- coding: utf-8 -*-
"""Геометрия ДЗ2: окружность через три точки в пространстве и ориентация инструмента вдоль траектории."""
import numpy as np


def circle_through_points(p1, p2, p3, tol=1e-9):
    """Центр C, радиус R, орты плоскости u, w и нормаль n окружности через три точки.
    ValueError, если точки лежат на одной прямой."""
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    a, b = p2 - p1, p3 - p1
    n = np.cross(a, b)
    if np.linalg.norm(n) < tol * max(1.0, np.linalg.norm(a) * np.linalg.norm(b)):
        raise ValueError('точки лежат на одной прямой — окружность не определена')
    # центр: C = p1 + alpha*a + beta*b, условия |C-p1| = |C-p2| = |C-p3|
    aa, bb, ab = a @ a, b @ b, a @ b
    det = aa * bb - ab * ab
    alpha = bb * (aa - ab) / (2 * det)
    beta = aa * (bb - ab) / (2 * det)
    c = p1 + alpha * a + beta * b
    r = np.linalg.norm(p1 - c)
    n = n / np.linalg.norm(n)
    u = (p1 - c) / r
    w = np.cross(n, u)
    return c, r, u, w, n


def circle_points(p1, p2, p3, num=72):
    """Точки окружности, начиная с p1, в направлении p1 -> p2 -> p3; num+1 точка (замкнуто)."""
    c, r, u, w, n = circle_through_points(p1, p2, p3)
    s = np.linspace(0.0, 2 * np.pi, num + 1)
    pts = c + r * (np.outer(np.cos(s), u) + np.outer(np.sin(s), w))
    return pts, s


def tool_orientation(n, approach='normal'):
    """Матрица ориентации инструмента: ось z инструмента направлена против нормали плоскости (к «бумаге»)."""
    z = -np.asarray(n, dtype=float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


if __name__ == '__main__':
    P = [(0.6, -0.2, 0.5), (0.8, 0.0, 0.7), (0.6, 0.2, 0.5)]
    c, r, u, w, n = circle_through_points(*P)
    print('C =', c.round(4), ' R = %.4f' % r, ' n =', n.round(4))
    for p in P:
        print('  расстояние от C до точки: %.6f, отклонение от плоскости: %.1e' % (np.linalg.norm(np.array(p) - c), abs((np.array(p) - c) @ n)))
