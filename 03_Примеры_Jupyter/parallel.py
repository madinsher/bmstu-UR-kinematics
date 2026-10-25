# -*- coding: utf-8 -*-
"""parallel — учебный модуль кинематики механизмов с замкнутыми цепями (занятие 5).

Обозначения — по 00_Паспорт/Обозначения.md и лекции 5:
  q, q_i      — приводные координаты (углы рычагов дельта-робота, ходы кареток d_i, длины ног l_i);
  gamma_i     — азимут плоскости ноги дельта-робота (массив GAMMA; PHI — старое имя);
  W = h (n - j) + sum f_i — структурная формула (n — подвижные звенья, j — сочленения);
  b           — половина общей хорды в прямой задаче пятизвенника;
  h0          — высота оси плеча параллелограммного манипулятора; theta3 = q3 - q2 — угол в локте;
  x = (Theta; p), Theta = (phi, theta, psi) — поза платформы Гью — Стюарта, R = Rz(psi) Ry(theta) Rx(phi);
  V = (omega; v) — скорость платформы (угловая часть первой), F = (m; f) — силовой винт (момент; сила);
  tau         — усилия в ногах платформы (или моменты приводов); K_q = diag(k_i) — жёсткости ног.
Единицы: метры, радианы, ньютоны.

Расчётные функции:
  delta_ik, delta_ik_forms, delta_fk, delta_jac, delta_knees, delta_virtual_centres — дельта-робот 3-RUU;
  delta_quality, delta_section, delta_cloud, delta_sv_grid, delta_smax_peak, delta_stiffness — рабочая зона,
                                       точность (максимум sigma_max уточняется у границы зоны), жёсткость;
  delta_leg_constraints, delta_constraint_matrix, delta_loop_H, delta_nu_det — матрицы связей ноги R-4S
                                       и всего механизма (в скоростях звеньев и в шарнирных координатах, § 5.11),
                                       определитель нормалей параллелограммов;
  lindelta_ik, lindelta_fk, lindelta_jac, lindelta_error — линейный дельта-робот 3-PUU и «чаша» калибровки;
  trilaterate                        — пересечение трёх сфер;
  fivebar_fk, fivebar_ik, fivebar_jac, fivebar_psi, fivebar_limits_ok, fivebar_workspace — пятизвенник 5R;
  pgram_fk, pgram_ik, pgram_jac_rz, pgram_workspace, pgram_dh_robot — параллелограммный манипулятор;
  stewart_geom, stewart_ik, stewart_fk, stewart_fk_hist, stewart_Jl, stewart_invjac, angle_rates_inv,
  stewart_leg_forces, stewart_wrench, rpy — платформа Гью — Стюарта 6-UPS;
  mobility, rank_nullity, ConstraintMatrix, pgram3_loop_H — подвижность по формуле и по рангу;
  draw_fivebar, draw_pgram, draw_delta, draw_lindelta, draw_stewart — скелеты механизмов;
  check                              — самопроверка (печать чисел примеров 5.1–5.7).
Только numpy и matplotlib (roboticstoolbox — по желанию, только в pgram_dh_robot).

Изменения 10.2026 (несовместимые): поза платформы x = (phi, theta, psi, px, py, pz) — углы первыми,
как в лекции (было (px, py, pz, al, be, ga)); mobility(n, j, fsum, h) считает по числу ПОДВИЖНЫХ
звеньев n (было N со стойкой и lam). Ключ h= у pgram_* оставлен как старое имя для h0=.
"""
import numpy as np

# ====================================================================== дельта-робот 3-RUU
# параметры ABB FlexPicker IRB 360 (по Р. Уильямсу), метры
SB, SP, LU, LL = 0.567, 0.076, 0.524, 1.244      # сторона основания s_B, сторона платформы s_P, рычаг L, штанга l
WB = np.sqrt(3) / 6 * SB                           # w_B: от центра основания до оси привода (середина стороны)
UB = np.sqrt(3) / 3 * SB                           # от центра основания до вершины треугольника
WP = np.sqrt(3) / 6 * SP
UP = np.sqrt(3) / 3 * SP                           # u_P: от центра платформы до шарнира (вершина)
GAMMA = np.radians([-90.0, 30.0, 150.0])           # азимуты ног gamma_i (как у Уильямса)
PHI = GAMMA                                        # старое имя
KZ = np.array([0.0, 0.0, 1.0])
Q_LIM = np.radians([-45.0, 95.0])                  # ограничения углов рычагов
TH_LIM = Q_LIM                                     # старое имя
BETA_MAX = np.radians(40.0)                        # наибольшее отклонение штанги от плоскости ноги


def uvec(i):
    """Радиальный орт u_i ноги i (i = 0, 1, 2)."""
    return np.array([np.cos(GAMMA[i]), np.sin(GAMMA[i]), 0.0])


def tvec(i):
    """Касательный орт t_i ноги i — направление оси двигателя."""
    return np.array([-np.sin(GAMMA[i]), np.cos(GAMMA[i]), 0.0])


def delta_ik(p, L=LU, l=LL, a=None, branch=-1):
    """ОЗК дельта-робота 3-RUU. p = (x, y, z), z < 0. Возвращает углы рычагов q (рад), отсчёт от
    горизонтали, положительно вниз. По ногам: E cos q + F sin q + G = 0, t = tg(q/2),
    t = (-F + branch*sqrt(E^2 + F^2 - G^2)) / (G - E).
    branch = -1 — колено наружу (рабочая ветвь), +1 — колено внутрь; можно задать по ногам: (-1, -1, +1).
    None — точка недостижима (подкоренное выражение отрицательно хотя бы у одной ноги)."""
    if a is None:
        a = WB - UP
    br = np.broadcast_to(np.asarray(branch, float), (3,))
    p = np.asarray(p, float)
    q = np.zeros(3)
    for i in range(3):
        c = p - a * uvec(i)                        # c_i: от оси привода B_i к шарниру платформы P_i
        E = -2 * L * c.dot(uvec(i))
        F = 2 * L * c[2]
        G = c.dot(c) + L ** 2 - l ** 2
        D = E * E + F * F - G * G
        if D < 0:
            return None
        t = (-F + br[i] * np.sqrt(D)) / (G - E)
        q[i] = 2 * np.arctan(t)
    return q


def delta_ik_table(p, L=LU, l=LL, a=None):
    """Промежуточные величины ОЗК по ногам (рабочая ветвь): строки (xi, eta, E, F, G, t, q[рад])."""
    if a is None:
        a = WB - UP
    p = np.asarray(p, float); rows = []
    for i in range(3):
        u = uvec(i); c = p - a * u
        xi, eta = c.dot(u), c.dot(tvec(i))
        E = -2 * L * xi; F = 2 * L * p[2]; G = xi ** 2 + eta ** 2 + p[2] ** 2 + L ** 2 - l ** 2
        D = E * E + F * F - G * G
        t = (-F - np.sqrt(D)) / (G - E) if D >= 0 else np.nan
        rows.append((xi, eta, E, F, G, t, 2 * np.arctan(t)))
    return np.array(rows)


def delta_ik_forms(p, i=0, L=LU, l=LL, a=None):
    """Обе формы решения уравнения ноги i и оба знака (рад):
    t-форма q = 2 arctg((-F -+ sqrt(D))/(G - E)) и atan2-форма q = atan2(F, E) +- arccos(-G/sqrt(E^2 + F^2)).
    Возвращает словарь {'t_minus', 't_plus', 'atan2_plus', 'atan2_minus'} (углы приведены к (-pi; pi])."""
    xi, eta, E, F, G, t, q = delta_ik_table(p, L, l, a)[i]
    D = E * E + F * F - G * G
    wrap = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
    r = np.hypot(E, F); base = np.arctan2(F, E); ac = np.arccos(-G / r)
    return dict(t_minus=2 * np.arctan((-F - np.sqrt(D)) / (G - E)), t_plus=2 * np.arctan((-F + np.sqrt(D)) / (G - E)),
                atan2_plus=wrap(base + ac), atan2_minus=wrap(base - ac))


def delta_knees(q, L=LU):
    """Колени A_i = B_i + L_i(q_i)."""
    return [WB * uvec(i) + L * (np.cos(q[i]) * uvec(i) - np.sin(q[i]) * KZ) for i in range(3)]


def delta_virtual_centres(q, L=LU, a=None):
    """Виртуальные центры сфер A_i^v = A_i - P_i = (a + L cos q_i) u_i - L sin q_i k."""
    if a is None:
        a = WB - UP
    return [(a + L * np.cos(q[i])) * uvec(i) - L * np.sin(q[i]) * KZ for i in range(3)]


def trilaterate(c1, c2, c3, r1, r2, r3, full=False):
    """Пересечение трёх сфер. Возвращает две точки (или None). Не имеет особенности при
    равных высотах центров; вырождается, только если центры лежат на одной прямой (j = 0).
    full=True — дополнительно словарь промежуточных величин d, i, j, X, Y, Z."""
    c1, c2, c3 = (np.asarray(c, float) for c in (c1, c2, c3))
    ex = (c2 - c1); d = np.linalg.norm(ex); ex = ex / d
    i = ex.dot(c3 - c1)
    ey = c3 - c1 - i * ex; j = np.linalg.norm(ey); ey = ey / j
    ez = np.cross(ex, ey)
    X = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    Y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2 - 2 * i * X) / (2 * j)
    Z2 = r1 ** 2 - X ** 2 - Y ** 2
    if Z2 < 0:
        return None
    Z = np.sqrt(Z2)
    base = c1 + X * ex + Y * ey
    sol = (base + Z * ez, base - Z * ez)
    if full:
        return sol, dict(d=d, i=i, j=j, X=X, Y=Y, Z=Z)
    return sol


def delta_fk(q, L=LU, l=LL, a=None, both=False):
    """ПЗК дельта-робота 3-RUU: пересечение трёх сфер радиуса l с виртуальными центрами.
    Возвращает решение под основанием (меньшее z); both=True — (нижнее, верхнее)."""
    cs = delta_virtual_centres(q, L, a)
    sol = trilaterate(cs[0], cs[1], cs[2], l, l, l)
    if sol is None:
        return None
    lo, hi = (sol[0], sol[1]) if sol[0][2] < sol[1][2] else (sol[1], sol[0])
    return (lo, hi) if both else lo


def delta_jac(p, q, L=LU, a=None):
    """Уравнение скоростей A pdot = B qdot: строки A — векторы штанг l_i, B = diag(b_i),
    b_i = L l_i^T w_i, w_i = -sin q_i u_i - cos q_i k. Возвращает A (м), B (м^2), J = A^-1 B (м/рад)."""
    if a is None:
        a = WB - UP
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    for i in range(3):
        u = uvec(i)
        s = np.asarray(p, float) - a * u - L * (np.cos(q[i]) * u - np.sin(q[i]) * KZ)   # вектор штанги l_i
        A[i] = s
        B[i, i] = L * s.dot(-np.sin(q[i]) * u - np.cos(q[i]) * KZ)
    return A, B, np.linalg.solve(A, B)


def delta_stiffness(p, k_q=1.0):
    """Матрица жёсткости платформы дельта-робота при крутильной жёсткости приводов k_q (Н·м/рад):
    K = J^-T K_q J^-1 = k_q A^T B^-2 A (Н/м). При k_q = 1 собственные числа равны 1/sigma_i^2(J)."""
    p = np.asarray(p, float); A, B, J = delta_jac(p, delta_ik(p))
    Bi = np.diag(1.0 / np.diag(B))
    return k_q * A.T.dot(Bi).dot(Bi).dot(A)


def delta_eval_many(P, L=LU, l=LL, a=None):
    """Векторный расчёт для массива точек P (N x 3): ОЗК (рабочая ветвь), признак принадлежности рабочей
    зоне и сингулярные числа J. Те же формулы, что в delta_ik и delta_jac, но без цикла по точкам.
    Возвращает словарь: q (N x 3, nan если недостижимо), inzone (N,), A (N x 3 x 3), b (N x 3),
    J (N x 3 x 3), sv (N x 3, по убыванию; nan вне зоны)."""
    if a is None:
        a = WB - UP
    P = np.atleast_2d(np.asarray(P, float)); N = len(P)
    U = np.array([uvec(i) for i in range(3)]); T = np.array([tvec(i) for i in range(3)])
    C = P[:, None, :] - a * U[None, :, :]                                  # c_i, N x 3 x 3
    E = -2 * L * np.einsum('nij,ij->ni', C, U); F = 2 * L * P[:, 2][:, None] * np.ones((1, 3))
    G = np.einsum('nij,nij->ni', C, C) + L ** 2 - l ** 2
    D = E * E + F * F - G * G
    ok = np.all(D >= 0, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Q = 2 * np.arctan((-F - np.sqrt(np.where(D >= 0, D, np.nan))) / (G - E))
    Q[~ok] = np.nan
    eta = np.einsum('nij,ij->ni', C, T)
    inzone = ok & np.all((Q >= Q_LIM[0]) & (Q <= Q_LIM[1]), axis=1) & np.all(np.abs(eta) <= l * np.sin(BETA_MAX), axis=1)
    cq, sq = np.cos(Q), np.sin(Q)
    S = C - L * (cq[:, :, None] * U[None] - sq[:, :, None] * KZ[None, None, :])   # векторы штанг
    Wv = -sq[:, :, None] * U[None] - cq[:, :, None] * KZ[None, None, :]
    b = L * np.einsum('nij,nij->ni', S, Wv)
    A = S.copy(); A[~ok] = np.eye(3); bb = b.copy(); bb[~ok] = 1.0
    J = np.linalg.solve(A, bb[:, :, None] * np.eye(3)[None])
    sv = np.linalg.svd(J, compute_uv=False)
    sv[~inzone] = np.nan; J[~ok] = np.nan
    return dict(q=Q, inzone=inzone, A=S, b=b, J=J, sv=sv)


def delta_quality(p):
    """Индекс обусловленности 1/kappa(J) в точке p; nan — точка вне рабочей зоны
    (ОЗК не решается, угол привода вне [-45; 95] градусов или штанга отклонена более чем на 40 градусов)."""
    sv = delta_eval_many(np.asarray(p, float)[None, :])['sv'][0]
    return sv[2] / sv[0]


def delta_section(plane='xz', value=0.0, n=90, lim=(-1.0, 1.0), zlim=(-1.85, -0.25), what='icond'):
    """Сечение рабочей зоны: plane='xz' (y = value), 'yz' (x = value) или 'xy' (z = value).
    what='icond' — 1/kappa(J); 'smax' — sigma_max(J), м/рад; 'smin' — sigma_min(J); nan вне зоны.
    Возвращает оси (горизонтальная, вертикальная) и карту M[строка = вертикальная ось, столбец]."""
    us = np.linspace(lim[0], lim[1], n)
    vs = np.linspace(zlim[0], zlim[1], n) if plane in ('xz', 'yz') else np.linspace(lim[0], lim[1], n)
    UU, VV = np.meshgrid(us, vs); cst = np.full(UU.shape, float(value))
    if plane == 'xz':
        P = np.stack([UU, cst, VV], -1)
    elif plane == 'yz':
        P = np.stack([cst, UU, VV], -1)
    else:
        P = np.stack([UU, VV, cst], -1)
    sv = delta_eval_many(P.reshape(-1, 3))['sv']
    val = {'icond': sv[:, 2] / sv[:, 0], 'smax': sv[:, 0], 'smin': sv[:, 2]}[what]
    return us, vs, val.reshape(UU.shape)


def delta_sigma_max(p):
    """Наибольшее сингулярное число J (м/рад) — коэффициент передачи ошибок датчиков; nan вне зоны."""
    return delta_eval_many(np.asarray(p, float)[None, :])['sv'][0, 0]


def delta_sv_grid(nx=35, ny=39, nz=43, xlim=(-0.85, 0.85), ylim=(-0.95, 0.95), zlim=(-1.80, -0.75)):
    """Сингулярные числа J на прямоугольной сетке точек рабочей зоны. Возвращает (точки зоны, sv[N, 3])."""
    g = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny), np.linspace(zlim[0], zlim[1], nz), indexing='ij')
    P = np.stack(g, -1).reshape(-1, 3); r = delta_eval_many(P)
    return P[r['inzone']], r['sv'][r['inzone']]


def delta_smax_peak(n=400, m=201, half=0.01):
    """Наибольшее sigma_max(J) в сечении x = 0 (плоскость ноги 1, где лежит максимум по зоне).
    Максимум достигается на границе зоны, поэтому после сетки n x n он уточняется на мелкой сетке
    m x m в квадрате +-half около лучшей точки. Возвращает (sigma_max, точка)."""
    us, vs, M = delta_section('yz', 0.0, n=n, lim=(-1.0, 1.0), zlim=(-1.85, -0.5), what='smax')
    iz, iy = np.unravel_index(np.nanargmax(M), M.shape)
    y0, z0 = us[iy], vs[iz]
    YY, ZZ = np.meshgrid(np.linspace(y0 - half, y0 + half, m), np.linspace(z0 - half, z0 + half, m))
    P = np.stack([np.zeros(YY.size), YY.ravel(), ZZ.ravel()], -1)
    s = delta_eval_many(P)['sv'][:, 0]; k = np.nanargmax(s)
    return s[k], P[k]


def delta_nu_det(P):
    """|det(nu_1, nu_2, nu_3)| нормалей параллелограммов ног R-4S для массива точек P (N x 3);
    nan вне рабочей зоны. nu_i = t_i x l_i / |t_i x l_i|."""
    r = delta_eval_many(np.atleast_2d(np.asarray(P, float)))
    T = np.array([tvec(i) for i in range(3)])
    with np.errstate(invalid='ignore'):
        NU = np.cross(T[None], r['A']); NU = NU / np.linalg.norm(NU, axis=2, keepdims=True)
        d = np.abs(np.linalg.det(np.nan_to_num(NU)))
    d[~r['inzone']] = np.nan
    return d


def delta_cloud(n=4000, seed=0):
    """Облако точек рабочей зоны через ПЗК: случайные углы приводов в допустимом диапазоне."""
    rng = np.random.RandomState(seed); pts = []
    for k in range(n):
        p = delta_fk(rng.uniform(Q_LIM[0], Q_LIM[1], 3))
        if p is not None:
            pts.append(p)
    pts = np.array(pts)
    return pts[delta_eval_many(pts)['inzone']]


# ---------------------------------------------------------------- связи ноги и подвижность по рангу
def mobility(n, j, fsum, h=6):
    """Структурная формула Чебышёва — Грюблера — Кутцбаха: W = h (n - j) + sum f_i.
    n — число подвижных звеньев (без стойки), j — число сочленений, fsum — сумма их подвижностей,
    h = 6 в пространстве и 3 на плоскости."""
    return h * (n - j) + fsum


def rank_nullity(H, rtol=1e-9):
    """Ранг матрицы связей H (через сингулярное разложение) и размерность её ядра = число неизвестных - ранг."""
    sv = np.linalg.svd(np.asarray(H, float), compute_uv=False)
    r = int(np.sum(sv > rtol * sv[0]))
    return r, H.shape[1] - r


def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


class ConstraintMatrix(object):
    """Матрица связей пространственного механизма в скоростях звеньев.
    Неизвестные — скорости (omega; v) всех nb подвижных звеньев (v — скорость точки звена, совпадающей
    с началом координат), всего 6*nb. Звено -1 — стойка. Сферический шарнир в точке c даёт 3 уравнения
    (скорости точки c на двух звеньях равны), вращательный с осью k — ещё 2 (относительная угловая
    скорость параллельна k), итого 5."""
    def __init__(self, nb):
        self.nb = nb; self.rows = []

    def _blk(self, body, M):
        r = np.zeros((M.shape[0], 6 * self.nb))
        if body >= 0:
            r[:, 6 * body:6 * body + 6] = M
        return r

    def S(self, a, b, c):
        M = np.hstack([-_skew(np.asarray(c, float)), np.eye(3)])      # v + omega x c = v - c x omega
        self.rows.append(self._blk(a, M) - self._blk(b, M))

    def R(self, a, b, c, k):
        self.S(a, b, c)
        k = np.asarray(k, float) / np.linalg.norm(k)
        u = np.cross(k, [1.0, 0, 0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(k, [0, 1.0, 0])
        u /= np.linalg.norm(u); w = np.cross(k, u)
        for e in (u, w):
            M = np.hstack([e[None, :], np.zeros((1, 3))])
            self.rows.append(self._blk(a, M) - self._blk(b, M))

    def H(self):
        return np.vstack(self.rows)


def delta_constraint_matrix(p, joint='4S', c=0.05):
    """Полная матрица связей дельта-робота в точке p (рабочая ветвь ОЗК).
    joint='4S' — нога R-4S: рычаг и две штанги на шаровых шарнирах (10 звеньев, 15 сочленений, 51 уравнение);
    joint='R'  — параллелограммы на вращательных шарнирах: рычаг, верхняя сторона, две штанги, нижняя
                 сторона (16 звеньев, 21 сочленение, 105 уравнений). c — полуширина параллелограмма, м.
    Звено 0 — платформа. Подвижность = размерность ядра (rank_nullity)."""
    p = np.asarray(p, float); q = delta_ik(p); knees = delta_knees(q)
    per = 3 if joint == '4S' else 5
    m = ConstraintMatrix(1 + 3 * per)
    for i in range(3):
        u = uvec(i); t = tvec(i); B = WB * u; K = knees[i]; Pi = p + UP * u
        lev = 1 + per * i
        m.R(lev, -1, B, t)                                    # привод: рычаг относительно стойки
        if joint == '4S':
            for s, r in ((1, lev + 1), (-1, lev + 2)):
                m.S(r, lev, K + s * c * t); m.S(r, 0, Pi + s * c * t)
        else:
            up, r1, r2, lo = lev + 1, lev + 2, lev + 3, lev + 4
            nu = np.cross(t, Pi - K); nu /= np.linalg.norm(nu)
            m.R(up, lev, K, t); m.R(lo, 0, Pi, t)            # стороны вращаются вокруг своих осей
            for s, r in ((1, r1), (-1, r2)):
                m.R(r, up, K + s * c * t, nu); m.R(r, lo, Pi + s * c * t, nu)   # плоский контур 4R
    return m.H()


def delta_loop_H(p, joint='4S', c=0.05):
    """Матрица уравнений замыкания контуров дельта-робота в шарнирных координатах (§ 5.11):
    H thetadot = 0, неизвестные — скорости всех сочленений (sum f_i штук: 1 на R, 3 на S),
    уравнения — по 6 на каждый из j - n независимых контуров (сумма твистов сочленений контура равна нулю).
    Те же звенья и сочленения, что в delta_constraint_matrix. Базис независимых контуров берётся как
    левое ядро блочной матрицы инцидентности «сочленение — звено». Подвижность = sum f_i - ранг H.
    Возвращает H размера 6 (j - n) x sum f_i."""
    p = np.asarray(p, float); q = delta_ik(p); knees = delta_knees(q)
    per = 3 if joint == '4S' else 5
    nb = 1 + 3 * per
    joints = []                                           # (звено a, звено b, столбцы-твисты (omega; v) в начале координат)

    def Rj(a, b, cc, k):
        k = np.asarray(k, float) / np.linalg.norm(k)
        joints.append((a, b, [np.hstack([k, np.cross(cc, k)])]))

    def Sj(a, b, cc):
        joints.append((a, b, [np.hstack([e, np.cross(cc, e)]) for e in np.eye(3)]))
    for i in range(3):
        u = uvec(i); t = tvec(i); B = WB * u; K = knees[i]; Pi = p + UP * u
        lev = 1 + per * i
        Rj(lev, -1, B, t)
        if joint == '4S':
            for s, r in ((1, lev + 1), (-1, lev + 2)):
                Sj(r, lev, K + s * c * t); Sj(r, 0, Pi + s * c * t)
        else:
            up, r1, r2, lo = lev + 1, lev + 2, lev + 3, lev + 4
            nu = np.cross(t, Pi - K); nu /= np.linalg.norm(nu)
            Rj(up, lev, K, t); Rj(lo, 0, Pi, t)
            for s, r in ((1, r1), (-1, r2)):
                Rj(r, up, K + s * c * t, nu); Rj(r, lo, Pi + s * c * t, nu)
    nj = len(joints)
    D = np.zeros((6 * nj, 6 * nb)); S = np.zeros((6 * nj, sum(len(j[2]) for j in joints))); col = 0
    for k, (a, b, cols) in enumerate(joints):     # V_a - V_b = S_k thetadot_k  (стойка: V = 0)
        r = slice(6 * k, 6 * k + 6)
        if a >= 0:
            D[r, 6 * a:6 * a + 6] += np.eye(6)
        if b >= 0:
            D[r, 6 * b:6 * b + 6] -= np.eye(6)
        for cc in cols:
            S[r, col] = cc; col += 1
    U, sv, Vt = np.linalg.svd(D.T)
    Lam = Vt[int(np.sum(sv > 1e-9 * sv[0])):]           # 6 (j - n) строк: базис контуров, Lam D = 0
    return Lam.dot(S)


def delta_leg_constraints(p, i, q=None, c=0.05):
    """Связи одной ноги R-4S на скорость платформы (omega; v) и скорости приводов qdot (9 неизвестных):
    по одному уравнению нерастяжимости на каждую из двух штанг. Возвращает (строки 2 x 9, нормаль nu_i)."""
    p = np.asarray(p, float)
    q = delta_ik(p) if q is None else q
    u = uvec(i); t = tvec(i); K = delta_knees(q)[i]; Pi = p + UP * u
    w = -np.sin(q[i]) * u - np.cos(q[i]) * KZ
    rows = []
    for s in (1, -1):
        bk = UP * u + s * c * t                               # точка платформы относительно её центра
        sk = (Pi + s * c * t) - (K + s * c * t)               # вектор штанги
        r = np.zeros(9); r[0:3] = np.cross(bk, sk); r[3:6] = sk; r[6 + i] = -LU * sk.dot(w)
        rows.append(r)
    nu = np.cross(t, Pi - K)
    return np.array(rows), nu / np.linalg.norm(nu)


def pgram3_loop_H(theta=np.radians(60.0), l=1.0, base=1.0, l3=None):
    """Уравнения замыкания контуров плоского параллелограмма с тремя кривошипами.
    Шарниры стойки O_k = ((k-1)*base, 0), кривошипы длины l под углом theta, шатун горизонтален.
    Неизвестные — 6 шарнирных углов: a1, a2, a3 (кривошипы относительно стойки) и t1, t2, t3 (шатун
    относительно кривошипов). Два контура O1-кривошип 1-шатун-кривошип k-O_k (k = 2, 3) дают по 3
    уравнения (x, y, угол). Возвращает H = dg/d(углы) размера 6 x 6 (центральные разности)."""
    l3 = l if l3 is None else l3
    e = lambda x: np.array([np.cos(x), np.sin(x)])

    def g(v):
        a1, a2, a3, t1, t2, t3 = v
        b1, b2, b3 = a1 + t1, a2 + t2, a3 + t3
        r = []
        for ak, bk, Ok, sk, lk in ((a2, b2, np.array([base, 0.0]), base, l), (a3, b3, np.array([2 * base, 0.0]), 2 * base, l3)):
            r += list(l * e(a1) + sk * e(b1) - Ok - lk * e(ak)); r.append(b1 - bk)
        return np.array(r)
    v0 = np.array([theta, theta, theta, -theta, -theta, -theta])
    H = np.zeros((6, 6)); hh = 1e-7
    for k in range(6):
        d = np.zeros(6); d[k] = hh; H[:, k] = (g(v0 + d) - g(v0 - d)) / (2 * hh)
    return H


# ====================================================================== линейный дельта-робот 3-PUU
# параметры настольного 3D-принтера (по Р. Уильямсу), метры
LIN_SB, LIN_SP, LIN_L = 0.246, 0.127, 0.264
LIN_RB = np.sqrt(3) / 3 * LIN_SB                   # r_B — радиус окружности направляющих
LIN_UP = np.sqrt(3) / 3 * LIN_SP
LIN_GAMMA = np.radians([210.0, 330.0, 90.0])       # азимуты направляющих (как у Уильямса и в Marlin)
LIN_PHI = LIN_GAMMA                                # старое имя


def lin_u(i):
    return np.array([np.cos(LIN_GAMMA[i]), np.sin(LIN_GAMMA[i]), 0.0])


def lindelta_ik(p, l=LIN_L, a=None):
    """ОЗК линейного дельта-робота: d_i = -z - sqrt(l^2 - rho_i^2) (каретка выше платформы);
    d_i — опускание каретки от плоскости основания (положительно вниз). None — недостижимо."""
    if a is None:
        a = LIN_RB - LIN_UP
    p = np.asarray(p, float); d = np.zeros(3)
    for i in range(3):
        c = p[:2] - a * lin_u(i)[:2]
        D = l ** 2 - c.dot(c)
        if D < 0:
            return None
        d[i] = -p[2] - np.sqrt(D)
    return d


def lindelta_fk(d, l=LIN_L, a=None):
    """ПЗК линейного дельта-робота: три сферы радиуса l с центрами a u_i - d_i k, нижнее решение."""
    if a is None:
        a = LIN_RB - LIN_UP
    cs = [a * lin_u(i) - d[i] * KZ for i in range(3)]
    sol = trilaterate(cs[0], cs[1], cs[2], l, l, l)
    if sol is None:
        return None
    return sol[0] if sol[0][2] < sol[1][2] else sol[1]


def lindelta_jac(p, l=LIN_L, a=None):
    """A pdot = B ddot для линейного дельта-робота: строки A — векторы штанг, b_i = -(z + d_i)."""
    if a is None:
        a = LIN_RB - LIN_UP
    p = np.asarray(p, float); d = lindelta_ik(p, l, a)
    A = np.array([p - a * lin_u(i) + d[i] * KZ for i in range(3)])
    B = np.diag([-(p[2] + d[i]) for i in range(3)])
    return A, B, np.linalg.solve(A, B)


def lindelta_error(p, da=0.0, dl=0.0):
    """«Сломай и посмотри»: прошивка считает ОЗК с ошибочными параметрами (a + da, l + dl), механизм движется
    по истинной геометрии. Возвращает отклонение фактического положения от заданного p (м)."""
    a0 = LIN_RB - LIN_UP
    d = lindelta_ik(p, l=LIN_L + dl, a=a0 + da)
    return lindelta_fk(d, l=LIN_L, a=a0) - np.asarray(p, float)


# ====================================================================== плоский пятизвенник 5R
FB_D, FB_L1, FB_L2 = 0.30, 0.30, 0.45              # расстояние между приводами d, ведущие l1 и ведомые l2 звенья


def fivebar_fk(q1, q2, mode=+1, d=FB_D, l1=FB_L1, l2=FB_L2):
    """ПЗК пятизвенника: пересечение окружностей радиуса l2 с центрами B1, B2:
    P = (B1 + B2)/2 + mode * b * n, b = sqrt(l2^2 - D^2/4), n — нормаль к e = B2 - B1 (поворот на +90°).
    mode=+1 — сборка «+» (точка слева от вектора B1B2), -1 — сборка «-». Возвращает (P или None, B1, B2)."""
    B1 = np.array([-d / 2 + l1 * np.cos(q1), l1 * np.sin(q1)])
    B2 = np.array([d / 2 + l1 * np.cos(q2), l1 * np.sin(q2)])
    e = B2 - B1; D = np.linalg.norm(e)
    b2 = l2 ** 2 - D ** 2 / 4
    if b2 < 0 or D < 1e-12:
        return None, B1, B2
    n = np.array([-e[1], e[0]]) / D
    return (B1 + B2) / 2 + mode * np.sqrt(b2) * n, B1, B2


def fivebar_ik(P, s1=+1, s2=-1, d=FB_D, l1=FB_L1, l2=FB_L2):
    """ОЗК пятизвенника: каждая нога — двухзвенник; q_i = atan2(y, x - x_Ai) + eps_i arccos(...),
    s1, s2 = eps_1, eps_2 = +-1 — рабочий режим. None — точка недостижима."""
    out = []
    for xa, s in ((-d / 2, s1), (d / 2, s2)):
        v = np.array([P[0] - xa, P[1]]); rho = np.linalg.norm(v)
        c = (rho ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * rho)
        if abs(c) > 1:
            return None
        out.append(np.arctan2(v[1], v[0]) + s * np.arccos(c))
    return np.array(out)


def fivebar_jac(P, q, d=FB_D, l1=FB_L1, l2=FB_L2):
    """Матрицы уравнения скоростей пятизвенника A Pdot = B qdot (строки A — векторы ведомых звеньев,
    b_i = l1 (P - B_i)^T e_i) и J = A^-1 B (None, если det A = 0)."""
    B1 = np.array([-d / 2 + l1 * np.cos(q[0]), l1 * np.sin(q[0])])
    B2 = np.array([d / 2 + l1 * np.cos(q[1]), l1 * np.sin(q[1])])
    P = np.asarray(P, float)
    A = np.array([P - B1, P - B2])
    e = [np.array([-np.sin(q[k]), np.cos(q[k])]) for k in range(2)]
    B = np.diag([l1 * (P - B1).dot(e[0]), l1 * (P - B2).dot(e[1])])
    J = np.linalg.solve(A, B) if abs(np.linalg.det(A)) > 1e-12 else None
    return A, B, J


def fivebar_psi(P, q, d=FB_D, l1=FB_L1):
    """Ориентированные углы psi_i от ведущего звена к ведомому (против часовой стрелки), рад:
    b_i = l1 l2 sin psi_i."""
    P = np.asarray(P, float); out = []
    for xa, qi in ((-d / 2, q[0]), (d / 2, q[1])):
        lead = np.array([np.cos(qi), np.sin(qi)]); Bk = np.array([xa, 0.0]) + l1 * lead; drv = P - Bk
        out.append(np.arctan2(lead[0] * drv[1] - lead[1] * drv[0], lead.dot(drv)))
    return np.array(out)


def fivebar_limits_ok(P, ymin=0.1):
    """Паспортное ограничение зоны пятизвенника примера 5.1: точка выше оси приводов, y >= ymin.
    Оно исключает особенность третьего типа в q = (180°; 0°), P = (0; 0) и особенности второго типа
    B1 = B2, у которых в режиме (+1, -1) точка P лежит ниже оси приводов (§ 5.11)."""
    return P[1] >= ymin


def fivebar_workspace(n=140, mode=+1, s1=+1, s2=-1, xlim=(-0.8, 0.8), ylim=(-0.1, 0.8), ymin=None, metric='icond'):
    """Карта пятизвенника по зоне выбранного рабочего режима (s1, s2) и режима сборки mode (через ОЗК).
    metric='icond' — 1/kappa(J); 'sminA' — наименьшее сингулярное число A/l2. ymin — паспортное
    ограничение fivebar_limits_ok (None — без ограничений, как на рис. 5.3).
    Те же формулы, что в fivebar_ik, fivebar_fk и fivebar_jac, но сразу для всей сетки (без цикла по точкам).
    Размеры — по умолчанию примера 5.1 (FB_D, FB_L1, FB_L2)."""
    d, l1, l2 = FB_D, FB_L1, FB_L2
    xs = np.linspace(xlim[0], xlim[1], n); ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)                                   # строка — y, столбец — x
    ok = np.ones(X.shape, bool) if ymin is None else (Y >= ymin)
    Q, Bs = [], []
    for xa, s in ((-d / 2, s1), (d / 2, s2)):                    # ОЗК по ногам
        rho = np.hypot(X - xa, Y)
        with np.errstate(invalid='ignore', divide='ignore'):
            c = (rho ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * rho)
        ok &= np.abs(c) <= 1
        q = np.arctan2(Y, X - xa) + s * np.arccos(np.clip(c, -1, 1)); Q.append(q)
        Bs.append(np.stack([xa + l1 * np.cos(q), l1 * np.sin(q)], -1))
    P = np.stack([X, Y], -1); e = Bs[1] - Bs[0]; D = np.linalg.norm(e, axis=-1)
    b2 = l2 ** 2 - D ** 2 / 4; ok &= (b2 >= 0) & (D >= 1e-12)    # ПЗК выбранного режима сборки
    with np.errstate(invalid='ignore', divide='ignore'):
        nrm = np.stack([-e[..., 1], e[..., 0]], -1) / D[..., None]
        Pm = (Bs[0] + Bs[1]) / 2 + mode * np.sqrt(np.maximum(b2, 0))[..., None] * nrm
    ok &= np.linalg.norm(Pm - P, axis=-1) <= 1e-7                # иначе точка принадлежит другому режиму сборки
    A = np.stack([P - Bs[0], P - Bs[1]], -2)                     # строки A — векторы ведомых звеньев
    bdiag = np.stack([l1 * np.sum((P - Bs[k]) * np.stack([-np.sin(Q[k]), np.cos(Q[k])], -1), -1) for k in range(2)], -1)
    A[~ok] = np.eye(2)
    M = np.full(X.shape, np.nan)
    if metric == 'sminA':
        sv = np.linalg.svd(A / l2, compute_uv=False); M[ok] = sv[ok][:, -1]
    else:
        good = ok & (np.abs(np.linalg.det(A)) > 1e-12); A[~good] = np.eye(2); bdiag[~good] = 1.0
        J = np.linalg.solve(A, bdiag[..., None] * np.eye(2))
        sv = np.linalg.svd(J, compute_uv=False)
        with np.errstate(invalid='ignore', divide='ignore'):
            M[good] = (sv[..., -1] / sv[..., 0])[good]
    return xs, ys, M


# ====================================================================== параллелограммный манипулятор
PG_H, PG_L2, PG_L3, PG_D, PG_E = 0.80, 1.00, 1.20, 0.20, 0.25   # h0, l2, l3, d, e (метры)
PG_GAMMA = np.radians([25.0, 155.0])               # допустимый угол q2 - q3 между плечом и предплечьем


def pgram_fk(q, h0=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E, h=None):
    """q1 — поворот колонны; q2 — угол плеча к горизонту; q3 — угол предплечья к горизонту
    (оба привода на оси плеча, q3 = q2 + theta3 передаётся параллелограммом). Фланец горизонтален.
    r = l2 cos q2 + l3 cos q3 + d, z = h0 + l2 sin q2 + l3 sin q3 - e. Ключ h — старое имя h0."""
    h0 = h0 if h is None else h
    q1, q2, q3 = q
    r = l2 * np.cos(q2) + l3 * np.cos(q3) + d
    z = h0 + l2 * np.sin(q2) + l3 * np.sin(q3) - e
    return np.array([r * np.cos(q1), r * np.sin(q1), z])


def pgram_ik(p, h0=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E, h=None):
    """ОЗК: q1 = atan2(y, x); в плоскости руки двухзвенник с theta3 = -arccos(...) («локоть вверх»);
    возвращает углы приводов (q1, q2, q3 = q2 + theta3) или None."""
    h0 = h0 if h is None else h
    x, y, z = p
    q1 = np.arctan2(y, x)
    rr = np.hypot(x, y) - d; zz = z - h0 + e
    c = (rr ** 2 + zz ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    if abs(c) > 1:
        return None
    th3 = -np.arccos(c)
    q2 = np.arctan2(zz, rr) - np.arctan2(l3 * np.sin(th3), l2 + l3 * np.cos(th3))
    return np.array([q1, q2, q2 + th3])


def pgram_jac_rz(q, l2=PG_L2, l3=PG_L3):
    """Матрица Якоби в плоскости руки: (rdot, zdot) = J_rz (q2dot, q3dot); det J_rz = l2 l3 sin(q3 - q2)."""
    return np.array([[-l2 * np.sin(q[1]), -l3 * np.sin(q[2])], [l2 * np.cos(q[1]), l3 * np.cos(q[2])]])


def pgram_limits_ok(q, lim2=(20.0, 130.0), lim3=(-75.0, 35.0)):
    g = q[1] - q[2]
    return (np.radians(lim2[0]) <= q[1] <= np.radians(lim2[1]) and np.radians(lim3[0]) <= q[2] <= np.radians(lim3[1])
            and PG_GAMMA[0] <= g <= PG_GAMMA[1])


def pgram_workspace(n=220, rlim=(-0.2, 2.6), zlim=(-0.55, 2.5)):
    """Сечение рабочей зоны в плоскости (r, z); значение — |sin(q2 - q3)| (пропорционально det J_rz)."""
    rs = np.linspace(rlim[0], rlim[1], n); zs = np.linspace(zlim[0], zlim[1], n)
    M = np.full((n, n), np.nan)
    for iz, z in enumerate(zs):
        for ir, r in enumerate(rs):
            if r <= 0:
                continue
            q = pgram_ik([r, 0.0, z])
            if q is None or not pgram_limits_ok(q):
                continue
            M[iz, ir] = abs(np.sin(q[1] - q[2]))
    return rs, zs, M


def pgram_dh_robot(h0=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E):
    """Эквивалентная последовательная рука в roboticstoolbox (DHRobot), классические ДХ:
    1) theta = q1, d = h0, alpha = pi/2 (колонна); 2) theta = q2, a = l2 (плечо);
    3) theta = theta3 = q3 - q2, a = l3 (локоть); 4) theta4 = -q3 (параллелограммная цепочка держит
    фланец горизонтально); инструмент — сдвиг (d, -e, 0) в горизонтальной системе фланца.
    Углы приводов (q1, q2, q3) переводит в углы ДХ функция pgram_q_to_dh."""
    import roboticstoolbox as rtb
    from spatialmath import SE3
    links = [rtb.RevoluteDH(d=h0, a=0.0, alpha=np.pi / 2), rtb.RevoluteDH(a=l2), rtb.RevoluteDH(a=l3), rtb.RevoluteDH(a=0.0)]
    return rtb.DHRobot(links, name='pgram_equiv', tool=SE3(d, -e, 0.0))


def pgram_q_to_dh(q):
    """Углы приводов (q1, q2, q3) -> углы ДХ эквивалентной руки (q1, q2, theta3 = q3 - q2, theta4 = -q3)."""
    q1, q2, q3 = q
    return np.array([q1, q2, q3 - q2, -q3])


def pgram_points(q, c=0.42, h0=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E):
    """Характерные точки в плоскости руки (r, z): ось плеча O2, локоть E, конец предплечья W,
    фланец F, рабочая точка TCP, конец кривошипа C и конец хвостовика предплечья Rr."""
    e2 = np.array([np.cos(q[1]), np.sin(q[1])]); e3 = np.array([np.cos(q[2]), np.sin(q[2])])
    O2 = np.array([0.0, h0]); E = O2 + l2 * e2; W = E + l3 * e3
    F = W + np.array([d, 0.0]); TCP = F - np.array([0.0, e])
    return dict(O2=O2, E=E, W=W, F=F, TCP=TCP, C=O2 - c * e3, Rr=E - c * e3)


# ====================================================================== платформа Гью — Стюарта 6-UPS
ST_RA, ST_RB, ST_DA, ST_DB = 0.60, 0.35, 12.0, 12.0   # радиусы окружностей шарниров и полууглы пар, градусы


def stewart_geom():
    """Точки крепления ног: a_i на основании (неподвижная система), b_i на платформе (её система).
    Углы a_i: -12, 12, 108, 132, 228, 252°; b_i: -48, 48, 72, 168, 192, 288° (таблица примера 5.7)."""
    ang_a = np.radians([0 - ST_DA, 0 + ST_DA, 120 - ST_DA, 120 + ST_DA, 240 - ST_DA, 240 + ST_DA])
    ang_b = np.radians([-60 + ST_DB, 60 - ST_DB, 60 + ST_DB, 180 - ST_DB, 180 + ST_DB, 300 - ST_DB])
    a = np.array([[ST_RA * np.cos(t), ST_RA * np.sin(t), 0] for t in ang_a])
    b = np.array([[ST_RB * np.cos(t), ST_RB * np.sin(t), 0] for t in ang_b])
    return a, b


def rpy(phi, theta, psi):
    """Углы крена, тангажа и рыскания (3-2-1): R = Rz(psi) Ry(theta) Rx(phi)."""
    ca, sa, cb, sb, cg, sg = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz.dot(Ry).dot(Rx)


def stewart_pose(Theta, p):
    """Поза платформы x = (Theta; p) = (phi, theta, psi, px, py, pz) из углов и положения."""
    return np.r_[np.asarray(Theta, float), np.asarray(p, float)]


def stewart_ik(x):
    """x = (phi, theta, psi, px, py, pz). Длины ног l_i = |p + R b_i - a_i|."""
    x = np.asarray(x, float)
    a, b = stewart_geom()
    R = rpy(*x[:3])
    return np.linalg.norm(x[3:] + b.dot(R.T) - a, axis=1)


def stewart_Jl(x, hh=1e-7):
    """J_l = dl/dx — производные длин ног по x = (Theta; p) (центральные разности)."""
    x = np.asarray(x, float); Jm = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros(6); dx[j] = hh
        Jm[:, j] = (stewart_ik(x + dx) - stewart_ik(x - dx)) / (2 * hh)
    return Jm


def stewart_fk_hist(lens, x0, tol=1e-12, itmax=40):
    """ПЗК платформы методом Ньютона x_{k+1} = x_k - J_l^-1 (l(x_k) - l*).
    Возвращает (x, список max|l(x_k) - l*| по итерациям, начиная с x0)."""
    x = np.array(x0, float); hist = []
    for k in range(itmax):
        f = stewart_ik(x) - lens
        hist.append(np.abs(f).max())
        if hist[-1] < tol or not np.isfinite(hist[-1]):
            break
        try:
            x = x - np.linalg.solve(stewart_Jl(x), f)
        except np.linalg.LinAlgError:
            hist.append(np.inf); break
    return x, hist


def stewart_fk(lens, x0, tol=1e-12, itmax=30):
    """ПЗК методом Ньютона; возвращает (x, число итераций)."""
    x, hist = stewart_fk_hist(lens, x0, tol, itmax)
    return x, len(hist) - 1


def stewart_invjac(x):
    """Обратная матрица Якоби платформы: ldot = J^-1 V, V = (omega; v);
    строка i = ((R b_i x n_i)^T, n_i^T) — плюккеровы координаты оси ноги."""
    x = np.asarray(x, float)
    a, b = stewart_geom(); R = rpy(*x[:3])
    Jinv = np.zeros((6, 6))
    for i in range(6):
        rb = R.dot(b[i]); s = x[3:] + rb - a[i]; n = s / np.linalg.norm(s)
        Jinv[i, :3] = np.cross(rb, n); Jinv[i, 3:] = n
    return Jinv


def angle_rates_inv(Theta):
    """W^-1(Theta): omega = W^-1 Thetadot для R = Rz(psi) Ry(theta) Rx(phi); det W^-1 = cos(theta)."""
    phi, th, psi = Theta
    cps, sps, cth, sth = np.cos(psi), np.sin(psi), np.cos(th), np.sin(th)
    return np.array([[cps * cth, -sps, 0.0], [sps * cth, cps, 0.0], [-sth, 0.0, 1.0]])


def stewart_wrench(x, tau):
    """Силовой винт F = (m; f), который ноги с усилиями tau передают через платформу: F = J^-T tau."""
    return stewart_invjac(x).T.dot(tau)


def stewart_leg_forces(x, F):
    """Усилия в ногах tau = J^T F, уравновешивающие силовой винт F = (m; f) (решение J^-T tau = F).
    Положительное усилие толкает платформу от основания вдоль n_i. При det J^-1 = 0 — inf."""
    try:
        return np.linalg.solve(stewart_invjac(x).T, np.asarray(F, float))
    except np.linalg.LinAlgError:
        return np.full(6, np.inf)


# ====================================================================== проверки и вспомогательное
def seg_dist(p1, p2, q1, q2, n=200):
    """Наименьшее расстояние между отрезками p1p2 и q1q2 (перебором по сетке параметров), м."""
    s = np.linspace(0, 1, n)[:, None]
    P = p1 + s * (np.asarray(p2) - p1); Q = q1 + s * (np.asarray(q2) - q1)
    return np.min(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2))


def check():
    deg = np.degrees
    print('--- delta 3-RUU: sB=%.3f sP=%.3f L=%.3f l=%.3f wB=%.5f uP=%.5f a=wB-uP=%.5f' % (SB, SP, LU, LL, WB, UP, WB - UP))
    for p in ([0, 0, -0.9], [0.3, 0.5, -1.1]):
        q = delta_ik(p); q_in = delta_ik(p, branch=+1)
        print('IK', p, '-> q(deg) =', np.round(deg(q), 3), ' inner branch:', np.round(deg(q_in), 2), ' FK(IK) err = %.2e' % np.linalg.norm(delta_fk(q) - p))
    for qd in ([0, 0, 0], [10, 20, 30]):
        p = delta_fk(np.radians(qd)); q = delta_ik(p)
        print('FK', qd, '-> p =', np.round(p, 4), ' IK(FK) err(deg) = %.2e' % np.abs(deg(q) - qd).max())
    p0 = np.array([0, 0, -0.9]); A, B, J = delta_jac(p0, delta_ik(p0))
    print('J at (0,0,-0.9):', np.round(J, 4).tolist(), ' sv =', np.round(np.linalg.svd(J)[1], 4))
    print('--- linear delta: rB=%.5f uP=%.5f a=%.5f l=%.3f' % (LIN_RB, LIN_UP, LIN_RB - LIN_UP, LIN_L))
    for p in ([0, 0, -0.5], [0.03, 0.05, -0.40]):
        print('IK', p, '-> d =', np.round(lindelta_ik(p), 4))
    print('--- fivebar d=%.2f l1=%.2f l2=%.2f' % (FB_D, FB_L1, FB_L2))
    P, B1, B2 = fivebar_fk(np.radians(100), np.radians(30), +1)
    print('q=(100,30): P+ =', np.round(P, 4), ' IK back (deg):', np.round(deg(fivebar_ik(P)), 4))
    print('--- parallelogram arm')
    q = np.radians([30, 60, -20]); p = pgram_fk(q)
    print('FK q=(30,60,-20) -> p =', np.round(p, 4), ' IK back (deg) =', np.round(deg(pgram_ik(p)), 4))
    print('--- Stewart')
    x0 = stewart_pose([0, 0, 0], [0, 0, 0.70]); print('home lengths:', np.round(stewart_ik(x0), 4))
    x = stewart_pose(np.radians([5, -8, 10]), [0.05, -0.03, 0.75])
    ln = stewart_ik(x); xs, hist = stewart_fk_hist(ln, x0)
    print('lengths', np.round(ln, 4), ' Newton:', ['%.1e' % h for h in hist])


# ====================================================================== рисование
NAVY, AMB, TEAL, RED, GREY, ICE, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#CADCFC', '#2C7A5A'


def draw_fivebar(ax, q1, q2, mode=+1, color=NAVY, alpha=1.0, label=None, d=FB_D):
    """Скелет пятизвенника в заданном режиме сборки. Возвращает точку P (или None)."""
    P, B1, B2 = fivebar_fk(q1, q2, mode)
    A1 = np.array([-d / 2, 0.0]); A2 = np.array([d / 2, 0.0])
    ax.plot([A1[0], A2[0]], [0, 0], color=GREY, lw=5, alpha=0.5, solid_capstyle='round')
    ax.plot([A1[0], B1[0]], [A1[1], B1[1]], color=color, lw=4, alpha=alpha, solid_capstyle='round')
    ax.plot([A2[0], B2[0]], [A2[1], B2[1]], color=color, lw=4, alpha=alpha, solid_capstyle='round')
    if P is not None:
        ax.plot([B1[0], P[0], B2[0]], [B1[1], P[1], B2[1]], color=color, lw=2.5, alpha=alpha, label=label)
        ax.plot([P[0]], [P[1]], 's', color=RED, ms=7, alpha=alpha, zorder=6)
    ax.plot([A1[0], A2[0]], [0, 0], 'o', color=AMB, ms=10, mec=NAVY, zorder=6)
    ax.plot([B1[0], B2[0]], [B1[1], B2[1]], 'o', color='white', ms=7, mec=color, alpha=alpha, zorder=6)
    return P


def draw_pgram(ax, q, color=NAVY, alpha=1.0, lw=4):
    """Скелет параллелограммного манипулятора в плоскости руки (r, z)."""
    P = pgram_points(q)
    ax.plot([0, 0], [0, PG_H], color=GREY, lw=lw + 3, alpha=0.5 * alpha, solid_capstyle='round')
    ax.plot([P['O2'][0], P['E'][0]], [P['O2'][1], P['E'][1]], color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    ax.plot([P['Rr'][0], P['W'][0]], [P['Rr'][1], P['W'][1]], color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    ax.plot([P['O2'][0], P['C'][0], P['Rr'][0]], [P['O2'][1], P['C'][1], P['Rr'][1]], color=TEAL, lw=2, alpha=alpha)
    ax.plot([P['W'][0], P['F'][0], P['TCP'][0]], [P['W'][1], P['F'][1], P['TCP'][1]], color=AMB, lw=3, alpha=alpha, solid_capstyle='round')
    ax.plot([P['TCP'][0]], [P['TCP'][1]], 's', color=RED, ms=6, alpha=alpha, zorder=6)
    ax.plot([P['O2'][0]], [P['O2'][1]], 'o', color=AMB, ms=10, mec=NAVY, alpha=alpha, zorder=6)
    ax.plot([P['E'][0], P['W'][0]], [P['E'][1], P['W'][1]], 'o', color='white', ms=6, mec=color, alpha=alpha, zorder=6)
    return P


def _tri(ax, c, r, phis, z, color, lw=2, alpha=1.0):
    v = np.array([[c[0] + r * np.cos(f), c[1] + r * np.sin(f), z] for f in list(phis) + [phis[0]]])
    ax.plot(v[:, 0], v[:, 1], v[:, 2], color=color, lw=lw, alpha=alpha)


def draw_delta(ax, p, q=None, color=NAVY, alpha=1.0, par=0.06, leg_colors=None, limits=True):
    """Трёхмерный скелет дельта-робота 3-RUU: основание, рычаги, штанги (парами), платформа.
    leg_colors — цвета рычагов по ногам (например, чтобы выделить ногу с неверной ветвью)."""
    p = np.asarray(p, float)
    if q is None:
        q = delta_ik(p)
    _tri(ax, (0, 0), UB, GAMMA + np.pi, 0.0, GREY, lw=2.5)               # вершины основания — напротив ног
    _tri(ax, p[:2], UP, GAMMA, p[2], AMB, lw=2.5, alpha=alpha)
    knees = delta_knees(q)
    for i in range(3):
        u = uvec(i); tv = tvec(i); B = WB * u; K = knees[i]; Pi = p + UP * u
        col = color if leg_colors is None else leg_colors[i]
        ax.plot([B[0], K[0]], [B[1], K[1]], [B[2], K[2]], color=col, lw=4, alpha=alpha, solid_capstyle='round')
        for s in (-1, 1):
            ax.plot([K[0] + s * par * tv[0], Pi[0] + s * par * tv[0]], [K[1] + s * par * tv[1], Pi[1] + s * par * tv[1]],
                    [K[2], Pi[2]], color=TEAL, lw=1.6, alpha=alpha)
        ax.scatter([B[0]], [B[1]], [B[2]], color=AMB, s=50, edgecolors=NAVY, depthshade=False)
    ax.scatter([p[0]], [p[1]], [p[2]], color=RED, s=30, depthshade=False)
    if limits:
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-1.5, 0.1)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')
    return knees


def draw_lindelta(ax, p, d=None, rail=0.45):
    """Скелет линейного дельта-робота 3-PUU с вертикальными направляющими."""
    p = np.asarray(p, float)
    if d is None:
        d = lindelta_ik(p)
    for i in range(3):
        u = lin_u(i); top = LIN_RB * u; car = LIN_RB * u - d[i] * KZ; Pi = p + LIN_UP * u
        ax.plot([top[0], top[0]], [top[1], top[1]], [0, -rail], color=GREY, lw=3, alpha=0.6)
        ax.plot([car[0], Pi[0]], [car[1], Pi[1]], [car[2], Pi[2]], color=TEAL, lw=2)
        ax.scatter([car[0]], [car[1]], [car[2]], color=AMB, s=50, edgecolors=NAVY, depthshade=False)
    _tri(ax, p[:2], LIN_UP, LIN_GAMMA, p[2], AMB, lw=2.5)
    ax.scatter([p[0]], [p[1]], [p[2]], color=RED, s=30, depthshade=False)
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')


def draw_stewart(ax, x, color=NAVY, alpha=1.0, legs=True, leg_colors=None):
    """Скелет платформы 6-UPS в позе x = (phi, theta, psi, px, py, pz)."""
    x = np.asarray(x, float)
    a, b = stewart_geom(); R = rpy(*x[:3]); top = x[3:] + b.dot(R.T)
    aa = np.vstack([a, a[:1]]); tt = np.vstack([top, top[:1]])
    ax.plot(aa[:, 0], aa[:, 1], aa[:, 2], color=GREY, lw=2.5)
    ax.plot(tt[:, 0], tt[:, 1], tt[:, 2], color=AMB, lw=2.5, alpha=alpha)
    if legs:
        for i in range(6):
            col = color if leg_colors is None else leg_colors[i]
            ax.plot([a[i, 0], top[i, 0]], [a[i, 1], top[i, 1]], [a[i, 2], top[i, 2]], color=col, lw=2.2, alpha=alpha)
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], color=GREY, s=25, depthshade=False)
    ax.scatter(top[:, 0], top[:, 1], top[:, 2], color=AMB, s=25, depthshade=False, alpha=alpha)
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')
    return top


if __name__ == '__main__':
    check()
