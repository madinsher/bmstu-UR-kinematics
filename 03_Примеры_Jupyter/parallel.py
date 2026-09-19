# -*- coding: utf-8 -*-
"""parallel — учебный модуль кинематики механизмов с замкнутыми цепями (занятие 5).

Расчётные функции (скопированы без изменений из генератора схем gen_figs_k5.py, числа
совпадают с конспектом лекции 5):
  delta_ik, delta_fk, delta_jac      — дельта-робот 3-RUU (вращательные приводы);
  lindelta_ik, lindelta_fk           — дельта-робот 3-PUU (линейные приводы);
  trilaterate                        — пересечение трёх сфер;
  fivebar_fk, fivebar_ik             — плоский пятизвенник 5R;
  pgram_fk, pgram_ik                 — параллелограммный манипулятор;
  stewart_geom, stewart_ik, stewart_fk, rpy — платформа Гью — Стюарта 6-UPS;
  check                              — самопроверка (печать чисел примеров 5.1–5.7).
Дополнения для ноутбука:
  mobility                           — структурная формула подвижности;
  delta_ik_table, fivebar_jac, pgram_jac_rz, stewart_fk_hist, stewart_invjac;
  delta_quality, delta_section, delta_cloud, pgram_workspace, fivebar_workspace — рабочие зоны;
  draw_fivebar, draw_pgram, draw_delta, draw_lindelta, draw_stewart — скелеты механизмов.
Только numpy и matplotlib.
"""
import numpy as np

# ====================================================================== расчётные функции
# ---- дельта-робот: параметры ABB FlexPicker IRB 360-1/1600 (по Р. Уильямсу), метры
SB, SP, LU, LL = 0.567, 0.076, 0.524, 1.244      # сторона базы, сторона платформы, рычаг L, штанга l
WB = np.sqrt(3) / 6 * SB                           # от центра базы до оси привода (середина стороны)
UB = np.sqrt(3) / 3 * SB                           # от центра базы до вершины
WP = np.sqrt(3) / 6 * SP
UP = np.sqrt(3) / 3 * SP                           # от центра платформы до шарнира (вершина)
PHI = np.radians([-90.0, 30.0, 150.0])             # азимуты ног
KZ = np.array([0.0, 0.0, 1.0])


def uvec(i):
    return np.array([np.cos(PHI[i]), np.sin(PHI[i]), 0.0])


def delta_ik(p, L=LU, l=LL, a=None, branch=-1):
    """ОЗК дельта-робота 3-RUU. p=(x,y,z), z<0. Возвращает углы theta (рад), отсчёт от
    горизонтали вниз. E cos(th) + F sin(th) + G = 0,  t = tg(th/2).
    branch=-1 — колени наружу (рабочая ветвь), +1 — колени внутрь. None — точка недостижима."""
    if a is None:
        a = WB - UP
    p = np.asarray(p, float)
    th = np.zeros(3)
    for i in range(3):
        c = p - a * uvec(i)                        # от оси привода B_i до шарнира платформы P_i
        E = -2 * L * c.dot(uvec(i))
        F = 2 * L * c[2]
        G = c.dot(c) + L ** 2 - l ** 2
        D = E * E + F * F - G * G
        if D < 0:
            return None
        t = (-F + branch * np.sqrt(D)) / (G - E)
        th[i] = 2 * np.arctan(t)
    return th


def delta_knees(th, L=LU):
    return [WB * uvec(i) + L * (np.cos(th[i]) * uvec(i) - np.sin(th[i]) * KZ) for i in range(3)]


def trilaterate(c1, c2, c3, r1, r2, r3):
    """Пересечение трёх сфер. Возвращает две точки (или None). Не имеет особенности при
    равных высотах центров; вырождается, только если центры лежат на одной прямой."""
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
    return base + Z * ez, base - Z * ez


def delta_fk(th, L=LU, l=LL, a=None):
    """ПЗК дельта-робота 3-RUU: пересечение трёх сфер радиуса l с центрами в коленях,
    сдвинутых на -P_i (виртуальные центры). Выбирается решение ниже базы."""
    if a is None:
        a = WB - UP
    cs = [(a + L * np.cos(th[i])) * uvec(i) - L * np.sin(th[i]) * KZ for i in range(3)]
    sol = trilaterate(cs[0], cs[1], cs[2], l, l, l)
    if sol is None:
        return None
    return sol[0] if sol[0][2] < sol[1][2] else sol[1]


def delta_jac(p, th, L=LU, a=None):
    """Матрицы уравнения скоростей A*pdot = B*thdot и якобиан J = A^-1 B (pdot = J thdot)."""
    if a is None:
        a = WB - UP
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    for i in range(3):
        u = uvec(i)
        s = np.asarray(p, float) - a * u - L * (np.cos(th[i]) * u - np.sin(th[i]) * KZ)   # вектор штанги
        A[i] = s
        B[i, i] = L * s.dot(-np.sin(th[i]) * u - np.cos(th[i]) * KZ)
    return A, B, np.linalg.solve(A, B)


# ---- линейный дельта-робот 3-PUU: параметры DeltaMaker (по Р. Уильямсу), метры
LIN_SB, LIN_SP, LIN_L = 0.246, 0.127, 0.264
LIN_RB = np.sqrt(3) / 3 * LIN_SB                   # радиус расположения направляющих
LIN_UP = np.sqrt(3) / 3 * LIN_SP
LIN_PHI = np.radians([210.0, 330.0, 90.0])         # азимуты направляющих (как у Уильямса)


def lin_u(i):
    return np.array([np.cos(LIN_PHI[i]), np.sin(LIN_PHI[i]), 0.0])


def lindelta_ik(p, l=LIN_L, a=None):
    """ОЗК линейного дельта-робота: d_i = -z - sqrt(l^2 - rho_i^2) (каретка выше платформы);
    d_i — опускание каретки от плоскости базы (положительно вниз)."""
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
    if a is None:
        a = LIN_RB - LIN_UP
    cs = [a * lin_u(i) - d[i] * KZ for i in range(3)]
    sol = trilaterate(cs[0], cs[1], cs[2], l, l, l)
    if sol is None:
        return None
    return sol[0] if sol[0][2] < sol[1][2] else sol[1]


# ---- плоский пятизвенник 5R
FB_D, FB_L1, FB_L2 = 0.30, 0.30, 0.45              # расстояние между приводами, ведущие и ведомые звенья


def fivebar_fk(q1, q2, mode=+1, d=FB_D, l1=FB_L1, l2=FB_L2):
    """ПЗК пятизвенника: пересечение окружностей радиуса l2 с центрами B1, B2.
    mode=+1 — верхняя сборка (точка слева от вектора B1B2), -1 — нижняя."""
    B1 = np.array([-d / 2 + l1 * np.cos(q1), l1 * np.sin(q1)])
    B2 = np.array([d / 2 + l1 * np.cos(q2), l1 * np.sin(q2)])
    e = B2 - B1; D = np.linalg.norm(e)
    h2 = l2 ** 2 - D ** 2 / 4
    if h2 < 0:
        return None, B1, B2
    n = np.array([-e[1], e[0]]) / D
    return (B1 + B2) / 2 + mode * np.sqrt(h2) * n, B1, B2


def fivebar_ik(P, s1=+1, s2=-1, d=FB_D, l1=FB_L1, l2=FB_L2):
    """ОЗК пятизвенника: каждая нога — двухзвенник; s1, s2 = +-1 — выбор излома."""
    out = []
    for xa, s in ((-d / 2, s1), (d / 2, s2)):
        v = np.array([P[0] - xa, P[1]]); rho = np.linalg.norm(v)
        c = (rho ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * rho)
        if abs(c) > 1:
            return None
        out.append(np.arctan2(v[1], v[0]) + s * np.arccos(c))
    return np.array(out)


# ---- параллелограммный манипулятор (типа палетайзера), метры
PG_H, PG_L2, PG_L3, PG_D, PG_E = 0.80, 1.00, 1.20, 0.20, 0.25
PG_GAMMA = np.radians([25.0, 155.0])               # допустимый угол q2 - q3 между рычагом и предплечьем


def pgram_fk(q, h=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E):
    """q1 — поворот колонны; q2 — угол плеча к горизонту; q3 — угол предплечья к горизонту
    (оба привода на оси плеча, q3 передаётся параллелограммом). Фланец горизонтален."""
    q1, q2, q3 = q
    r = l2 * np.cos(q2) + l3 * np.cos(q3) + d
    z = h + l2 * np.sin(q2) + l3 * np.sin(q3) - e
    return np.array([r * np.cos(q1), r * np.sin(q1), z])


def pgram_ik(p, h=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E):
    x, y, z = p
    q1 = np.arctan2(y, x)
    rr = np.hypot(x, y) - d; zz = z - h + e
    c = (rr ** 2 + zz ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    if abs(c) > 1:
        return None
    v = -np.arccos(c)                               # локоть вверх: относительный угол отрицателен
    q2 = np.arctan2(zz, rr) - np.arctan2(l3 * np.sin(v), l2 + l3 * np.cos(v))
    return np.array([q1, q2, q2 + v])


# ---- платформа Гью — Стюарта 6-UPS
ST_RA, ST_RB, ST_DA, ST_DB = 0.60, 0.35, 12.0, 12.0


def stewart_geom():
    ang_a = np.radians([0 - ST_DA, 0 + ST_DA, 120 - ST_DA, 120 + ST_DA, 240 - ST_DA, 240 + ST_DA])
    ang_b = np.radians([-60 + ST_DB, 60 - ST_DB, 60 + ST_DB, 180 - ST_DB, 180 + ST_DB, 300 - ST_DB])
    a = np.array([[ST_RA * np.cos(t), ST_RA * np.sin(t), 0] for t in ang_a])
    b = np.array([[ST_RB * np.cos(t), ST_RB * np.sin(t), 0] for t in ang_b])
    return a, b


def rpy(al, be, ga):
    """R = Rz(ga) Ry(be) Rx(al)."""
    ca, sa, cb, sb, cg, sg = np.cos(al), np.sin(al), np.cos(be), np.sin(be), np.cos(ga), np.sin(ga)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz.dot(Ry).dot(Rx)


def stewart_ik(x):
    """x = (px,py,pz, al,be,ga). Длины ног l_i = |p + R b_i - a_i|."""
    a, b = stewart_geom()
    R = rpy(*x[3:])
    return np.linalg.norm(x[:3] + b.dot(R.T) - a, axis=1)


def stewart_fk(lens, x0, tol=1e-12, itmax=30):
    """Численная ПЗК методом Ньютона с конечно-разностной матрицей Якоби."""
    x = np.array(x0, float)
    for k in range(itmax):
        f = stewart_ik(x) - lens
        if np.abs(f).max() < tol:
            return x, k
        Jm = np.zeros((6, 6))
        for j in range(6):
            dx = np.zeros(6); dx[j] = 1e-7
            Jm[:, j] = (stewart_ik(x + dx) - stewart_ik(x - dx)) / 2e-7
        x = x - np.linalg.solve(Jm, f)
    return x, itmax


def check():
    deg = np.degrees
    print('--- delta 3-RUU: sB=%.3f sP=%.3f L=%.3f l=%.3f wB=%.5f uB=%.5f wP=%.5f uP=%.5f a=wB-uP=%.5f' % (SB, SP, LU, LL, WB, UB, WP, UP, WB - UP))
    for p in ([0, 0, -0.9], [0.3, 0.5, -1.1]):
        th = delta_ik(p); th_in = delta_ik(p, branch=+1)
        pb = delta_fk(th)
        print('IK', p, '-> theta(deg) =', np.round(deg(th), 3), ' inner branch:', np.round(deg(th_in), 2), ' FK(IK) err = %.2e' % np.linalg.norm(pb - p))
    for thd in ([0, 0, 0], [10, 20, 30]):
        p = delta_fk(np.radians(thd)); th = delta_ik(p)
        print('FK', thd, '-> p =', np.round(p, 4), ' IK(FK) err(deg) = %.2e' % np.abs(deg(th) - thd).max())
    rng = np.random.RandomState(1); worst = 0
    for k in range(2000):
        p = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-1.3, -0.8)])
        th = delta_ik(p)
        if th is None:
            continue
        worst = max(worst, np.linalg.norm(delta_fk(th) - p))
    print('random 2000 points: max |FK(IK(p)) - p| = %.2e' % worst)
    p = np.array([0.3, 0.5, -1.1]); th = delta_ik(p)
    A, B, J = delta_jac(p, th)
    Jn = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3); d[j] = 1e-6
        Jn[:, j] = (delta_fk(th + d) - delta_fk(th - d)) / 2e-6
    print('J at (0.3,0.5,-1.1):'); print(np.round(J, 4)); print('max |J - Jnum| = %.2e' % np.abs(J - Jn).max())
    print('det A = %.5f, diag B =' % np.linalg.det(A), np.round(np.diag(B), 5), ' cond J = %.3f' % np.linalg.cond(J))
    p0 = np.array([0, 0, -0.9]); A, B, J = delta_jac(p0, delta_ik(p0))
    print('J at (0,0,-0.9):'); print(np.round(J, 4)); print('cond = %.3f' % np.linalg.cond(J), ' sv =', np.round(np.linalg.svd(J)[1], 4))
    print('--- linear delta: RB=%.5f uP=%.5f a=%.5f l=%.3f' % (LIN_RB, LIN_UP, LIN_RB - LIN_UP, LIN_L))
    for p in ([0, 0, -0.5], [0.03, 0.05, -0.40]):
        d = lindelta_ik(p); print('IK', p, '-> d =', np.round(d, 4), ' FK err = %.2e' % np.linalg.norm(lindelta_fk(d) - p))
    for d in ([0.2, 0.2, 0.2], [0.14, 0.15, 0.16]):
        print('FK', d, '-> p =', np.round(lindelta_fk(np.array(d)), 4))
    print('--- fivebar d=%.2f l1=%.2f l2=%.2f' % (FB_D, FB_L1, FB_L2))
    for q in ((120, 60), (100, 30)):
        for m in (+1, -1):
            P, B1, B2 = fivebar_fk(np.radians(q[0]), np.radians(q[1]), m)
            print('q =', q, 'mode', m, 'P =', np.round(P, 4), ' B1 =', np.round(B1, 4), ' B2 =', np.round(B2, 4))
    P, _, _ = fivebar_fk(np.radians(100), np.radians(30), +1)
    print('IK back (deg):', np.round(deg(fivebar_ik(P)), 4))
    print('--- parallelogram arm h=%.2f l2=%.2f l3=%.2f d=%.2f e=%.2f' % (PG_H, PG_L2, PG_L3, PG_D, PG_E))
    q = np.radians([30, 60, -20]); p = pgram_fk(q)
    print('FK q=(30,60,-20) -> p =', np.round(p, 4), ' IK back (deg) =', np.round(deg(pgram_ik(p)), 4))
    print('  r = %.4f, elbow relative angle = %.1f deg, det Jrz = l2 l3 sin(q3-q2) = %.4f' % (np.hypot(p[0], p[1]), -80.0, PG_L2 * PG_L3 * np.sin(q[2] - q[1])))
    qq = pgram_ik([1.5, 0.8, 0.9]); print('IK (1.5,0.8,0.9) -> q(deg) =', np.round(deg(qq), 3), ' FK back =', np.round(pgram_fk(qq), 6))
    print('--- Stewart rA=%.2f rB=%.2f half-angles %.0f/%.0f deg' % (ST_RA, ST_RB, ST_DA, ST_DB))
    x0 = np.array([0, 0, 0.70, 0, 0, 0]); print('home lengths:', np.round(stewart_ik(x0), 4))
    x = np.array([0.05, -0.03, 0.75, np.radians(5), np.radians(-8), np.radians(10)])
    ln = stewart_ik(x); print('pose', np.round(x, 4), '-> lengths', np.round(ln, 4))
    xs, it = stewart_fk(ln, x0); print('Newton from home: %d iterations, err = %.2e' % (it, np.abs(xs - x).max()))

# ====================================================================== дополнения для ноутбука
NAVY, AMB, TEAL, RED, GREY, ICE, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#CADCFC', '#2C7A5A'


def mobility(N, j, fsum, lam=6):
    """Структурная формула Чебышёва — Грюблера — Кутцбаха: W = lam (N - 1 - j) + sum f_i.
    N — число звеньев со стойкой, j — число сочленений, fsum — сумма их подвижностей."""
    return lam * (N - 1 - j) + fsum


# ---------------------------------------------------------------- пятизвенник
def fivebar_jac(P, q, d=FB_D, l1=FB_L1, l2=FB_L2):
    """Матрицы уравнения скоростей пятизвенника A*Pdot = B*qdot и J = A^-1 B."""
    B1 = np.array([-d / 2 + l1 * np.cos(q[0]), l1 * np.sin(q[0])])
    B2 = np.array([d / 2 + l1 * np.cos(q[1]), l1 * np.sin(q[1])])
    A = np.array([P - B1, P - B2])
    e = [np.array([-np.sin(q[k]), np.cos(q[k])]) for k in range(2)]
    B = np.diag([l1 * (P - B1).dot(e[0]), l1 * (P - B2).dot(e[1])])
    J = np.linalg.solve(A, B) if abs(np.linalg.det(A)) > 1e-12 else None
    return A, B, J


def fivebar_workspace(n=140, mode=+1, s1=+1, s2=-1, xlim=(-0.8, 0.8), ylim=(-0.1, 0.8)):
    """Карта индекса обусловленности 1/kappa(J) пятизвенника по рабочей зоне (через ОЗК)."""
    xs = np.linspace(xlim[0], xlim[1], n); ys = np.linspace(ylim[0], ylim[1], n)
    M = np.full((n, n), np.nan)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            P = np.array([x, y]); q = fivebar_ik(P, s1, s2)
            if q is None:
                continue
            Pm, _, _ = fivebar_fk(q[0], q[1], mode)
            if Pm is None or np.linalg.norm(Pm - P) > 1e-7:      # точка принадлежит другому режиму сборки
                continue
            A, B, J = fivebar_jac(P, q)
            if J is not None:
                M[iy, ix] = 1.0 / np.linalg.cond(J)
    return xs, ys, M


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


# ---------------------------------------------------------------- параллелограммный манипулятор
def pgram_jac_rz(q, l2=PG_L2, l3=PG_L3):
    """Матрица Якоби в плоскости руки: (rdot, zdot) = J (q2dot, q3dot)."""
    return np.array([[-l2 * np.sin(q[1]), -l3 * np.sin(q[2])], [l2 * np.cos(q[1]), l3 * np.cos(q[2])]])


def pgram_limits_ok(q, lim2=(20.0, 130.0), lim3=(-75.0, 35.0)):
    g = q[1] - q[2]
    return (np.radians(lim2[0]) <= q[1] <= np.radians(lim2[1]) and np.radians(lim3[0]) <= q[2] <= np.radians(lim3[1])
            and PG_GAMMA[0] <= g <= PG_GAMMA[1])


def pgram_workspace(n=220, rlim=(-0.2, 2.6), zlim=(-0.55, 2.5)):
    """Сечение рабочей зоны в плоскости (r, z); значение — |sin(q2 - q3)| (пропорционально det J)."""
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


def pgram_points(q, c=0.42, h=PG_H, l2=PG_L2, l3=PG_L3, d=PG_D, e=PG_E):
    """Характерные точки в плоскости руки (r, z): ось плеча O2, локоть E, конец предплечья W,
    фланец F, рабочая точка TCP, конец кривошипа C и конец хвостовика предплечья Rr."""
    e2 = np.array([np.cos(q[1]), np.sin(q[1])]); e3 = np.array([np.cos(q[2]), np.sin(q[2])])
    O2 = np.array([0.0, h]); E = O2 + l2 * e2; W = E + l3 * e3
    F = W + np.array([d, 0.0]); TCP = F - np.array([0.0, e])
    return dict(O2=O2, E=E, W=W, F=F, TCP=TCP, C=O2 - c * e3, Rr=E - c * e3)


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


# ---------------------------------------------------------------- дельта-робот
TH_LIM = np.radians([-45.0, 95.0]); BETA_MAX = np.radians(40.0)


def delta_ik_table(p, L=LU, l=LL, a=None):
    """Промежуточные величины ОЗК по ногам: xi, eta, E, F, G, t (рабочая ветвь), theta (рад)."""
    if a is None:
        a = WB - UP
    p = np.asarray(p, float); rows = []
    for i in range(3):
        u = uvec(i); tv = np.array([-u[1], u[0], 0.0]); c = p - a * u
        xi, eta = c.dot(u), c.dot(tv)
        E = -2 * L * xi; F = 2 * L * p[2]; G = xi ** 2 + eta ** 2 + p[2] ** 2 + L ** 2 - l ** 2
        D = E * E + F * F - G * G
        t = (-F - np.sqrt(D)) / (G - E) if D >= 0 else np.nan
        rows.append((xi, eta, E, F, G, t, 2 * np.arctan(t)))
    return np.array(rows)


def delta_quality(p):
    """Индекс обусловленности 1/kappa(J) в точке p; nan — точка вне рабочей зоны
    (ОЗК не решается, угол привода вне [-45; 95] градусов или штанга отклонена более чем на 40 градусов)."""
    p = np.asarray(p, float)
    th = delta_ik(p)
    if th is None or th.min() < TH_LIM[0] or th.max() > TH_LIM[1]:
        return np.nan
    a = WB - UP
    for i in range(3):
        u = uvec(i); eta = (p - a * u).dot([-u[1], u[0], 0])
        if abs(eta) > LL * np.sin(BETA_MAX):
            return np.nan
    A, B, J = delta_jac(p, th)
    return 1.0 / np.linalg.cond(J)


def delta_section(plane='xz', value=0.0, n=90, lim=(-1.0, 1.0), zlim=(-1.85, -0.25)):
    """Сечение рабочей зоны дельта-робота: plane='xz' (y = value) или 'xy' (z = value).
    Возвращает сетки осей и карту 1/kappa(J)."""
    us = np.linspace(lim[0], lim[1], n)
    if plane == 'xz':
        vs = np.linspace(zlim[0], zlim[1], n)
        M = np.array([[delta_quality([x, value, z]) for x in us] for z in vs])
    else:
        vs = np.linspace(lim[0], lim[1], n)
        M = np.array([[delta_quality([x, y, value]) for x in us] for y in vs])
    return us, vs, M


def delta_cloud(n=4000, seed=0):
    """Облако точек рабочей зоны через ПЗК: случайные углы приводов в допустимом диапазоне."""
    rng = np.random.RandomState(seed); pts = []
    for k in range(n):
        th = rng.uniform(TH_LIM[0], TH_LIM[1], 3)
        p = delta_fk(th)
        if p is not None and not np.isnan(delta_quality(p)):
            pts.append(p)
    return np.array(pts)


def _tri(ax, c, r, phis, z, color, lw=2, alpha=1.0):
    v = np.array([[c[0] + r * np.cos(f), c[1] + r * np.sin(f), z] for f in list(phis) + [phis[0]]])
    ax.plot(v[:, 0], v[:, 1], v[:, 2], color=color, lw=lw, alpha=alpha)


def draw_delta(ax, p, th=None, color=NAVY, alpha=1.0, par=0.06):
    """Трёхмерный скелет дельта-робота 3-RUU: основание, рычаги, штанги (парами), платформа."""
    p = np.asarray(p, float)
    if th is None:
        th = delta_ik(p)
    _tri(ax, (0, 0), UB, PHI + np.pi, 0.0, GREY, lw=2.5)               # вершины базы — напротив ног
    _tri(ax, p[:2], UP, PHI, p[2], AMB, lw=2.5, alpha=alpha)
    knees = delta_knees(th)
    for i in range(3):
        u = uvec(i); tv = np.array([-u[1], u[0], 0.0]); B = WB * u; K = knees[i]; Pi = p + UP * u
        ax.plot([B[0], K[0]], [B[1], K[1]], [B[2], K[2]], color=color, lw=4, alpha=alpha, solid_capstyle='round')
        for s in (-1, 1):
            ax.plot([K[0] + s * par * tv[0], Pi[0] + s * par * tv[0]], [K[1] + s * par * tv[1], Pi[1] + s * par * tv[1]],
                    [K[2], Pi[2]], color=TEAL, lw=1.6, alpha=alpha)
        ax.scatter([B[0]], [B[1]], [B[2]], color=AMB, s=50, edgecolors=NAVY, depthshade=False)
    ax.scatter([p[0]], [p[1]], [p[2]], color=RED, s=30, depthshade=False)
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
    _tri(ax, p[:2], LIN_UP, LIN_PHI, p[2], AMB, lw=2.5)
    ax.scatter([p[0]], [p[1]], [p[2]], color=RED, s=30, depthshade=False)
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')


# ---------------------------------------------------------------- платформа Гью — Стюарта
def stewart_fk_hist(lens, x0, tol=1e-12, itmax=40):
    """Метод Ньютона для ПЗК платформы с историей невязки. Возвращает (x, список max|l(x) - l*|)."""
    x = np.array(x0, float); hist = []
    for k in range(itmax):
        f = stewart_ik(x) - lens
        hist.append(np.abs(f).max())
        if hist[-1] < tol or not np.isfinite(hist[-1]):
            break
        Jm = np.zeros((6, 6))
        for j in range(6):
            dx = np.zeros(6); dx[j] = 1e-7
            Jm[:, j] = (stewart_ik(x + dx) - stewart_ik(x - dx)) / 2e-7
        try:
            x = x - np.linalg.solve(Jm, f)
        except np.linalg.LinAlgError:
            hist.append(np.inf); break
    return x, hist


def stewart_invjac(x):
    """Обратная матрица Якоби платформы: ldot = Jinv * (omega; v); строка i = ((R b_i x n_i)^T, n_i^T)."""
    a, b = stewart_geom(); R = rpy(*x[3:])
    Jinv = np.zeros((6, 6))
    for i in range(6):
        rb = R.dot(b[i]); s = np.asarray(x[:3], float) + rb - a[i]; n = s / np.linalg.norm(s)
        Jinv[i, :3] = np.cross(rb, n); Jinv[i, 3:] = n
    return Jinv


def draw_stewart(ax, x, color=NAVY, alpha=1.0, legs=True):
    """Скелет платформы 6-UPS в позе x = (px, py, pz, alpha, beta, gamma)."""
    a, b = stewart_geom(); R = rpy(*x[3:]); top = np.asarray(x[:3], float) + b.dot(R.T)
    aa = np.vstack([a, a[:1]]); tt = np.vstack([top, top[:1]])
    ax.plot(aa[:, 0], aa[:, 1], aa[:, 2], color=GREY, lw=2.5)
    ax.plot(tt[:, 0], tt[:, 1], tt[:, 2], color=AMB, lw=2.5, alpha=alpha)
    if legs:
        for i in range(6):
            ax.plot([a[i, 0], top[i, 0]], [a[i, 1], top[i, 1]], [a[i, 2], top[i, 2]], color=color, lw=2.2, alpha=alpha)
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], color=GREY, s=25, depthshade=False)
    ax.scatter(top[:, 0], top[:, 1], top[:, 2], color=AMB, s=25, depthshade=False, alpha=alpha)
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')
    return top


if __name__ == '__main__':
    check()
