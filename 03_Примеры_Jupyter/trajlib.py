# -*- coding: utf-8 -*-
"""trajlib — вспомогательные функции занятия 4 «Планирование траекторий и кинематическое управление».

Учебник: Зенкевич С. Л., Ющенко А. С. Основы управления манипуляционными роботами, глава 4.
Номера формул в комментариях — номера учебника; исправления опечаток оговорены в конспекте лекции 4.

Состав:
  профили разгона — торможения ......... lam_rect, lam_tri, lam_trap, lam_sin2, lam_poly3, lam_poly5
  трапеция по ограничениям привода ..... trapezoid, trap_eval, sync_trapezoid
  многочлены ........................... poly5_time, cubic_coeffs, cubic_eval
  кубический сплайн .................... sweep_book, thomas, spline_m, spline_eval, spline_extrema
  плоский двухзвенник .................. fk2, ik2, J2, arm_points, ik2_newton
  декартова интерполяция ............... line_deviation, joint_line, cart_line, split_error, chord_error
  программные траектории схвата ........ circle_ref, line_ref
  управление по вектору скорости ....... dls_pinv, scale_velocities, clip_velocities, track, pass_near
Только numpy.
"""
import numpy as np

NAVY, AMB, TEAL, RED, GREY = '#1F3A68', '#E8A33D', '#2A9D8F', '#C0392B', '#7A8599'


# ====================================================================== профили разгона — торможения
# Каждая функция принимает безразмерное время tau = t/T из [0, 1] и возвращает (lam, dlam, ddlam)
# по безразмерному времени; размерные производные: dlam/T и ddlam/T**2.
def lam_rect(tau):
    tau = np.asarray(tau, dtype=float)
    return tau, np.ones_like(tau), np.zeros_like(tau)


def lam_tri(tau):
    tau = np.asarray(tau, dtype=float)
    up = tau < 0.5
    lam = np.where(up, 2 * tau ** 2, 1 - 2 * (1 - tau) ** 2)
    dlam = np.where(up, 4 * tau, 4 * (1 - tau))
    ddlam = np.where(up, 4.0, -4.0)
    return lam, dlam, ddlam


def lam_trap(tau, a=1.0 / 3.0):
    """Трапеция с долей разгона a = t_a/T; высота профиля 1/(1 - a) (при a = 1/3 это 3/2)."""
    tau = np.asarray(tau, dtype=float)
    h = 1.0 / (1.0 - a)
    lam = np.where(tau < a, h * tau ** 2 / (2 * a),
                   np.where(tau <= 1 - a, h * (tau - a / 2), 1 - h * (1 - tau) ** 2 / (2 * a)))
    dlam = np.where(tau < a, h * tau / a, np.where(tau <= 1 - a, h, h * (1 - tau) / a))
    ddlam = np.where(tau < a, h / a, np.where(tau <= 1 - a, 0.0, -h / a))
    return lam, dlam, ddlam


def lam_sin2(tau):
    tau = np.asarray(tau, dtype=float)
    return (tau - np.sin(2 * np.pi * tau) / (2 * np.pi), 2 * np.sin(np.pi * tau) ** 2,
            2 * np.pi * np.sin(2 * np.pi * tau))


def lam_poly3(tau):
    tau = np.asarray(tau, dtype=float)
    return 3 * tau ** 2 - 2 * tau ** 3, 6 * tau - 6 * tau ** 2, 6 - 12 * tau


def lam_poly5(tau):
    tau = np.asarray(tau, dtype=float)
    return (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5, 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4,
            60 * tau - 180 * tau ** 2 + 120 * tau ** 3)


# ====================================================================== трапеция по ограничениям привода
def trapezoid(dq, vmax, amax):
    """Времена трапецеидального профиля для одного сочленения.
    Возвращает словарь: ta, T, vpeak, kind ('trapezoid' | 'triangle')."""
    dq = abs(float(dq))
    if dq >= vmax ** 2 / amax:
        ta = vmax / amax
        return dict(ta=ta, T=dq / vmax + ta, vpeak=vmax, acc=amax, kind='trapezoid')
    ta = np.sqrt(dq / amax)
    return dict(ta=ta, T=2 * ta, vpeak=np.sqrt(dq * amax), acc=amax, kind='triangle')


def trap_eval(t, q0, q1, ta, T):
    """Закон движения q(t), скорость и ускорение для трапеции с общими ta и T (скорость полки dq/(T - ta))."""
    t = np.clip(np.asarray(t, dtype=float), 0, T)
    lam, dlam, ddlam = lam_trap(t / T, ta / T)
    d = q1 - q0
    return q0 + d * lam, d * dlam / T, d * ddlam / T ** 2


def sync_trapezoid(dq, vmax, amax):
    """Синхронизация сочленений: общие ta = max ta_i и T = max(dq_i/vmax_i) + ta.
    Возвращает ta, T, скорости полки и ускорения каждого сочленения."""
    dq = np.abs(np.asarray(dq, dtype=float)); vmax = np.asarray(vmax, dtype=float); amax = np.asarray(amax, dtype=float)
    ta = np.max(vmax / amax)
    T = np.max(dq / vmax) + ta
    if T < 2 * ta:                      # у ведущего сочленения нет полки — общий треугольник
        T = 2 * np.max(np.sqrt(dq / amax)); ta = T / 2
    v = dq / (T - ta)
    return ta, T, v, v / ta


# ====================================================================== многочлены
def poly5_time(dq, vmax, amax):
    """Минимальное время для многочлена 5-й степени: пики 15/8 * D/T и 10/sqrt(3) * D/T^2."""
    dq = abs(float(dq))
    Tv = 15.0 / 8.0 * dq / vmax
    Ta = np.sqrt(10.0 / np.sqrt(3.0) * dq / amax)
    return max(Tv, Ta), Tv, Ta


def cubic_coeffs(q0, q1, v0, v1, T):
    """Кубический многочлен с заданными скоростями на концах (многочлен Эрмита)."""
    D = q1 - q0
    return np.array([q0, v0, (3 * D - (2 * v0 + v1) * T) / T ** 2, (-2 * D + (v0 + v1) * T) / T ** 3])


def cubic_eval(a, t):
    t = np.asarray(t, dtype=float)
    return (a[0] + a[1] * t + a[2] * t ** 2 + a[3] * t ** 3, a[1] + 2 * a[2] * t + 3 * a[3] * t ** 2,
            2 * a[2] + 6 * a[3] * t)


# ====================================================================== кубический сплайн
def spline_system(t, y):
    """Коэффициенты alpha_k (4.22) и правые части M_k (4.23, с исправленными индексами) для k = 1..n-1."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    h = np.diff(t)
    al = h[1:] / (h[1:] + h[:-1])
    M = 6.0 / (h[1:] + h[:-1]) * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    return al, M


def sweep_book(t, y, m0=0.0, mn=0.0, verbose=False):
    """Метод прогонки в форме учебника (4.27)–(4.32): x_k = c_k x_{k+1} + d_k, заданы q''_0 и q''_n.
    Возвращает вторые производные в узлах и коэффициенты прогонки."""
    al, M = spline_system(t, y)
    n = len(t) - 1
    c = np.zeros(n); d = np.zeros(n)
    c[0], d[0] = 0.0, m0
    for k in range(1, n):
        den = 2 + (1 - al[k - 1]) * c[k - 1]
        c[k] = -al[k - 1] / den
        d[k] = (M[k - 1] - (1 - al[k - 1]) * d[k - 1]) / den
        if verbose:
            print('k = %d: alpha = %.4f, M = %.4f, znamenatel = %.4f, c = %.4f, d = %.4f' % (k, al[k - 1], M[k - 1], den, c[k], d[k]))
    x = np.zeros(n + 1); x[n] = mn
    for k in range(n - 1, -1, -1):
        x[k] = c[k] * x[k + 1] + d[k]
    return x, c, d


def thomas(lo, di, up, rhs):
    """Прогонка для произвольной трёхдиагональной системы: lo[k] x[k-1] + di[k] x[k] + up[k] x[k+1] = rhs[k]."""
    n = len(di)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = up[0] / di[0]; dp[0] = rhs[0] / di[0]
    for k in range(1, n):
        den = di[k] - lo[k] * cp[k - 1]
        cp[k] = up[k] / den if k < n - 1 else 0.0
        dp[k] = (rhs[k] - lo[k] * dp[k - 1]) / den
    x = np.zeros(n); x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def spline_m(t, y, bc='clamped', left=0.0, right=0.0):
    """Вторые производные кубического сплайна в узлах.
    bc='natural' — заданы ускорения на концах (left, right); bc='clamped' — заданы скорости на концах."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    if bc == 'natural':
        return sweep_book(t, y, left, right)[0]
    n = len(t) - 1
    h = np.diff(t)
    al, M = spline_system(t, y)
    lo = np.zeros(n + 1); di = 2 * np.ones(n + 1); up = np.zeros(n + 1); rhs = np.zeros(n + 1)
    up[0] = 1.0; rhs[0] = 6 / h[0] * ((y[1] - y[0]) / h[0] - left)
    lo[n] = 1.0; rhs[n] = 6 / h[-1] * (right - (y[n] - y[n - 1]) / h[-1])
    lo[1:n] = 1 - al; up[1:n] = al; rhs[1:n] = M
    return thomas(lo, di, up, rhs)


def spline_eval(t, y, m, x):
    """Значение, скорость и ускорение сплайна в точках x (формула, следующая из (4.16) и (4.18))."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    h = np.diff(t)
    k = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
    a = t[k + 1] - x; b = x - t[k]; hk = h[k]
    A = y[k] / hk - m[k] * hk / 6; B = y[k + 1] / hk - m[k + 1] * hk / 6
    s = (m[k] * a ** 3 + m[k + 1] * b ** 3) / (6 * hk) + A * a + B * b
    ds = (-m[k] * a ** 2 + m[k + 1] * b ** 2) / (2 * hk) - A + B
    dds = (m[k] * a + m[k + 1] * b) / hk
    return s, ds, dds


def spline_extrema(t, y, m):
    """Наибольшие |скорость| и |ускорение| сплайна: ускорение — в узлах, скорость — в узлах и в нулях q''."""
    t = np.asarray(t, dtype=float)
    xs = list(t)
    for k in range(len(t) - 1):
        if m[k] * m[k + 1] < 0:
            xs.append(t[k] + (t[k + 1] - t[k]) * m[k] / (m[k] - m[k + 1]))
    _, ds, _ = spline_eval(t, y, m, np.array(xs))
    return np.max(np.abs(ds)), np.max(np.abs(m))


# ====================================================================== плоский двухзвенник
def fk2(q, l1=1.0, l2=1.0):
    return np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]), l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])])


def ik2(p, l1=1.0, l2=1.0, elbow=1):
    c2 = np.clip((p[0] ** 2 + p[1] ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1, 1)
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


def J2(q, l1=1.0, l2=1.0):
    return np.array([[-l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1]), -l2 * np.sin(q[0] + q[1])],
                     [l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]), l2 * np.cos(q[0] + q[1])]])


def arm_points(q, l1=1.0, l2=1.0):
    p1 = np.array([l1 * np.cos(q[0]), l1 * np.sin(q[0])])
    return np.array([[0.0, 0.0], p1, fk2(q, l1, l2)])


def draw_arm(ax, q, color=NAVY, alpha=1.0, lw=3.0):
    P = arm_points(q)
    ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    ax.plot(P[1, 0], P[1, 1], 'o', color='white', mec=color, mew=1.3, ms=5.5, alpha=alpha)


def ik2_newton(p, q0, tol=1e-9, max_iter=50):
    """Итерационная ОЗК: q <- q + J^+(q)(p - f(q)) из начального приближения q0. Возвращает q и число итераций."""
    q = np.array(q0, dtype=float)
    for it in range(max_iter):
        e = np.asarray(p) - fk2(q)
        if np.linalg.norm(e) < tol:
            return q, it
        q = q + np.linalg.pinv(J2(q)) @ e
    return q, max_iter


# ====================================================================== интерполяция в Q и в декартовом пространстве
def line_deviation(P, A, B):
    """Расстояния от точек P [n,2] до прямой AB."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float); P = np.asarray(P, dtype=float)
    d = B - A
    return np.abs(d[0] * (P[:, 1] - A[1]) - d[1] * (P[:, 0] - A[0])) / np.linalg.norm(d)


def joint_line(A, B, n=201):
    """Прямая в пространстве сочленений между ОЗК(A) и ОЗК(B): углы и путь схвата."""
    qA, qB = ik2(A), ik2(B)
    lam = np.linspace(0, 1, n)
    Q = qA[None, :] + lam[:, None] * (qB - qA)[None, :]
    return Q, np.array([fk2(q) for q in Q])


def cart_line(A, B, n=201):
    """Прямая в декартовом пространстве: путь схвата и соответствующие углы (ОЗК в каждой точке)."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    lam = np.linspace(0, 1, n)
    P = A[None, :] + lam[:, None] * (B - A)[None, :]
    return np.array([ik2(p) for p in P]), P


def split_error(A, B, nseg, nfine=4001):
    """Наибольшее отклонение схвата от AB, если на AB взять nseg+1 узел, а между узлами углы менять линейно."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    lk = np.linspace(0, 1, nseg + 1)
    qk = np.array([ik2(A + l * (B - A)) for l in lk])
    ll = np.linspace(0, 1, nfine)
    Q = np.vstack([np.interp(ll, lk, qk[:, 0]), np.interp(ll, lk, qk[:, 1])]).T
    return line_deviation(np.array([fk2(q) for q in Q]), A, B).max()


def chord_error(R, n):
    """Ошибка хорды для окружности радиуса R, разбитой на n равных дуг."""
    return R * (1 - np.cos(np.pi / n))


# ====================================================================== программные траектории схвата
def circle_ref(t, T=4.0, c=(1.0, 0.8), R=0.4, lam=lam_poly5):
    """Окружность примера 4.4: s*(t) и её скорость; phi = 2 pi lam(t/T)."""
    tau = np.clip(t / T, 0, 1)
    l, dl, _ = lam(tau)
    ph = 2 * np.pi * l
    s = np.array([c[0] + R * np.cos(ph), c[1] + R * np.sin(ph)])
    ds = np.array([-R * np.sin(ph), R * np.cos(ph)]) * 2 * np.pi * dl / T
    return s, ds


def line_ref(sig, d=0.05, v=0.4, x0=-0.7):
    """Прямая y = d мимо основания; параметр sig — «программное время»."""
    return np.array([x0 + v * sig, d]), np.array([v, 0.0])


# ====================================================================== управление по вектору скорости
def dls_pinv(J, lam=0.0):
    """Демпфированная псевдообратная J^T (J J^T + lam^2 E)^-1; при lam = 0 — обычная псевдообратная."""
    if lam == 0:
        return np.linalg.pinv(J)
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(J.shape[0]))


def scale_velocities(qd, qmax):
    """Общее масштабирование: все компоненты делятся на k = max(1, max|qd_i|/qmax_i). Возвращает (qd/k, k)."""
    k = max(1.0, float(np.max(np.abs(qd) / qmax)))
    return qd / k, k


def clip_velocities(qd, qmax):
    """Срезание по компонентам (так делать НЕ надо: меняется направление скорости схвата)."""
    return np.clip(qd, -np.asarray(qmax) * np.ones_like(qd), np.asarray(qmax) * np.ones_like(qd))


def track(K, dt=0.02, T=4.0, off=(0.0, 0.0), ref=circle_ref, lam=0.0, ff=True):
    """Слежение за программной траекторией: q' = J^+(s*' + K e), интегрирование по Эйлеру с тактом dt.
    off — начальное смещение схвата от программы; ff=False отключает прямую связь по скорости.
    Возвращает t, путь схвата P, норму ошибки E, скорости сочленений QD."""
    s0, _ = ref(0.0, T)
    q = ik2(s0 + np.array(off))
    ts = np.arange(0, T + 1e-9, dt); P = []; E = []; QD = []
    for t in ts:
        s, ds = ref(t, T)
        p = fk2(q); e = s - p
        qd = dls_pinv(J2(q), lam) @ ((ds if ff else 0.0) + K * e)
        P.append(p); E.append(np.linalg.norm(e)); QD.append(qd)
        q = q + dt * qd
    return ts, np.array(P), np.array(E), np.array(QD)


def pass_near(d, mode='inv', lam=0.15, v=0.4, dt=0.002, qmax=None, limit='scale', K=10.0, x0=-0.7, x1=0.7):
    """Проход прямой y = d рядом с особой точкой (сложенная рука над основанием).
    mode: 'inv' — обратная матрица Якоби, 'dls' — демпфированная псевдообратная с параметром lam.
    qmax/limit: None — без ограничения; 'clip' — срезание по компонентам; 'scale' — общее масштабирование;
                'time' — общее масштабирование и замедление программы в то же число раз.
    Возвращает t, скорости QD, норму ошибки E, путь схвата P, путь программы S."""
    Tn = (x1 - x0) / v
    q = ik2(np.array([x0, d]))
    sig = 0.0; t = 0.0
    ts = []; QD = []; E = []; P = []; S = []
    while sig <= Tn + 1e-9 and t < 5 * Tn:
        s, ds = line_ref(sig, d, v, x0)
        p = fk2(q); e = s - p
        J = J2(q)
        Ji = np.linalg.inv(J) if mode == 'inv' else dls_pinv(J, lam)
        qd = Ji @ (ds + K * e)
        k = 1.0
        if qmax is not None:
            if limit == 'clip':
                qd = clip_velocities(qd, qmax)
            elif limit == 'scale':
                qd, k = scale_velocities(qd, qmax); k = 1.0
            else:
                k = max(1.0, float(np.max(np.abs(Ji @ ds) / qmax)))
                qd, _ = scale_velocities(Ji @ (ds / k + K * e), qmax)
        ts.append(t); QD.append(qd); E.append(np.linalg.norm(e)); P.append(p); S.append(s)
        q = q + dt * qd
        sig += dt / k; t += dt
    return np.array(ts), np.array(QD), np.array(E), np.array(P), np.array(S)
