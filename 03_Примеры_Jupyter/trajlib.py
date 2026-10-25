# -*- coding: utf-8 -*-
"""trajlib — вспомогательные функции занятия 4 «Планирование траекторий и кинематическое управление».

Учебник: Зенкевич С. Л., Ющенко А. С. Основы управления манипуляционными роботами, глава 4.
Номера формул в комментариях — номера учебника; исправления опечаток оговорены в конспекте лекции 4.
Обозначения — по файлу 00_Паспорт/Обозначения.md: t̄ = t/T — безразмерное время, λ(t) — параметр
пути, μ — коэффициент демпфирования, κ — коэффициент растяжения времени, ρ — радиус окружности,
y₀ — расстояние прямой от основания, ε_ch — ошибка хорды, 𝒱* — командная скорость схвата.

Состав:
  профили разгона — торможения ......... lam_rect, lam_tri, lam_trap, lam_sin2, lam_poly3, lam_poly5, lam_poly7
  трапеция по ограничениям привода ..... trapezoid, trap_eval, sync_trapezoid
  ограничение рывка (S-кривая) ......... scurve_time, scurve_eval
  упругая нагрузка ..................... residual_vibration, residual_pc, flex_response, zv_shaper
  многочлены ........................... poly5_time, cubic_coeffs, cubic_eval
  кубический сплайн .................... sweep_book, thomas, spline_m ('natural' | 'clamped' | 'periodic'),
                                         spline_eval, spline_extrema
  плоский двухзвенник .................. fk2, ik2, J2, arm_points, draw_arm, ik2_newton, ik2_chain
  декартова интерполяция ............... line_deviation, joint_line, cart_line, split_error, chord_error
  программные траектории схвата ........ circle_ref, line_ref, circle_path
  оптимальная по времени параметризация  topp_path
  управление по вектору скорости ....... dls_pinv, scale_velocities, clip_velocities, track, regulate,
                                         pass_near, speed_limit_line
numpy; scipy нужен функциям topp_path (линейное программирование на шаге) и flex_response.
"""
import numpy as np

NAVY, AMB, TEAL, RED, GREY = '#1F3A68', '#E8A33D', '#2A9D8F', '#C0392B', '#7A8599'


# ====================================================================== профили разгона — торможения
# Каждая функция принимает безразмерное время t̄ = t/T из [0, 1] и возвращает (λ, λ', λ'')
# по безразмерному времени; размерные производные: λ'/T и λ''/T**2.
def lam_rect(tau):
    """Постоянная скорость λ = t̄. Внимание: возвращаемое ускорение — это ускорение ВНУТРИ
    интервала (нуль). На концах скорость скачком меняется от 0 до 1 и обратно, поэтому настоящее
    ускорение там — дельта-функция (бесконечно); численно её не изобразить."""
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


def lam_poly7(tau):
    """Многочлен седьмой степени 35t̄⁴ − 84t̄⁵ + 70t̄⁶ − 20t̄⁷: на концах нулевые скорость, ускорение и рывок.
    Пик скорости 35/16 при t̄ = 1/2."""
    tau = np.asarray(tau, dtype=float)
    return (35 * tau ** 4 - 84 * tau ** 5 + 70 * tau ** 6 - 20 * tau ** 7,
            140 * tau ** 3 - 420 * tau ** 4 + 420 * tau ** 5 - 140 * tau ** 6,
            420 * tau ** 2 - 1680 * tau ** 3 + 2100 * tau ** 4 - 840 * tau ** 5)


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
    """Закон движения q(t), скорость и ускорение для трапеции с общими ta и T (скорость полки dq/(T - ta)).
    До начала (t < 0) и после конца (t > T) движения — покой: q = q0 или q1, скорость и ускорение нулевые."""
    t = np.asarray(t, dtype=float)
    inside = (t >= 0) & (t <= T)
    lam, dlam, ddlam = lam_trap(np.clip(t, 0, T) / T, ta / T)
    d = q1 - q0
    return q0 + d * lam, np.where(inside, d * dlam / T, 0.0), np.where(inside, d * ddlam / T ** 2, 0.0)


def sync_trapezoid(dq, vmax, amax):
    """Синхронизация сочленений: общие ta = max ta_i (ta_i = vmax_i/amax_i) и
    T = max(max_i dq_i/vmax_i + ta, 2·ta).
    Если max_i dq_i/vmax_i < ta, то есть T < 2·ta, полагаем T = 2·ta: профиль становится
    треугольным, а ограничения выполняются: v_i = dq_i/ta < vmax_i и
    a_i = dq_i/ta² < vmax_i/ta ≤ vmax_i/ta_i = amax_i. Такой профиль допустим, но не обязательно
    минимален по времени. (Прежняя ветка T = 2·max √(dq_i/amax_i) учитывала только ускорения
    и при разных пределах сочленений нарушала предел скорости.)
    Возвращает ta, T, скорости полки и ускорения каждого сочленения."""
    dq = np.abs(np.asarray(dq, dtype=float)); vmax = np.asarray(vmax, dtype=float); amax = np.asarray(amax, dtype=float)
    ta = np.max(vmax / amax)
    T = np.max(dq / vmax) + ta
    if T < 2 * ta:                      # у ведущего сочленения нет полки — общий треугольник длительностью 2·ta
        T = 2 * ta
    v = dq / (T - ta)
    return ta, T, v, v / ta


# ====================================================================== S-кривая (ограничение рывка)
def scurve_time(dq, vmax, amax, jmax):
    """Семиучастковый профиль с пределами скорости, ускорения и рывка (движение из покоя в покой).
    Полный случай: t_j = amax/jmax, разгон t_acc = vmax/amax + t_j, T = dq/vmax + t_acc.
    Если vmax < amax²/jmax, ускорение не выходит на предел: t_j = √(vmax/jmax), t_acc = 2t_j.
    Если dq < vmax·t_acc, полки скорости нет: пиковая скорость уменьшается так, чтобы dq = v·t_acc(v).
    Возвращает словарь: tj (нарастание ускорения), ta_flat (полка ускорения), tc (полка скорости),
    tacc, T, vpeak, apeak, j."""
    dq = abs(float(dq))

    def acc_phase(v):
        if v >= amax ** 2 / jmax:
            return amax / jmax, v / amax - amax / jmax, amax
        tj = np.sqrt(v / jmax)
        return tj, 0.0, jmax * tj

    v = float(vmax)
    tj, taf, ap = acc_phase(v)
    tacc = 2 * tj + taf
    if dq < v * tacc:                   # полки скорости нет: ищем пиковую скорость делением пополам
        lo, hi = 0.0, v
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            tj_, taf_, _ = acc_phase(mid)
            if mid * (2 * tj_ + taf_) > dq:
                hi = mid
            else:
                lo = mid
        v = lo
        tj, taf, ap = acc_phase(v)
        tacc = 2 * tj + taf
    tc = max(dq / v - tacc, 0.0)
    return dict(tj=tj, ta_flat=taf, tc=tc, tacc=tacc, T=2 * tacc + tc, vpeak=v, apeak=ap, j=jmax)


def scurve_eval(t, q0, q1, vmax, amax, jmax):
    """Положение, скорость, ускорение и рывок S-кривой (кусочно-постоянный рывок на семи участках).
    После конца движения (t > T) — покой: q = q1, производные нулевые."""
    p = scurve_time(abs(q1 - q0), vmax, amax, jmax)
    sg = 1.0 if q1 >= q0 else -1.0
    j = p['j']
    durs = [p['tj'], p['ta_flat'], p['tj'], p['tc'], p['tj'], p['ta_flat'], p['tj']]
    jerks = [j, 0.0, -j, 0.0, -j, 0.0, j]
    t_in = np.asarray(t, dtype=float)
    if np.any(t_in > p['T']) or np.any(t_in < 0):
        q, v, a, jj = scurve_eval(np.clip(t_in, 0, p['T']), q0, q1, vmax, amax, jmax)
        out = (t_in < 0) | (t_in > p['T'])
        return q, np.where(out, 0.0, v), np.where(out, 0.0, a), np.where(out, 0.0, jj)
    t = t_in
    q = np.zeros_like(t); v = np.zeros_like(t); a = np.zeros_like(t); jj = np.zeros_like(t)
    s0 = np.array([0.0, 0.0, 0.0])                # (положение, скорость, ускорение) в начале участка
    t0 = 0.0
    for k, (d, jk) in enumerate(zip(durs, jerks)):
        m = (t >= t0) & ((t < t0 + d) if k < 6 else (t <= t0 + d + 1e-12))
        tau = t[m] - t0
        q[m] = s0[0] + s0[1] * tau + s0[2] * tau ** 2 / 2 + jk * tau ** 3 / 6
        v[m] = s0[1] + s0[2] * tau + jk * tau ** 2 / 2
        a[m] = s0[2] + jk * tau
        jj[m] = jk
        s0 = np.array([s0[0] + s0[1] * d + s0[2] * d ** 2 / 2 + jk * d ** 3 / 6,
                       s0[1] + s0[2] * d + jk * d ** 2 / 2, s0[2] + jk * d])
        t0 += d
    return q0 + sg * q, sg * v, sg * a, sg * jj


# ====================================================================== упругая нагрузка
def residual_vibration(t, acc, f, zeta=0.02):
    """Амплитуда колебаний упругой нагрузки θ'' + 2ζω_nθ' + ω_n²θ = −q̈(t) после окончания движения.
    t — равномерная сетка от 0 до момента t_end, после которого q̈ = 0; acc — q̈ на этой сетке.
    По интегралу Дюамеля: θ = Im z, z(t_end) = −(1/ω_d)∫ q̈(τ) e^{(−ζω_n + iω_d)(t_end − τ)} dτ,
    и амплитуда свободных колебаний в момент t_end равна |z(t_end)|."""
    t = np.asarray(t, dtype=float); acc = np.asarray(acc, dtype=float)
    wn = 2 * np.pi * f; wd = wn * np.sqrt(1 - zeta ** 2)
    ker = np.exp((-zeta * wn + 1j * wd) * (t[-1] - t))
    return abs(np.trapz(acc * ker, t)) / wd


def residual_pc(edges, values, f, zeta=0.02):
    """Тот же остаток, но точно (без сетки) для кусочно-постоянного ускорения: q̈ = values[k] на
    [edges[k], edges[k+1]); движение кончается в edges[-1]. Годится для трапеции и для её
    суммы с задержанной копией (двухимпульсный формирователь)."""
    wn = 2 * np.pi * f; wd = wn * np.sqrt(1 - zeta ** 2); s = -zeta * wn + 1j * wd
    e = np.asarray(edges, dtype=float); tend = e[-1]
    tot = sum(a * (np.exp(s * (tend - t0)) - np.exp(s * (tend - t1))) / s
              for t0, t1, a in zip(e[:-1], e[1:], values))
    return abs(tot) / wd


def flex_response(t, acc, f, zeta=0.02):
    """Отклонение θ(t) упругой нагрузки θ'' + 2ζω_nθ' + ω_n²θ = −q̈(t) из покоя (scipy.signal.lsim)."""
    from scipy.signal import lsim
    wn = 2 * np.pi * f
    _, y, _ = lsim(([-1.0], [1.0, 2 * zeta * wn, wn ** 2]), np.asarray(acc, dtype=float), np.asarray(t, dtype=float))
    return y


def zv_shaper(f, zeta=0.02):
    """Двухимпульсный формирователь (ZV, Сингер и Сиринг, 1990) для моды f, Гц, с демпфированием ζ:
    веса A1 = 1/(1 + K), A2 = K/(1 + K), K = exp(−ζπ/√(1 − ζ²)), задержка — половина периода
    затухающих колебаний π/ω_d. Возвращает (A1, A2, задержка)."""
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2))
    wd = 2 * np.pi * f * np.sqrt(1 - zeta ** 2)
    return 1 / (1 + K), K / (1 + K), np.pi / wd


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
    """Коэффициенты alpha_k (4.22) и правые части g_k (4.23, с исправленными индексами) для k = 1..n-1."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    h = np.diff(t)
    al = h[1:] / (h[1:] + h[:-1])
    g = 6.0 / (h[1:] + h[:-1]) * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    return al, g


def sweep_book(t, y, m0=0.0, mn=0.0, verbose=False):
    """Метод прогонки в форме учебника (4.27)–(4.32): x_k = c_k x_{k+1} + d_k, заданы q''_0 и q''_n.
    Возвращает вторые производные в узлах и коэффициенты прогонки."""
    al, g = spline_system(t, y)
    n = len(t) - 1
    c = np.zeros(n); d = np.zeros(n)
    c[0], d[0] = 0.0, m0
    for k in range(1, n):
        den = 2 + (1 - al[k - 1]) * c[k - 1]
        c[k] = -al[k - 1] / den
        d[k] = (g[k - 1] - (1 - al[k - 1]) * d[k - 1]) / den
        if verbose:
            print('k = %d: alpha = %.4f, g = %.4f, знаменатель = %.4f, c = %.4f, d = %.4f' % (k, al[k - 1], g[k - 1], den, c[k], d[k]))
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
    bc='natural'  — заданы ускорения на концах (left, right);
    bc='clamped'  — заданы скорости на концах v_0 = left, v_n = right;
    bc='periodic' — замкнутая траектория: требуется y[-1] = y[0], и скорости и ускорения на концах
                    совпадают. Система циклически трёхдиагональна (угловые элементы), здесь она решается
                    прямым методом — для учебных размеров этого достаточно."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    if bc == 'natural':
        return sweep_book(t, y, left, right)[0]
    n = len(t) - 1
    h = np.diff(t)
    if bc == 'periodic':
        if abs(y[-1] - y[0]) > 1e-9 * max(1.0, np.abs(y).max()):
            raise ValueError("periodic: значения в первом и последнем узлах должны совпадать")
        A = np.zeros((n, n)); b = np.zeros(n)
        for k in range(n):
            hl, hr = h[k - 1], h[k]                 # h[-1] = h[n-1]: левый сосед узла 0 — узел n-1
            yl = y[k - 1] if k > 0 else y[n - 1]
            al = hr / (hr + hl)
            A[k, k] = 2.0; A[k, (k + 1) % n] += al; A[k, (k - 1) % n] += 1 - al
            b[k] = 6.0 / (hr + hl) * ((y[k + 1] - y[k]) / hr - (y[k] - yl) / hl)
        m = np.linalg.solve(A, b)
        return np.r_[m, m[0]]
    al, g = spline_system(t, y)
    lo = np.zeros(n + 1); di = 2 * np.ones(n + 1); up = np.zeros(n + 1); rhs = np.zeros(n + 1)
    up[0] = 1.0; rhs[0] = 6 / h[0] * ((y[1] - y[0]) / h[0] - left)
    lo[n] = 1.0; rhs[n] = 6 / h[-1] * (right - (y[n] - y[n - 1]) / h[-1])
    lo[1:n] = 1 - al; up[1:n] = al; rhs[1:n] = g
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
    """ОЗК плоского двухзвенника; elbow=+1 — ветвь q₂ > 0 («локоть вниз» в осях курса: x вправо,
    y вверх), elbow=−1 — ветвь q₂ < 0 («локоть вверх»)."""
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


def ik2_chain(S, q_start, chain=True, tol=1e-9):
    """ОЗК в узлах S [N, 2]. chain=True — «цепочка близких целей»: начальное приближение для узла k —
    решение для узла k − 1 (для первого узла — q_start); chain=False — каждый узел из одного и того же
    q_start («сломанная цепочка»). Возвращает Q [N, 2] и число итераций по узлам."""
    q = np.array(q_start, dtype=float); Q = []; its = []
    for s in S:
        q, it = ik2_newton(s, q if chain else q_start, tol=tol)
        Q.append(q.copy()); its.append(it)
    return np.array(Q), np.array(its)


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


def chord_error(rho, n):
    """Ошибка хорды ε_ch = ρ(1 − cos(Δφ/2)) для окружности радиуса ρ, разбитой на n равных дуг."""
    return rho * (1 - np.cos(np.pi / n))


# ====================================================================== программные траектории схвата
def circle_ref(t, T=4.0, c=(1.0, 0.8), rho=0.4, profile=lam_poly5, R=None, lam=None):
    """Окружность примера 4.5: s*(t) и её скорость; φ = 2π·λ(t/T), λ — функция profile.
    R и lam — прежние имена радиуса и профиля (оставлены как псевдонимы)."""
    rho = rho if R is None else R
    profile = profile if lam is None else lam
    tau = np.clip(t / T, 0, 1)
    l, dl, _ = profile(tau)
    ph = 2 * np.pi * l
    s = np.array([c[0] + rho * np.cos(ph), c[1] + rho * np.sin(ph)])
    ds = np.array([-rho * np.sin(ph), rho * np.cos(ph)]) * 2 * np.pi * dl / T
    return s, ds


def line_ref(sig, y0=0.05, v=0.4, x0=-0.7):
    """Прямая y = y0 мимо основания; параметр sig — «программное время»."""
    return np.array([x0 + v * sig, y0]), np.array([v, 0.0])


def circle_path(s, c=(1.0, 0.8), rho=0.4, h=1e-5):
    """Путь в пространстве сочленений для окружности примера 4.5 как функция параметра s ∈ [0, 1]
    (φ = 2πs, ветвь q₂ > 0): q(s), q'(s), q''(s); производные — центральными разностями с шагом h."""
    c = np.asarray(c, dtype=float)
    def qs(x):
        return np.array([ik2(c + rho * np.array([np.cos(2 * np.pi * v), np.sin(2 * np.pi * v)])) for v in np.atleast_1d(x)])
    s = np.asarray(s, dtype=float)
    Q, Qa, Qb = qs(s), qs(s + h), qs(s - h)
    return Q, (Qa - Qb) / (2 * h), (Qa - 2 * Q + Qb) / h ** 2


# ====================================================================== оптимальная по времени параметризация
def topp_path(Qp, Qpp, s, vmax, amax):
    """Оптимальная по времени параметризация пути q(s), s ∈ [0, 1], из покоя в покой (идея TOPP-RA:
    переменная x = ṡ², на каждом шаге — маленькая задача линейного программирования).
    Qp, Qpp [N+1, n] — производные q'(s), q''(s) на равномерной сетке s; vmax, amax [n] — пределы.
    Ограничения: |q'ᵢ|·ṡ ≤ vmaxᵢ, |q'ᵢ s̈ + q''ᵢ ṡ²| ≤ amaxᵢ; на шаге x_{i+1} = x_i + 2Δs·s̈_i.
    Проход назад строит наибольшие x, из которых ещё можно остановиться к концу пути;
    проход вперёд выбирает наибольшее допустимое s̈. Возвращает ṡ(s), кривую предельной скорости
    ṡ_max(s) (по пределам скорости) и время прохождения ∫ds/ṡ."""
    from scipy.optimize import linprog
    Qp = np.asarray(Qp, dtype=float); Qpp = np.asarray(Qpp, dtype=float)
    vmax = np.asarray(vmax, dtype=float); amax = np.asarray(amax, dtype=float)
    N = len(s) - 1; ds = s[1] - s[0]; nj = Qp.shape[1]
    xmax = np.min((vmax[None, :] / np.maximum(np.abs(Qp), 1e-12)) ** 2, axis=1)
    kmax = np.zeros(N + 1)
    for i in range(N - 1, -1, -1):                  # проход назад: управляемые множества [0, kmax_i]
        A = []; b = []
        for j in range(nj):
            A.append([Qpp[i, j], Qp[i, j]]); b.append(amax[j])
            A.append([-Qpp[i, j], -Qp[i, j]]); b.append(amax[j])
        A.append([1, 2 * ds]); b.append(kmax[i + 1])
        A.append([-1, -2 * ds]); b.append(0)
        r = linprog([-1, 0], A_ub=A, b_ub=b, bounds=[(0, xmax[i]), (None, None)], method='highs')
        kmax[i] = r.x[0] if r.success else 0.0
    x = np.zeros(N + 1)
    for i in range(N):                              # проход вперёд: наибольшее допустимое s̈
        A = []; b = []
        for j in range(nj):
            A.append([Qp[i, j]]); b.append(amax[j] - Qpp[i, j] * x[i])
            A.append([-Qp[i, j]]); b.append(amax[j] + Qpp[i, j] * x[i])
        A.append([2 * ds]); b.append(min(kmax[i + 1], xmax[i + 1]) - x[i])
        A.append([-2 * ds]); b.append(x[i])
        r = linprog([-1], A_ub=A, b_ub=b, bounds=[(None, None)], method='highs')
        u = r.x[0] if r.success else 0.0
        x[i + 1] = max(0.0, x[i] + 2 * ds * u)
    sd = np.sqrt(x)
    T = np.sum(2 * ds / (sd[:-1] + sd[1:] + 1e-12))
    return sd, np.sqrt(xmax), T


def accel_limit_curve(Qp, Qpp, amax, xcap=1e4, iters=80):
    """Граница фазовой плоскости по пределам ускорений: наибольшее ṡ, при котором отрезок допустимых s̈
    из условий |q'ᵢ s̈ + q''ᵢ ṡ²| ≤ amaxᵢ ещё не пуст. Множество допустимых x = ṡ² при каждом s — отрезок
    [0, x*] (проекция выпуклого многогранника, содержащая нуль), поэтому x* ищется делением пополам.
    Qp, Qpp [N+1, n] — производные пути на сетке s; возвращает ṡ_acc(s) (не больше √xcap)."""
    Qp = np.asarray(Qp, dtype=float); Qpp = np.asarray(Qpp, dtype=float)
    amax = np.asarray(amax, dtype=float)[None, :]
    small = np.abs(Qp) < 1e-12
    c = np.where(small, 1.0, Qp)

    def feasible(x):
        b = Qpp * x[:, None]
        l1, u1 = (-amax - b) / c, (amax - b) / c
        lo_, up_ = np.minimum(l1, u1), np.maximum(l1, u1)
        lo_ = np.where(small, -np.inf, lo_); up_ = np.where(small, np.inf, up_)
        ok_small = np.all(~small | (np.abs(b) <= amax), axis=1)
        return (lo_.max(axis=1) <= up_.min(axis=1)) & ok_small

    lo = np.zeros(Qp.shape[0]); hi = np.full(Qp.shape[0], float(xcap))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = feasible(mid)
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return np.sqrt(lo)


# ====================================================================== управление по вектору скорости
def _damping(mu, lam, default):
    """Коэффициент демпфирования: новое имя mu, старое lam (оставлено как псевдоним)."""
    if mu is not None:
        return mu
    return default if lam is None else lam


def dls_pinv(J, mu=None, lam=None):
    """Демпфированная псевдообратная J^T (J J^T + mu^2 E)^-1; при mu = 0 — обычная псевдообратная.
    mu — коэффициент демпфирования μ (в лекциях); lam — его прежнее имя, работает как псевдоним
    (буква λ в курсе оставлена за параметром пути λ(t)). Можно передавать позиционно: dls_pinv(J, 0.15)."""
    mu = _damping(mu, lam, 0.0)
    if mu == 0:
        return np.linalg.pinv(J)
    return J.T @ np.linalg.inv(J @ J.T + mu ** 2 * np.eye(J.shape[0]))


def scale_velocities(qd, qmax):
    """Общее масштабирование: все компоненты делятся на κ = max(1, max|qd_i|/qmax_i). Возвращает (qd/κ, κ)."""
    k = max(1.0, float(np.max(np.abs(qd) / qmax)))
    return qd / k, k


def clip_velocities(qd, qmax):
    """Срезание по компонентам (так делать НЕ надо: меняется направление скорости схвата)."""
    return np.clip(qd, -np.asarray(qmax) * np.ones_like(qd), np.asarray(qmax) * np.ones_like(qd))


def track(K, dt=0.02, T=4.0, off=(0.0, 0.0), ref=circle_ref, lam=None, ff=True, mu=None, delay=0):
    """CLIK — слежение за программной траекторией: q' = J^+(𝒱* + K e), интегрирование по Эйлеру с тактом dt.
    off — начальное смещение схвата от программы; ff=False отключает прямую связь по скорости;
    mu — коэффициент демпфирования псевдообратной (0 — обычная J^+), lam — его прежнее имя;
    delay — запаздывание в тактах: скорость, вычисленная на такте k, применяется на такте k + delay
    (до этого приводы отрабатывают прежние команды, в начале — нулевые).
    Возвращает t, путь схвата P, норму ошибки E, скорости сочленений QD."""
    mu = _damping(mu, lam, 0.0)
    s0, _ = ref(0.0, T)
    q = ik2(s0 + np.array(off))
    ts = np.arange(0, T + 1e-9, dt); P = []; E = []; QD = []
    pending = [np.zeros(2)] * delay
    for t in ts:
        s, ds = ref(t, T)
        p = fk2(q); e = s - p
        qd = dls_pinv(J2(q), mu) @ ((ds if ff else 0.0) + K * e)
        P.append(p); E.append(np.linalg.norm(e)); QD.append(qd)
        pending.append(qd)
        q = q + dt * pending.pop(0)
    return ts, np.array(P), np.array(E), np.array(QD)


def regulate(mode, K, q0, target, dt=1e-3, tol=1e-3, Tmax=3.0):
    """Вывод схвата в неподвижную цель: mode='pinv' — q' = K·J⁺e (CLIK), mode='JT' — q' = K·Jᵀe
    (алгоритм с транспонированной матрицей Якоби). Малый шаг dt приближает непрерывное время.
    Возвращает момент, когда ‖e‖ < tol (или None), и историю (t, ‖e‖)."""
    q = np.array(q0, dtype=float); t = 0.0; hist = []
    while t < Tmax:
        e = np.asarray(target) - fk2(q)
        hist.append((t, np.linalg.norm(e)))
        if np.linalg.norm(e) < tol:
            return t, np.array(hist)
        J = J2(q)
        qd = (np.linalg.pinv(J) if mode == 'pinv' else J.T) @ (K * e)
        q = q + dt * qd; t += dt
    return None, np.array(hist)


def pass_near(y0, mode='inv', lam=None, v=0.4, dt=0.002, qmax=None, limit='scale', K=10.0, x0=-0.7, x1=0.7,
              mu=None, d=None):
    """Проход прямой y = y0 рядом с особой точкой (над основанием рука почти сложена, q₂ ≈ π − y₀).
    mode: 'inv' — обратная матрица Якоби, 'dls' — демпфированная псевдообратная с коэффициентом
          демпфирования mu (по умолчанию 0,15; lam — прежнее имя того же параметра).
    qmax/limit: None — без ограничения; 'clip' — срезание по компонентам; 'scale' — общее масштабирование;
                'time' — общее масштабирование и замедление программы в то же число раз.
    d — прежнее имя y0. Возвращает t, скорости QD, норму ошибки E, путь схвата P, путь программы S."""
    y0 = y0 if d is None else d
    mu = _damping(mu, lam, 0.15)
    Tn = (x1 - x0) / v
    q = ik2(np.array([x0, y0]))
    sig = 0.0; t = 0.0
    ts = []; QD = []; E = []; P = []; S = []
    while sig <= Tn + 1e-9 and t < 5 * Tn:
        s, ds = line_ref(sig, y0, v, x0)
        p = fk2(q); e = s - p
        J = J2(q)
        Ji = np.linalg.inv(J) if mode == 'inv' else dls_pinv(J, mu)
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


def speed_limit_line(y0, qmax, x0=-0.7, x1=0.7, n=14001):
    """Кривая предельной скорости схвата вдоль прямой y = y0: в каждой точке наибольшая скорость
    вдоль оси x, при которой ни одно сочленение не превышает qmax (ОЗК на ветви q₂ > 0).
    Возвращает x и v_lim(x)."""
    xs = np.linspace(x0, x1, n)
    vl = np.empty(n)
    for i, x in enumerate(xs):
        g = np.linalg.solve(J2(ik2(np.array([x, y0]))), [1.0, 0.0])     # q̇ при скорости схвата 1 м/с
        with np.errstate(divide='ignore'):                               # нулевая компонента q̇ не ограничивает
            vl[i] = np.min(np.asarray(qmax) / np.abs(g))
    return xs, vl
