# -*- coding: utf-8 -*-
"""aerial — учебный модуль кинематики летающих роботов (занятие 7).

Соглашения лекции 7: инерциальная система с осью z вверх, связанные оси «вперёд — влево — вверх»,
углы 3-2-1: R = Rz(psi) Ry(theta) Rx(phi); кватернион e = (e0, e1, e2, e3), скалярная часть первая;
матрица смешивания схемы «X» из раздела 7.5; аппарат Hummingbird (m = 0,5 кг, L = 0,175 м).

Расчётные функции (скопированы из генератора схем gen_figs_k7.py, числа совпадают с конспектом):
  Rx, Ry, Rz, R321, skew, Wmat, expm_so3, qmul, q2R, angerr, omega_loop, integrate_orientation;
  mod2pi, dubins_words, dubins_sample, dubins_all.
Дополнения для ноутбука:
  euler321, Wdirect                     — углы по матрице и прямая матрица omega = Wdirect * dTheta;
  mixer_x, mixer_plus, mixer_general, mixer_x_inv, hexa_geometry, allocate, hover_feasible;
  accel_to_tilt, simulate_quad, ref_circle, ref_square — модель «наклон — ускорение» с каскадом;
  turn_rate, turn_radius, crab_angle, simulate_plane   — самолёт;
  draw_quad                             — рисование корпуса квадрокоптера в 3D.
Только numpy, scipy (linprog в hover_feasible) и matplotlib.
"""
import numpy as np

G = 9.81

# ---- аппарат Hummingbird (глава Пауэрс, Меллингера, Кумара)
M_Q = 0.5                      # масса, кг
L_Q = 0.175                    # от центра масс до оси мотора, м
K_F = 6.11e-8                  # Н/(об/мин)^2
K_M = 1.5e-9                   # Н*м/(об/мин)^2
KAPPA = K_M / K_F              # 0,02455 м
D_Q = L_Q / np.sqrt(2)         # плечо схемы «X», 0,1237 м
F_HOVER = M_Q * G / 4          # 1,226 Н
F_MAX = 2 * F_HOVER            # 2,45 Н


# ====================================================================== ориентация
def Rx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R321(psi, th, phi):
    """Матрица поворота по углам рыскания, тангажа и крена: R = Rz(psi) Ry(theta) Rx(phi)."""
    return Rz(psi).dot(Ry(th)).dot(Rx(phi))


def euler321(R):
    """Углы (psi, theta, phi) по матрице поворота; теряет смысл при cos(theta) = 0."""
    return np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def Wdirect(phi, th):
    """Прямая матрица: omega_b = Wdirect * (dphi, dtheta, dpsi); det = cos(theta)."""
    return np.array([[1, 0, -np.sin(th)],
                     [0, np.cos(phi), np.sin(phi) * np.cos(th)],
                     [0, -np.sin(phi), np.cos(phi) * np.cos(th)]])


def Wmat(phi, th):
    """Матрица W: (dphi, dtheta, dpsi) = W(phi, theta) * omega_b."""
    return np.array([[1, np.sin(phi) * np.tan(th), np.cos(phi) * np.tan(th)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(th), np.cos(phi) / np.cos(th)]])


def expm_so3(w):
    """Матричная экспонента exp(Lambda(w)) по формуле Родрига."""
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    K = skew(w / a)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K.dot(K)


def qmul(a, b):
    """Кватернионное произведение, скалярная часть первая."""
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angerr(Ra, Rb):
    """Угол (град) между двумя ориентациями; Rb предварительно ортогонализуется."""
    U, s, Vt = np.linalg.svd(Rb); Rb = U.dot(Vt)
    c = (np.trace(Ra.T.dot(Rb)) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def omega_loop(t):
    """Угловая скорость манёвра «петля» из раздела 7.4, рад/с."""
    return np.array([0.2 * np.sin(1.3 * t), 0.8, 0.1 * np.cos(t)])


def integrate_orientation(h=0.01, T=20.0, sub=20, omega=omega_loop):
    """Счисление ориентации методом Эйлера тремя способами. Столбцы результата:
    t | тангаж эталона, град | ошибка углов Эйлера | ошибка матрицы (с ортогонализацией) |
    ошибка кватерниона (с нормировкой) | ||R^T R - E|| без ортогонализации | | ||e|| - 1 | без нормировки."""
    N = int(round(T / h))
    Rref = np.eye(3); E = np.zeros(3); Rm = np.eye(3); Rn = np.eye(3); q = np.array([1.0, 0, 0, 0]); qn = q.copy()
    out = np.zeros((N, 7))
    for i in range(N):
        t = i * h; w = omega(t)
        E = E + h * Wmat(E[0], E[1]).dot(w)
        Rm = Rm.dot(np.eye(3) + h * skew(w))
        U, s, Vt = np.linalg.svd(Rm); Rm = U.dot(Vt)
        Rn = Rn.dot(np.eye(3) + h * skew(w))
        q = q + 0.5 * h * qmul(q, np.r_[0, w]); q = q / np.linalg.norm(q)
        qn = qn + 0.5 * h * qmul(qn, np.r_[0, w])
        for j in range(sub):
            Rref = Rref.dot(expm_so3(omega(t + (j + 0.5) * h / sub) * h / sub))
        out[i] = (t + h, -np.degrees(np.arcsin(np.clip(Rref[2, 0], -1, 1))), angerr(Rref, R321(E[2], E[1], E[0])),
                  angerr(Rref, Rm), angerr(Rref, q2R(q)), np.linalg.norm(Rn.T.dot(Rn) - np.eye(3)), abs(np.linalg.norm(qn) - 1))
    return out


# ====================================================================== матрица смешивания
def mixer_general(xy, sigma, kappa=KAPPA):
    """Матрица смешивания 4 x n: T = sum f, tau_x = sum y f, tau_y = -sum x f, tau_z = -kappa sum sigma f.
    sigma = +1 — винт вращается против часовой стрелки (вид сверху), -1 — по часовой."""
    xy = np.asarray(xy, float); sigma = np.asarray(sigma, float)
    return np.vstack([np.ones(len(sigma)), xy[:, 1], -xy[:, 0], -kappa * sigma])


def mixer_x(d=D_Q, kappa=KAPPA):
    """Схема «X»: 1 — спереди справа, 2 — сзади слева (оба против часовой), 3 — спереди слева, 4 — сзади справа."""
    return mixer_general([(d, -d), (-d, d), (d, d), (-d, -d)], [1, 1, -1, -1], kappa)


def mixer_plus(L=L_Q, kappa=KAPPA):
    """Схема «+»: 1 — спереди, 2 — слева, 3 — сзади, 4 — справа; 1 и 3 вращаются по часовой."""
    return mixer_general([(L, 0), (0, L), (-L, 0), (0, -L)], [-1, 1, -1, 1], kappa)


def mixer_x_inv(d=D_Q, kappa=KAPPA):
    """Обратная матрица схемы «X» по формуле M^-1 = M^T D^-1, D = diag(4, 4d^2, 4d^2, 4kappa^2)."""
    M = mixer_x(d, kappa)
    return M.T.dot(np.diag(1.0 / np.array([4, 4 * d * d, 4 * d * d, 4 * kappa * kappa])))


def hexa_geometry(L=L_Q):
    """Гексакоптер: шесть лучей через 60 градусов, направления вращения чередуются."""
    ang = np.radians(30 + 60 * np.arange(6))
    return np.c_[L * np.cos(ang), L * np.sin(ang)], np.array([1, -1, 1, -1, 1, -1])


def allocate(u, M=None, fmax=F_MAX):
    """Распределение управления: f = M^+ u, затем обрезание 0 <= f <= fmax.
    Возвращает (f без ограничений, f после обрезания, реализованная команда M f)."""
    if M is None:
        M = mixer_x()
    f = np.linalg.pinv(M).dot(u)
    fc = np.clip(f, 0.0, fmax)
    return f, fc, M.dot(fc)


def hover_feasible(M, weight, fmax):
    """Есть ли тяги 0 <= f <= fmax с M f = (weight, 0, 0, 0)? Возвращает (ответ, f, запас).
    Запас — наибольшее s, при котором s <= f_i <= fmax - s (линейное программирование)."""
    from scipy.optimize import linprog
    n = M.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_eq = np.c_[M, np.zeros(4)]; b_eq = np.array([weight, 0, 0, 0.0])
    A_ub = np.vstack([np.c_[-np.eye(n), np.ones(n)], np.c_[np.eye(n), np.ones(n)]])
    b_ub = np.r_[np.zeros(n), fmax * np.ones(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * (n + 1), method='highs')
    if not res.success:
        return False, None, None
    s = res.x[-1]
    return bool(s >= -1e-9), res.x[:n], s


# ====================================================================== квадрокоптер: наклон -> ускорение
def accel_to_tilt(acc, psi=0.0):
    """Требуемые тангаж и крен по требуемому горизонтальному ускорению (линеаризованная модель)."""
    c, s = np.cos(psi), np.sin(psi)
    return (acc[0] * c + acc[1] * s) / G, (acc[0] * s - acc[1] * c) / G


def ref_circle(rho=2.0, period=10.0):
    """Программная траектория примера 7.5: окружность радиуса rho, высота 1 м."""
    w = 2 * np.pi / period

    def ref(t):
        p = np.array([rho * np.cos(w * t), rho * np.sin(w * t), 1.0])
        v = np.array([-rho * w * np.sin(w * t), rho * w * np.cos(w * t), 0.0])
        a = np.array([-rho * w * w * np.cos(w * t), -rho * w * w * np.sin(w * t), 0.0])
        return p, v, a
    return ref


def ref_square(side=4.0, speed=1.0):
    """Квадрат со стороной side, обход с постоянной скоростью; в углах скорость меняется скачком."""
    P = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], float)

    def ref(t):
        s = (speed * t) % (4 * side); k = int(s // side); r = s - k * side
        e = (P[k + 1] - P[k]) / side
        return np.r_[P[k] + r * e, 1.0], np.r_[speed * e, 0.0], np.zeros(3)
    return ref


def simulate_quad(ref, T=30.0, dt=0.005, Kp=2.0, Kd=2.5, tilt_max=np.radians(20), tau=0.15, psi=0.0, p0=None):
    """Каскад: ПД-регулятор положения -> требуемые углы -> инерционное звено (внутренний контур) -> тяга.
    Корпус: m p'' = -m g z0 + T R z0, тяга T = m (g + z''_d) / (cos(phi) cos(theta)).
    Возвращает словарь массивов t, p, pref, phi, theta, phi_d, theta_d."""
    N = int(round(T / dt))
    pr0, vr0, _ = ref(0.0)
    p = np.array(pr0 if p0 is None else p0, float); v = np.zeros(3); phi = th = 0.0
    out = dict(t=np.zeros(N), p=np.zeros((N, 3)), pref=np.zeros((N, 3)), phi=np.zeros(N), theta=np.zeros(N), phi_d=np.zeros(N), theta_d=np.zeros(N))
    for i in range(N):
        t = i * dt
        pr, vr, ar = ref(t)
        ad = ar + Kd * (vr - v) + Kp * (pr - p)
        th_d, phi_d = accel_to_tilt(ad, psi)
        th_d = np.clip(th_d, -tilt_max, tilt_max); phi_d = np.clip(phi_d, -tilt_max, tilt_max)
        out['t'][i] = t; out['p'][i] = p; out['pref'][i] = pr
        out['phi'][i] = phi; out['theta'][i] = th; out['phi_d'][i] = phi_d; out['theta_d'][i] = th_d
        phi += dt * (phi_d - phi) / tau; th += dt * (th_d - th) / tau
        zb = R321(psi, th, phi)[:, 2]
        thrust = M_Q * (G + ad[2]) / max(zb[2], 0.2)
        acc = thrust / M_Q * zb - np.array([0, 0, G])
        v = v + dt * acc; p = p + dt * v
    return out


# ====================================================================== самолёт
def turn_rate(V, phi):
    """Модуль угловой скорости координированного разворота, рад/с."""
    return G / V * np.tan(np.abs(phi))


def turn_radius(V, phi):
    return V ** 2 / (G * np.tan(np.abs(phi)))


def crab_angle(Va, wind, chi=0.0):
    """Курс psi, при котором путевая скорость направлена по путевому углу chi; возвращает (psi, Vg)."""
    wx, wy = wind
    w_cross = -wx * np.sin(chi) + wy * np.cos(chi)          # составляющая ветра поперёк линии пути
    w_along = wx * np.cos(chi) + wy * np.sin(chi)
    psi = chi + np.arcsin(-w_cross / Va)
    return psi, Va * np.cos(psi - chi) + w_along


def simulate_plane(Va, psi_fun, wind=(0.0, 0.0), T=30.0, dt=0.05, start=(0.0, 0.0)):
    """Кинематическая модель самолёта в горизонтальной плоскости: x' = Va cos(psi) + wx, y' = Va sin(psi) + wy."""
    N = int(round(T / dt)); xy = np.zeros((N + 1, 2)); xy[0] = start
    for i in range(N):
        psi = psi_fun(i * dt, xy[i])
        xy[i + 1] = xy[i] + dt * (Va * np.array([np.cos(psi), np.sin(psi)]) + np.asarray(wind))
    return xy


# ---------------------------------------------------------------------- пути Дубинса
def mod2pi(a):
    return a - 2 * np.pi * np.floor(a / (2 * np.pi))


def dubins_words(alpha, beta, d):
    """Длины участков (в долях радиуса) для шести типов пути Дубинса в нормированной постановке:
    старт (0, 0, alpha), цель (d, 0, beta). Формулы Шкеля и Лумельского."""
    sa, sb, ca, cb, cab = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta), np.cos(alpha - beta)
    res = {}
    p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp = np.arctan2(cb - ca, d + sa - sb)
        res['LSL'] = (mod2pi(-alpha + tmp), np.sqrt(p2), mod2pi(beta - tmp))
    p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp = np.arctan2(ca - cb, d - sa + sb)
        res['RSR'] = (mod2pi(alpha - tmp), np.sqrt(p2), mod2pi(-beta + tmp))
    p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2); tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        res['LSR'] = (mod2pi(-alpha + tmp), p, mod2pi(-mod2pi(beta) + tmp))
    p2 = d * d - 2 + 2 * cab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = np.sqrt(p2); tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        res['RSL'] = (mod2pi(alpha - tmp), p, mod2pi(beta - tmp))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(2 * np.pi - np.arccos(tmp)); t = mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + p / 2)
        res['RLR'] = (t, p, mod2pi(alpha - beta - t + p))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(2 * np.pi - np.arccos(tmp)); t = mod2pi(-alpha - np.arctan2(ca - cb, d + sa - sb) + p / 2)
        res['LRL'] = (t, p, mod2pi(mod2pi(beta) - alpha - t + p))
    return res


def dubins_sample(start, word, segs, rho, n=80):
    """Точки пути: start = (x, y, psi), word — например 'LSL', segs — длины участков в долях rho."""
    x, y, th = start
    xs, ys = [x], [y]
    for ch, s in zip(word, segs):
        for k in range(n):
            ds = s * rho / n
            if ch == 'S':
                x += ds * np.cos(th); y += ds * np.sin(th)
            else:
                sg = 1 if ch == 'L' else -1
                dth = sg * ds / rho
                x += rho * sg * (np.sin(th + dth) - np.sin(th)); y += -rho * sg * (np.cos(th + dth) - np.cos(th)); th += dth
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys), th


def dubins_all(start, goal, rho):
    """Все допустимые пути Дубинса между позами, по возрастанию длины: (длина, тип, участки, xs, ys).
    Каждый кандидат проверяется прямым построением: конец пути должен попасть в цель."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    D = np.hypot(dx, dy); phi = np.arctan2(dy, dx)
    words = dubins_words(mod2pi(start[2] - phi), mod2pi(goal[2] - phi), D / rho)
    out = []
    for wname, segs in words.items():
        xs, ys, th = dubins_sample(start, wname, segs, rho)
        err = np.hypot(xs[-1] - goal[0], ys[-1] - goal[1])
        if err < 1e-3 * rho + 1e-6:
            out.append((sum(segs) * rho, wname, segs, xs, ys))
    out.sort(key=lambda r: r[0])
    return out


# ====================================================================== рисование
def draw_quad(ax, R, c=(0.0, 0.0, 0.0), L=1.0, rr=0.38, thrust=None, accel=None, x_config=True):
    """Корпус квадрокоптера на трёхмерных осях matplotlib: лучи, винты, связанные оси,
    вектор тяги (красный) и горизонтальное ускорение (янтарный)."""
    NAVY, AMB, TEAL, RED, GREY, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#2C7A5A'
    c = np.asarray(c, float)
    angs = np.radians([45, 135, 225, 315]) if x_config else np.radians([0, 90, 180, 270])
    tt = np.linspace(0, 2 * np.pi, 40)
    for a in angs:
        tip = c + R.dot([L * np.cos(a), L * np.sin(a), 0])
        ax.plot(*np.c_[c, tip], color=NAVY, lw=3)
        ring = np.array([tip + R.dot([rr * np.cos(u), rr * np.sin(u), 0.05]) for u in tt])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=TEAL, lw=1.5)
    for k, (col, nm) in enumerate(((RED, 'x_b'), (GREEN, 'y_b'), (TEAL, 'z_b'))):
        e = c + 1.3 * L * R[:, k]
        ax.plot(*np.c_[c, e], color=col, lw=2); ax.text(e[0], e[1], e[2], nm, color=col, fontsize=10)
    if thrust is not None:
        e = c + thrust * R[:, 2]
        ax.quiver(c[0], c[1], c[2], *(e - c), color=RED, lw=2.5, arrow_length_ratio=0.12)
    if accel is not None:
        ax.quiver(c[0], c[1], c[2], accel[0], accel[1], 0.0, color=AMB, lw=2.5, arrow_length_ratio=0.2)
    for d in np.eye(3):
        ax.plot(*np.c_[c, c + 1.6 * L * d], color=GREY, lw=0.8, ls='--')
    lim = 1.7 * L
    ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim); ax.set_zlim(c[2] - lim, c[2] + lim)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def check():
    """Самопроверка: числа примеров 7.1–7.7 конспекта."""
    rad = np.radians
    print('7.1:', R321(rad(30), rad(10), rad(-20)).dot([5, 0, -1]).round(2))
    print('7.2:', Wmat(rad(20), rad(60)).dot([0.1, 0.2, 0.3]).round(3), Wmat(rad(20), rad(85)).dot([0.1, 0.2, 0.3]).round(2))
    print('7.3:', mixer_x_inv().dot([M_Q * G, 0.05, 0, 0.01]).round(2))
    print('7.4: %.2f %.2f' % (4 * D_Q * F_HOVER, 4 * KAPPA * F_HOVER))
    print('turn: %.1f deg/s, %.1f m' % (np.degrees(turn_rate(20, rad(30))), turn_radius(20, rad(30))))
    psi, Vg = crab_angle(20, (0, 6)); print('7.6: %.1f deg, %.1f m/s' % (np.degrees(psi), Vg))
    for st, go in (((0, 0, 0), (400, 250, np.pi)), ((0, 0, np.pi / 2), (90, 0, -np.pi / 2))):
        print('7.7:', [(w, round(Ln)) for Ln, w, _, _, _ in dubins_all(st, go, 70.6)])


if __name__ == '__main__':
    check()
