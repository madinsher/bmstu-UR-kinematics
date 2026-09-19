# -*- coding: utf-8 -*-
"""wheeled — учебный модуль кинематики колёсных мобильных роботов (занятие 6).

Обозначения конспекта лекции 6: q = (x, y, theta) — положение корпуса в неподвижной системе,
u = (vx, vy, omega) — скорость корпуса в связанной системе (x вперёд, y влево), phi_dot — скорости колёс.
Расчётные функции шагов интегрирования, квадратного маршрута, парковки и манёвра скобки Ли
скопированы из генератора схем gen_figs_k6.py, числа совпадают с конспектом.

  rotz, body_to_world                      — связь скоростей в двух системах;
  dd_fk, dd_ik, dd_icc, dd_G, dd_saturate   — дифференциальная схема шасси (примеры 6.1, 6.2);
  step_exact, step_euler, step_mid, integrate, odo_update, square_run — ПЗК на шаге и одометрия (пример 6.3);
  tri_front, tri_rear, ackermann, rear_wheel_speeds — трицикл и принцип Аккермана (пример 6.4);
  omni_H                                    — трёхколёсная омни-платформа (пример 6.5);
  mec_H, mec_Hpinv, mec_fk, mec_vmax        — меканум-платформа (примеры 6.6, 6.7);
  roll_row, slide_row, swedish_row, maneuverability — связи колёс и манёвренность шасси;
  lie_maneuver, lie_bracket_num, parking_path — неголономность (раздел 6.10);
  simulate_body                             — интегрирование движения корпуса по команде u(t);
  draw_diffdrive, draw_mecanum, draw_omni3, draw_car, draw_path — рисование платформ и траекторий;
  check                                     — самопроверка (печать чисел примеров 6.1–6.7).
Только numpy и matplotlib.
"""
import numpy as np

NAVY, AMB, TEAL, RED, GREY, ICE, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#CADCFC', '#2C7A5A'
BG = '#F3F6FB'

# ---- параметры примеров конспекта, метры
DD_R, DD_L = 0.05, 0.30                 # дифференциальная схема: радиус колеса, колея
OMNI_R, OMNI_LL = 0.05, 0.20            # омни-платформа: радиус колеса, радиус окружности колёс
OMNI_ALPHA = np.radians([0.0, 120.0, 240.0])
MEC_R, MEC_LX, MEC_LY = 0.05, 0.235, 0.15   # меканум-платформа (близко к youBot): радиус, полубаза, полуколея
ACK_D, ACK_L = 2.7, 1.6                 # автомобиль примера 6.4: база, колея


# ====================================================================== системы координат
def rotz(th):
    """Матрица R_z(theta): q_dot = R_z(theta) u."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_world(u, th):
    return rotz(th).dot(np.asarray(u, float))


# ====================================================================== дифференциальная схема
def dd_fk(phr, phl, r=DD_R, l=DD_L):
    """Прямая задача о скорости: скорости колёс (рад/с) -> (v, omega)."""
    Vr, Vl = r * phr, r * phl
    return (Vr + Vl) / 2.0, (Vr - Vl) / l


def dd_ik(v, w, r=DD_R, l=DD_L):
    """Обратная задача о скорости: (v, omega) -> (phi_r, phi_l), рад/с."""
    return (v + l / 2.0 * w) / r, (v - l / 2.0 * w) / r


def dd_icc(q, v, w):
    """Радиус поворота R = v/omega (со знаком) и координаты МЦВ; при omega = 0 — (inf, None)."""
    if abs(w) < 1e-12:
        return np.inf, None
    R = v / w
    return R, np.array([q[0] - R * np.sin(q[2]), q[1] + R * np.cos(q[2])])


def dd_G(th, r=DD_R, l=DD_L):
    """Матрица G(theta) = R_z(theta) F размера 3x2: q_dot = G (phi_r, phi_l)."""
    F = np.array([[r / 2, r / 2], [0.0, 0.0], [r / l, -r / l]])
    return rotz(th).dot(F)


def dd_saturate(phr, phl, pmax, mode='scale'):
    """Ограничение скоростей колёс: 'clip' — срезание каждого колеса, 'scale' — общий коэффициент."""
    ph = np.array([phr, phl], float)
    if mode == 'clip':
        return np.clip(ph, -pmax, pmax)
    k = min(1.0, pmax / np.abs(ph).max())
    return ph * k


# ====================================================================== ПЗК на шаге и одометрия
def step_exact(q, v, w, dt):
    x, y, th = q
    if abs(w) < 1e-9:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th])
    return np.array([x + v / w * (np.sin(th + w * dt) - np.sin(th)),
                     y - v / w * (np.cos(th + w * dt) - np.cos(th)), th + w * dt])


def step_euler(q, v, w, dt):
    x, y, th = q
    return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th + w * dt])


def step_mid(q, v, w, dt):
    x, y, th = q
    return np.array([x + v * dt * np.cos(th + w * dt / 2), y + v * dt * np.sin(th + w * dt / 2), th + w * dt])


STEPS = {'euler': step_euler, 'mid': step_mid, 'exact': step_exact}


def integrate(q0, v, w, dt, n, method='exact'):
    """n шагов интегрирования при постоянных v, omega. Возвращает массив (n+1) x 3."""
    f = STEPS[method]; q = np.array(q0, float); P = [q.copy()]
    for i in range(n):
        q = f(q, v, w, dt); P.append(q.copy())
    return np.array(P)


def odo_update(q, dphr, dphl, r=DD_R, l=DD_L, method='exact'):
    """Один такт одометрии по приращениям углов колёс (рад)."""
    dsr, dsl = r * dphr, r * dphl
    return STEPS[method](q, (dsr + dsl) / 2.0, (dsr - dsl) / l, 1.0)


def square_run(rr, rl, base, r_nom=0.05, l_nom=0.30, side=4.0, ccw=True, n=200):
    """Робот отрабатывает квадрат по номинальной модели; реальные радиусы rr, rl и колея base."""
    q = np.array([0.0, 0.0, 0.0]); path = [q.copy()]
    sgn = 1.0 if ccw else -1.0
    for k in range(4):
        dphi = side / r_nom / n
        for i in range(n):
            vr, vl = rr * dphi, rl * dphi
            q = step_exact(q, (vr + vl) / 2, (vr - vl) / base, 1.0); path.append(q.copy())
        dphi = (np.pi / 2) * (l_nom / 2) / r_nom / n
        for i in range(n):
            vr, vl = sgn * rr * dphi, -sgn * rl * dphi
            q = step_exact(q, (vr + vl) / 2, (vr - vl) / base, 1.0); path.append(q.copy())
    return np.array(path)


# ====================================================================== трицикл и Аккерман
def tri_front(vs, delta, d):
    """Трицикл с приводным рулевым колесом: (v_s, delta) -> (v, omega) центра задней оси."""
    return vs * np.cos(delta), vs * np.sin(delta) / d


def tri_rear(v, delta, d):
    """Трицикл с задним приводом (велосипедная модель): (v, delta) -> (v, omega)."""
    return v, v * np.tan(delta) / d


def ackermann(R, d=ACK_D, l=ACK_L):
    """Углы внутреннего и наружного рулевых колёс и эквивалентного трицикла (рад)."""
    return np.arctan(d / (R - l / 2.0)), np.arctan(d / (R + l / 2.0)), np.arctan(d / R)


def rear_wheel_speeds(v, R, l):
    """Линейные скорости внутреннего и наружного задних колёс при повороте радиуса R."""
    w = v / R
    return w * (R - l / 2.0), w * (R + l / 2.0)


# ====================================================================== омни-платформа
def omni_H(r=OMNI_R, L=OMNI_LL, alphas=OMNI_ALPHA):
    """Матрица обратной задачи о скорости: phi_dot = H u, строка i = (-sin a_i, cos a_i, L)/r."""
    a = np.asarray(alphas, float)
    return np.column_stack([-np.sin(a), np.cos(a), L * np.ones(len(a))]) / r


# ====================================================================== меканум-платформа
def mec_H(r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Матрица 4x3 обратной задачи о скорости; колёса: 1 — ПЛ, 2 — ПП, 3 — ЗЛ, 4 — ЗП."""
    k = lx + ly
    return np.array([[1, -1, -k], [1, 1, k], [1, 1, -k], [1, -1, k]], float) / r


def mec_Hpinv(r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Псевдообратная H+ = (H^T H)^-1 H^T в явном виде."""
    k = lx + ly
    return r / 4.0 * np.array([[1, 1, 1, 1], [-1, 1, 1, -1], [-1 / k, 1 / k, -1 / k, 1 / k]], float)


def mec_fk(phi, r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Прямая задача о скорости по наименьшим квадратам.
    Возвращает (u, невязка phi - H u, левая часть условия совместности phi1 + phi2 - phi3 - phi4)."""
    phi = np.asarray(phi, float)
    u = mec_Hpinv(r, lx, ly).dot(phi)
    return u, phi - mec_H(r, lx, ly).dot(u), phi[0] + phi[1] - phi[2] - phi[3]


def mec_vmax(direction, omega=0.0, pmax=20.0, r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Наибольшая скорость в направлении direction (рад) при заданной omega: граница октаэдра."""
    rest = r * pmax - (lx + ly) * abs(omega)
    if rest <= 0:
        return 0.0
    return rest / (abs(np.cos(direction)) + abs(np.sin(direction)))


# ====================================================================== связи колёс и манёвренность
def roll_row(alpha, beta, l):
    """Строка условия качения фиксированного (рулевого) колеса: row . u = r phi_dot."""
    return np.array([np.sin(alpha + beta), -np.cos(alpha + beta), -l * np.cos(beta)])


def slide_row(alpha, beta, l):
    """Строка условия отсутствия бокового скольжения: row . u = 0."""
    return np.array([np.cos(alpha + beta), np.sin(alpha + beta), l * np.sin(beta)])


def swedish_row(alpha, beta, gamma, l):
    """Строка связи шведского колеса: row . u = r phi_dot cos(gamma)."""
    return np.array([np.sin(alpha + beta + gamma), -np.cos(alpha + beta + gamma), -l * np.cos(beta + gamma)])


def maneuverability(fixed=(), steer=()):
    """fixed, steer — списки колёс (alpha, beta, l). Возвращает C1, rank C1, delta_m, delta_s, delta_M."""
    rf = [slide_row(*w) for w in fixed]; rs = [slide_row(*w) for w in steer]
    C1 = np.array(rf + rs).reshape(-1, 3)
    rank = int(np.linalg.matrix_rank(C1, tol=1e-9)) if len(C1) else 0
    ds = int(np.linalg.matrix_rank(np.array(rs), tol=1e-9)) if rs else 0
    dm = 3 - rank
    return C1, rank, dm, ds, dm + ds


# ====================================================================== неголономность
def lie_maneuver(eps, n=20):
    """Манёвр из четырёх шагов: вперёд, поворот, назад, поворот обратно (единичные скорости).
    Возвращает (траектория, узловые положения)."""
    q = np.zeros(3); P = [q.copy()]; marks = [q.copy()]
    for (v, w) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        for i in range(n):
            q = step_exact(q, v, w, eps / float(n)); P.append(q.copy())
        marks.append(q.copy())
    return np.array(P), np.array(marks)


def g1(q):
    return np.array([np.cos(q[2]), np.sin(q[2]), 0.0])


def g2(q):
    return np.array([0.0, 0.0, 1.0])


def lie_bracket_num(f, g, q, h=1e-6):
    """Скобка Ли [f, g] = (dg/dq) f - (df/dq) g численным дифференцированием."""
    q = np.asarray(q, float); n = len(q)
    Df = np.zeros((n, n)); Dg = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        Df[:, j] = (f(q + e) - f(q - e)) / (2 * h); Dg[:, j] = (g(q + e) - g(q - e)) / (2 * h)
    return Dg.dot(f(q)) - Df.dot(g(q))


def parking_path(Rm=4.0, psi=np.radians(20.0), ncyc=4, n=40):
    """Параллельная парковка велосипедной модели: задний ход с рулём вправо, затем влево, возврат вперёд.
    Возвращает (траектория, узловые положения)."""
    q = np.array([0.0, 0.0, 0.0]); path = [q.copy()]; keys = [q.copy()]
    for k in range(ncyc):
        for (v, w) in ((-1.0, 1.0 / Rm), (-1.0, -1.0 / Rm)):
            for i in range(n):
                q = step_exact(q, v, w, psi * Rm / n); path.append(q.copy())
            keys.append(q.copy())
        for i in range(n):
            q = step_exact(q, 1.0, 0.0, 2 * Rm * np.sin(psi) / n); path.append(q.copy())
        keys.append(q.copy())
    return np.array(path), np.array(keys)


# ====================================================================== движение корпуса по команде
def simulate_body(u_fun, q0=(0.0, 0.0, 0.0), T=5.0, n=500):
    """Интегрирование q_dot = R_z(theta) u(t) методом средней точки. u_fun(t) -> (vx, vy, omega)."""
    q = np.array(q0, float); P = [q.copy()]; dt = T / float(n)
    for i in range(n):
        u = np.asarray(u_fun((i + 0.5) * dt), float)
        q = q + rotz(q[2] + u[2] * dt / 2).dot(u) * dt; P.append(q.copy())
    return np.array(P)


# ====================================================================== рисование
def _b2w(q, bx, by):
    c, s = np.cos(q[2]), np.sin(q[2])
    return np.array([q[0] + c * bx - s * by, q[1] + s * bx + c * by])


def _poly(ax, q, pts, fc, ec, lw=1.2, alpha=1.0, z=3):
    P = np.array([_b2w(q, bx, by) for bx, by in pts])
    ax.fill(P[:, 0], P[:, 1], facecolor=fc, edgecolor=ec, lw=lw, alpha=alpha, zorder=z)


def _wheel(ax, q, cx, cy, L, W, ang=0.0, fc=NAVY, ec=NAVY, alpha=1.0, z=5):
    c, s = np.cos(ang), np.sin(ang)
    pts = [(cx + c * a - s * b, cy + s * a + c * b) for a, b in ((L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2))]
    _poly(ax, q, pts, fc, ec, lw=1.0, alpha=alpha, z=z)


def draw_path(ax, P, color=RED, lw=1.6, label=None, ls='-'):
    P = np.asarray(P); ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, label=label, ls=ls, zorder=4)


def draw_diffdrive(ax, q, l=DD_L, color=NAVY, alpha=1.0, scale=1.0):
    """Робот с дифференциальной схемой: круглый корпус диаметра l*scale, два колеса, метка носа."""
    a = l * scale / 2.0; t = np.linspace(0, 2 * np.pi, 60)
    ax.fill(q[0] + a * np.cos(t), q[1] + a * np.sin(t), facecolor=BG, edgecolor=color, lw=1.2, alpha=alpha, zorder=3)
    for sy in (1, -1):
        _wheel(ax, q, 0.0, sy * a, 0.8 * a, 0.25 * a, fc=color, ec=color, alpha=alpha)
    n = _b2w(q, a, 0.0); ax.plot([q[0], n[0]], [q[1], n[1]], color=color, lw=1.5, alpha=alpha, zorder=5)


def draw_mecanum(ax, q, lx=MEC_LX, ly=MEC_LY, color=NAVY, alpha=1.0, scale=1.0):
    """Меканум-платформа, вид сверху: штриховка — верхние ролики (рисунок X), треугольник — нос."""
    lx, ly = lx * scale, ly * scale; wl, ww = 0.75 * lx, 0.42 * ly
    bl, bw = lx + 0.35 * wl, ly - ww / 2 - 0.05 * ly
    _poly(ax, q, [(bl, bw), (-bl, bw), (-bl, -bw), (bl, -bw)], BG, color, alpha=alpha)
    _poly(ax, q, [(bl, 0.0), (bl - 0.3 * lx, 0.3 * bw), (bl - 0.3 * lx, -0.3 * bw)], color, color, alpha=alpha, z=4)
    top = [1, -1, -1, 1]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        _wheel(ax, q, sx * lx, sy * ly, wl, ww, fc=ICE, ec=color, alpha=alpha)
        for s in np.linspace(-wl / 2 + ww / 2, wl / 2 - ww / 2, 3):
            p0 = _b2w(q, sx * lx + s - top[i] * 0.4 * ww, sy * ly - 0.4 * ww)
            p1 = _b2w(q, sx * lx + s + top[i] * 0.4 * ww, sy * ly + 0.4 * ww)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.4, alpha=alpha, zorder=6)


def draw_omni3(ax, q, L=OMNI_LL, color=NAVY, alpha=1.0, scale=1.0):
    """Трёхколёсная омни-платформа: круглый корпус, три колеса по касательной, метка носа."""
    a = L * scale; t = np.linspace(0, 2 * np.pi, 60)
    ax.fill(q[0] + a * np.cos(t), q[1] + a * np.sin(t), facecolor=BG, edgecolor=color, lw=1.2, alpha=alpha, zorder=3)
    for k, al in enumerate(OMNI_ALPHA):
        _wheel(ax, q, a * np.cos(al), a * np.sin(al), 0.55 * a, 0.2 * a, ang=al + np.pi / 2, fc=(AMB if k == 0 else color), ec=color, alpha=alpha)
    n = _b2w(q, 0.7 * a, 0.0); ax.plot([q[0], n[0]], [q[1], n[1]], color=color, lw=1.5, alpha=alpha, zorder=5)


def draw_car(ax, q, d=ACK_D, l=ACK_L, di=0.0, do=0.0, color=NAVY, fc=ICE, alpha=1.0, fill=True, wheels=True):
    """Автомобиль: точка q — центр задней оси; di, do — углы левого и правого передних колёс."""
    ov = 0.18 * d
    pts = [(d + 2 * ov, l / 2 + 0.05), (-2 * ov, l / 2 + 0.05), (-2 * ov, -l / 2 - 0.05), (d + 2 * ov, -l / 2 - 0.05)]
    if fill:
        _poly(ax, q, pts, fc, color, alpha=alpha, z=2)
    else:
        P = np.array([_b2w(q, bx, by) for bx, by in pts + pts[:1]]); ax.plot(P[:, 0], P[:, 1], color=color, lw=0.9, alpha=alpha, zorder=2)
    if not wheels:
        return
    wl, ww = 0.24 * d, 0.08 * d
    for sy in (1, -1):
        _wheel(ax, q, 0.0, sy * l / 2, wl, ww, fc=color, ec=color, alpha=alpha)
    _wheel(ax, q, d, l / 2, wl, ww, ang=di, fc=AMB, ec=AMB, alpha=alpha)
    _wheel(ax, q, d, -l / 2, wl, ww, ang=do, fc=AMB, ec=AMB, alpha=alpha)


# ====================================================================== самопроверка
def check():
    v, w = dd_fk(12.0, 8.0); print('6.1: v = %.3f, w = %.3f, R = %.3f' % (v, w, v / w))
    pr, pl = dd_ik(0.8, 2.0); print('6.2: phi = %.1f, %.1f' % (pr, pl))
    for m in ('clip', 'scale'):
        s = dd_saturate(pr, pl, 20.0, m); v, w = dd_fk(s[0], s[1]); print('     %s: v = %.3f, w = %.3f, R = %.3f' % (m, v, w, v / w))
    n = int(round(2 * np.pi / 0.5)); qt = step_exact(np.zeros(3), 0.5, 0.5, n * 0.5)
    print('6.3:', ['%s %.4f' % (m, np.hypot(*(integrate(np.zeros(3), 0.5, 0.5, 0.5, n, m)[-1, :2] - qt[:2]))) for m in ('euler', 'mid', 'exact')])
    di, do, de = ackermann(6.0); print('6.4: %.1f %.1f %.1f, ratio %.2f' % (np.degrees(di), np.degrees(do), np.degrees(de), 6.8 / 5.2))
    print('6.5:', omni_H().dot([0.2, 0.1, 0.5]).round(2))
    print('6.6:', mec_H().dot([0.3, 0.2, 0.5]).round(2))
    u, res, c = mec_fk([10.0, 10.4, 9.6, 10.2]); print('6.7: u =', u.round(4), 'res =', res.round(3), 'compat = %.1f' % c)
    P = square_run(0.05 * 1.005, 0.05 * 0.995, 0.30)[-1]; print('square, radii: %.3f %.3f %.1f deg' % (P[0], P[1], np.degrees(P[2])))
    P = square_run(0.05, 0.05, 0.30 * 1.03)[-1]; print('square, track: %.3f %.3f %.1f deg' % (P[0], P[1], np.degrees(P[2])))
    print('parking: %.3f m per cycle' % (2 * 4.0 * (1 - np.cos(np.radians(20)))))
    print('Lie eps = 0.6:', lie_maneuver(0.6)[1][-1].round(4))


if __name__ == '__main__':
    check()
