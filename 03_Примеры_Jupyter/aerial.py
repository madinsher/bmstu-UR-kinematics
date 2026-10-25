# -*- coding: utf-8 -*-
"""aerial — учебный модуль кинематики летающих роботов (занятие 7).

Соглашения лекции 7 (и файла 00_Паспорт/Обозначения.md, раздел 2.8):
  инерциальная система Oxyz с осью z вверх, связанные оси «вперёд — влево — вверх» (FLU);
  углы 3-2-1: R = Rz(psi) Ry(theta) Rx(phi), phi — крен, theta — тангаж, psi — рыскание;
  кватернион e = (e0, e1, e2, e3), скалярная часть первая (как у Бирда — Маклейна и в PX4;
  scipy и geometry_msgs/Quaternion хранят (x, y, z, w));
  матрица смешивания u = M f, u = (T, tau_x, tau_y, tau_z), eps_i = +1 — винт вращается против часовой
  стрелки при виде сверху; реактивный момент винта Q_i = kappa f_i; аппарат Hummingbird (m = 0,5 кг, L = 0,175 м);
  командное ускорение p''_cmd = p''* + Kv (p'* - p') + Kp (p* - p); радиус разворота rho; шаг интегрирования dt.

Состав:
  ориентация        Rx, Ry, Rz, R321, euler321, skew, vee, Wdirect, Wmat, expm_so3, logm_so3, qmul, q2R, q2R_h,
                    angerr, omega_loop, integrate_orientation;
  оси ROS и PX4     P_ENU_NED, F_FLU_FRD, enu_to_ned, yaw_enu_to_ned, R_to_ned_frd;
  распределение     mixer_general, mixer_x, mixer_plus, mixer_x_inv, hexa_geometry, allocate (обрезание),
                    allocate_scale_moments, allocate_scale_all, allocate_priority, allocate_weighted, hover_feasible;
  квадрокоптер      accel_to_tilt (линейный пересчёт), thrust_vector (точный пересчёт), attitude_from_zb,
                    flat_state (плоскостность: R, T и omega^b через рывок), circle_derivs,
                    ref_circle, ref_square, simulate_quad (модель «наклон — ускорение»),
                    simulate_quad_full (полный каскад: ориентация, угловая скорость, распределение);
  самолёт           turn_rate, turn_radius, crab_angle, simulate_plane, dubins_words, dubins_sample, dubins_all;
  оценка ориентации complementary_pitch, mahony_hover;
  рисование         draw_quad.
Только numpy, scipy (linprog, lsq_linear) и matplotlib. Самопроверка: python aerial.py.
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
# Тензор инерции в лекции не задан. Для полного каскада принято допущение порядка величины для аппарата
# массой 0,5 кг с лучом 17,5 см (это не паспортные данные Hummingbird): регулятор угловой скорости
# умножает требуемое угловое ускорение на ту же матрицу I, поэтому качественные выводы от неё не зависят.
I_Q = np.diag([2.5e-3, 2.5e-3, 4.5e-3])   # кг*м^2


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
    """Кососимметрическая матрица Lambda(w): Lambda(w) a = w x a."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(S):
    """Обратная к skew операция: вектор по кососимметрической матрице."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def Wdirect(phi, th):
    """Прямая матрица: omega^b = Wdirect * (dphi, dtheta, dpsi); det = cos(theta)."""
    return np.array([[1, 0, -np.sin(th)],
                     [0, np.cos(phi), np.sin(phi) * np.cos(th)],
                     [0, -np.sin(phi), np.cos(phi) * np.cos(th)]])


def Wmat(phi, th):
    """Матрица W: (dphi, dtheta, dpsi) = W(phi, theta) * omega^b."""
    return np.array([[1, np.sin(phi) * np.tan(th), np.cos(phi) * np.tan(th)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(th), np.cos(phi) / np.cos(th)]])


def expm_so3(w):
    """Матричная экспонента exp(Lambda(w)) по формуле Родрига: ось k = w/|w|, угол alpha = |w|."""
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / a)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K.dot(K)


def logm_so3(R):
    """Вектор поворота alpha*k = log(R)^v, alpha = arccos((tr R - 1)/2) (при alpha < 179 град)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); a = np.arccos(c)
    if a < 1e-9:
        return vee(R - R.T) / 2
    return a / (2 * np.sin(a)) * vee(R - R.T)


def qmul(a, b):
    """Кватернионное произведение (Гамильтон), скалярная часть первая."""
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q2R(q):
    """Матрица поворота по единичному кватерниону (формула лекции, использует |e| = 1)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def q2R_h(q):
    """Однородная форма той же формулы: (e0^2 - |v|^2) E + 2 v v^T + 2 e0 Lambda(v) = |e|^2 * поворот.
    Для ненормированного кватерниона растягивает векторы ровно в |e|^2 раз."""
    w, v = q[0], np.asarray(q[1:], float)
    return (w * w - v.dot(v)) * np.eye(3) + 2 * np.outer(v, v) + 2 * w * skew(v)


def angerr(Ra, Rb):
    """Угол (град) между двумя ориентациями; Rb предварительно ортогонализуется."""
    U, s, Vt = np.linalg.svd(Rb); Rb = U.dot(Vt)
    c = (np.trace(Ra.T.dot(Rb)) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def omega_loop(t):
    """Угловая скорость манёвра «кувырок» из раздела 7.4 (вращение на месте с опусканием носа), рад/с."""
    return np.array([0.2 * np.sin(1.3 * t), 0.8, 0.1 * np.cos(t)])


INTEGRATION_COLUMNS = ('t', 'тангаж эталона, град', 'ошибка углов', 'ошибка матрицы + ортогонализация',
                       'ошибка кватерниона + нормировка', '||R^T R - E||_F без ортогонализации',
                       '| ||e|| - 1 | без нормировки', 'ошибка экспоненты, omega в начале шага',
                       'ошибка экспоненты, omega в середине шага', 'sigma_max(R) - 1 без ортогонализации',
                       '||e||^2 - 1 без нормировки (растяжение матрицы q2R_h)')


def integrate_orientation(h=0.01, T=20.0, sub=20, omega=omega_loop):
    """Счисление ориентации четырьмя способами (раздел 7.4); h — шаг Delta t.
    Углы, матрица и кватернион — методом Эйлера первого порядка; экспоненциальное обновление R exp(h Lambda(omega))
    — с угловой скоростью в начале и в середине шага. Эталон — экспоненциальное обновление с шагом h/sub
    и угловой скоростью в середине подшага. Столбцы результата — в INTEGRATION_COLUMNS."""
    N = int(round(T / h))
    Rref = np.eye(3); E = np.zeros(3); Rm = np.eye(3); Rn = np.eye(3); Re0 = np.eye(3); Re1 = np.eye(3)
    q = np.array([1.0, 0, 0, 0]); qn = q.copy()
    out = np.zeros((N, len(INTEGRATION_COLUMNS)))
    for i in range(N):
        t = i * h; w = omega(t)
        E = E + h * Wmat(E[0], E[1]).dot(w)
        Rm = Rm.dot(np.eye(3) + h * skew(w))
        U, s, Vt = np.linalg.svd(Rm); Rm = U.dot(Vt)
        Rn = Rn.dot(np.eye(3) + h * skew(w))
        q = q + 0.5 * h * qmul(q, np.r_[0, w]); q = q / np.linalg.norm(q)
        qn = qn + 0.5 * h * qmul(qn, np.r_[0, w])
        Re0 = Re0.dot(expm_so3(h * w)); Re1 = Re1.dot(expm_so3(h * omega(t + h / 2)))
        for j in range(sub):
            Rref = Rref.dot(expm_so3(omega(t + (j + 0.5) * h / sub) * h / sub))
        out[i] = (t + h, -np.degrees(np.arcsin(np.clip(Rref[2, 0], -1, 1))), angerr(Rref, R321(E[2], E[1], E[0])),
                  angerr(Rref, Rm), angerr(Rref, q2R(q)), np.linalg.norm(Rn.T.dot(Rn) - np.eye(3)), abs(np.linalg.norm(qn) - 1),
                  angerr(Rref, Re0), angerr(Rref, Re1), np.linalg.svd(Rn, compute_uv=False)[0] - 1, qn.dot(qn) - 1)
    return out


# ====================================================================== оси ROS (ENU, FLU) и автопилота (NED, FRD)
P_ENU_NED = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])     # p_NED = P p_ENU  (det = +1)
F_FLU_FRD = np.diag([1., -1, -1])                             # a^FRD = F a^FLU  (поворот на 180 град вокруг x_b)


def enu_to_ned(p):
    return P_ENU_NED.dot(p)


def yaw_enu_to_ned(psi):
    """Курс: от востока против часовой (ENU) -> от севера по часовой (NED): psi_NED = pi/2 - psi_ENU."""
    a = np.pi / 2 - psi
    return (a + np.pi) % (2 * np.pi) - np.pi


def R_to_ned_frd(R):
    """Ориентация FLU -> ENU (матрица лекции) в ориентацию FRD -> NED: R' = P R F."""
    return P_ENU_NED.dot(R).dot(F_FLU_FRD)


# ====================================================================== матрица смешивания и распределение
def mixer_general(xy, eps, kappa=KAPPA):
    """Матрица смешивания 4 x n: T = sum f, tau_x = sum y f, tau_y = -sum x f, tau_z = -kappa sum eps f.
    eps_i = +1 — винт вращается против часовой стрелки (вид сверху), -1 — по часовой."""
    xy = np.asarray(xy, float); eps = np.asarray(eps, float)
    return np.vstack([np.ones(len(eps)), xy[:, 1], -xy[:, 0], -kappa * eps])


def mixer_x(d=D_Q, kappa=KAPPA, eps=(1, 1, -1, -1)):
    """Схема «X» (нумерация Quadrotor X в PX4): 1 — спереди справа, 2 — сзади слева (оба против часовой),
    3 — спереди слева, 4 — сзади справа (по часовой). Аргумент eps позволяет «перепутать» направление мотора."""
    return mixer_general([(d, -d), (-d, d), (d, d), (-d, -d)], eps, kappa)


def mixer_plus(L=L_Q, kappa=KAPPA):
    """Схема «+»: 1 — спереди, 2 — слева, 3 — сзади, 4 — справа; 1 и 3 вращаются по часовой."""
    return mixer_general([(L, 0), (0, L), (-L, 0), (0, -L)], [-1, 1, -1, 1], kappa)


def mixer_x_inv(d=D_Q, kappa=KAPPA):
    """Обратная матрица схемы «X» по формуле M^-1 = M^T D^-1, D = diag(4, 4d^2, 4d^2, 4kappa^2)."""
    M = mixer_x(d, kappa)
    return M.T.dot(np.diag(1.0 / np.array([4, 4 * d * d, 4 * d * d, 4 * kappa * kappa])))


def hexa_geometry(L=L_Q, layout='PNPNPN'):
    """Гексакоптер: шесть лучей через 60 градусов (первый — под 30 градусов к x_b), винты по кругу против часовой.
    layout — направления вращения: P — против часовой стрелки (eps = +1), N — по часовой (eps = -1)."""
    ang = np.radians(30 + 60 * np.arange(6))
    return np.c_[L * np.cos(ang), L * np.sin(ang)], np.array([1 if ch == 'P' else -1 for ch in layout])


def allocate(u, M=None, fmax=F_MAX):
    """Распределение с покомпонентным обрезанием: f = M^+ u, затем 0 <= f <= fmax.
    Возвращает (f без ограничений, f после обрезания, реализованная команда M f)."""
    if M is None:
        M = mixer_x()
    f = np.linalg.pinv(M).dot(u)
    fc = np.clip(f, 0.0, fmax)
    return f, fc, M.dot(fc)


def _lp_max_scale(M, fixed, scaled, fmax):
    """max s в [0, 1] при M f = fixed + s * scaled, 0 <= f <= fmax (линейное программирование)."""
    from scipy.optimize import linprog
    n = M.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_eq = np.c_[M, -np.asarray(scaled, float)]
    res = linprog(c, A_eq=A_eq, b_eq=np.asarray(fixed, float), bounds=[(0, fmax)] * n + [(0, 1)], method='highs')
    if not res.success:
        return 0.0, None
    return res.x[-1], res.x[:n]


def allocate_scale_moments(u, M=None, fmax=F_MAX):
    """Тяга сохраняется, все три момента умножаются на одно наибольшее допустимое число s.
    Возвращает (s, f, M f)."""
    M = mixer_x() if M is None else M; u = np.asarray(u, float)
    s, f = _lp_max_scale(M, np.r_[u[0], 0, 0, 0], np.r_[0, u[1:]], fmax)
    return s, f, (M.dot(f) if f is not None else None)


def allocate_scale_all(u, M=None, fmax=F_MAX):
    """Весь вектор u (вместе с тягой) умножается на наибольшее допустимое s. Возвращает (s, f, M f)."""
    M = mixer_x() if M is None else M; u = np.asarray(u, float)
    s, f = _lp_max_scale(M, np.zeros(4), u, fmax)
    return s, f, (M.dot(f) if f is not None else None)


def _bound_scale(c, d, lo, hi):
    """Наибольшее s в [0, 1], при котором lo <= c + d s <= hi покомпонентно (c уже допустим)."""
    s = 1.0
    for ci, di in zip(c, d):
        if di > 1e-15:
            s = min(s, (hi - ci) / di)
        elif di < -1e-15:
            s = min(s, (lo - ci) / di)
    return max(0.0, s)


_INV_CACHE = {}


def _inv4(M):
    """Обратная матрица смешивания 4 x 4 с запоминанием (в моделировании её спрашивают на каждом шаге)."""
    key = M.tobytes()
    Mi = _INV_CACHE.get(key)
    if Mi is None:
        if len(_INV_CACHE) > 64:
            _INV_CACHE.clear()
        Mi = _INV_CACHE[key] = np.linalg.inv(M)
    return Mi


def _cross3(a, b):
    """Векторное произведение трёхмерных векторов (быстрее np.cross на одиночных векторах)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def allocate_priority(u, M=None, fmax=F_MAX, fmin=0.0):
    """Приоритетное распределение «крен и тангаж -> тяга -> рыскание».
    1) Моменты крена и тангажа сохраняются целиком, если это возможно хотя бы при какой-то тяге
       (при нулевом рыскании); иначе они масштабируются наибольшим допустимым множителем.
    2) Тяга выбирается ближайшей к запрошенной при уже зафиксированных крене и тангаже.
    3) Рыскание урезается до грани многогранника достижимых команд.
    Для квадратной M (квадрокоптер) — точные формулы, для n > 4 — три задачи линейного программирования.
    Возвращает (f, M f)."""
    M = mixer_x() if M is None else M; u = np.asarray(u, float)
    n = M.shape[1]
    if n == 4:
        Mi = _inv4(M); a = Mi[:, 0]; B = Mi[:, 1:3].dot(u[1:3]); dz = Mi[:, 3] * u[3]
        if np.any(a <= 0):
            raise ValueError('ожидается схема, у которой тяга входит во все моторы с положительным знаком')

        def T_interval(s):
            lo = np.max((fmin - s * B) / a); hi = np.min((fmax - s * B) / a); return lo, hi
        lo, hi = T_interval(1.0)
        if lo > hi:                                    # шаг 1: крен и тангаж не помещаются — делим отрезок пополам
            s0, s1 = 0.0, 1.0
            for _ in range(60):
                sm_ = 0.5 * (s0 + s1); l_, h_ = T_interval(sm_)
                s0, s1 = (sm_, s1) if l_ <= h_ else (s0, sm_)
            s_rp = s0; lo, hi = T_interval(s_rp)
        else:
            s_rp = 1.0
        Tc = float(np.clip(u[0], lo, hi))             # шаг 2: тяга
        base = a * Tc + s_rp * B
        s_y = _bound_scale(base, dz, fmin, fmax)       # шаг 3: рыскание
        f = np.clip(base + s_y * dz, fmin, fmax)
        return f, M.dot(f)
    # общий случай n > 4: последовательность задач линейного программирования
    from scipy.optimize import linprog
    bnd = [(fmin, fmax)] * n
    # шаг 1: max s_rp, тяга свободна, tau_z = 0
    c = np.zeros(n + 1); c[-1] = -1
    A = np.c_[M[1:4], -np.r_[u[1:3], 0.0]]
    r = linprog(c, A_eq=A, b_eq=np.zeros(3), bounds=bnd + [(0, 1)], method='highs'); s_rp = r.x[-1] * (1 - 1e-9)
    # шаг 2: min |T - T_req| при tau_xy = s_rp tau_xy, tau_z = 0
    c = np.zeros(n + 1); c[-1] = 1
    A_ub = np.vstack([np.r_[M[0], -1], np.r_[-M[0], -1]]); b_ub = np.array([u[0], -u[0]])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.c_[M[1:4], np.zeros(3)], b_eq=np.r_[s_rp * u[1:3], 0.0],
                bounds=bnd + [(0, None)], method='highs'); Tc = M[0].dot(r.x[:n])
    # шаг 3: max s_y при зафиксированных T и tau_xy
    s_y, f = _lp_max_scale(M, np.r_[Tc, s_rp * u[1:3], 0.0] * np.r_[1, 1, 1, 0] , np.r_[0, 0, 0, u[3]], fmax)
    return f, M.dot(f)


def allocate_weighted(u, M=None, fmax=F_MAX, w=(10.0, 10.0, 10.0, 1.0)):
    """Распределение как задача оптимизации (в духе Дарема): min ||diag(w) (M f - u)||^2 при 0 <= f <= fmax
    (scipy.optimize.lsq_linear). Веса задают приоритеты каналов (T, tau_x, tau_y, tau_z). Возвращает (f, M f)."""
    from scipy.optimize import lsq_linear
    M = mixer_x() if M is None else M; W = np.diag(w)
    res = lsq_linear(W.dot(M), W.dot(u), bounds=(0.0, fmax))
    return res.x, M.dot(res.x)


def hover_feasible(M, weight, fmax, rows=None):
    """Есть ли тяги 0 <= f <= fmax с M f = (weight, 0, 0, 0)? Возвращает (ответ, f, запас).
    Запас — наибольшее s, при котором s <= f_i <= fmax - s (линейное программирование).
    rows — какие строки требовать (по умолчанию все четыре; (0, 1, 2) — отказ от рыскания)."""
    from scipy.optimize import linprog
    rows = list(range(4)) if rows is None else list(rows)
    n = M.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_eq = np.c_[M[rows], np.zeros(len(rows))]; b_eq = np.array([weight, 0, 0, 0.0])[rows]
    A_ub = np.vstack([np.c_[-np.eye(n), np.ones(n)], np.c_[np.eye(n), np.ones(n)]])
    b_ub = np.r_[np.zeros(n), fmax * np.ones(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * (n + 1), method='highs')
    if not res.success:
        return False, None, None
    s = res.x[-1]
    return bool(s >= -1e-9), res.x[:n], s


# ====================================================================== квадрокоптер: наклон -> ускорение
def accel_to_tilt(acc, psi=0.0):
    """Линейный пересчёт: заданные тангаж и крен по командному горизонтальному ускорению (малые углы)."""
    c, s = np.cos(psi), np.sin(psi)
    return (acc[0] * c + acc[1] * s) / G, (acc[0] * s - acc[1] * c) / G


def attitude_from_zb(zb, psi):
    """Матрица R = (x_b y_b z_b) по оси тяги z_b и курсу psi: y_c = (-sin psi, cos psi, 0), x_b = y_c x z_b / |.|."""
    yc = np.array([-np.sin(psi), np.cos(psi), 0.0])
    xb = _cross3(yc, zb); xb = xb / np.linalg.norm(xb)
    return np.c_[xb, _cross3(zb, xb), zb]


def thrust_vector(acc_cmd, psi=0.0, m=M_Q):
    """Точный пересчёт: a_T = p''_cmd + g z0, z_b* = a_T/|a_T|, T = m |a_T|, R* по курсу. Возвращает (R*, T, a_T)."""
    aT = np.asarray(acc_cmd, float) + np.array([0, 0, G])
    zb = aT / np.linalg.norm(aT)
    return attitude_from_zb(zb, psi), m * np.linalg.norm(aT), aT


def flat_state(acc, jerk, psi, dpsi, m=M_Q):
    """Дифференциальная плоскостность (§ 7.6, прил. 7.Б): по p'', p''' и (psi, psi') находим R, T, dT/dt и omega^b.
    h = (m/T)(j - (z_b.j) z_b), omega_x = -h.y_b, omega_y = h.x_b, omega_z = -s_phi theta' + c_phi c_theta psi'."""
    R, T, aT = thrust_vector(acc, psi, m)
    xb, yb, zb = R[:, 0], R[:, 1], R[:, 2]
    j = np.asarray(jerk, float)
    h = m / T * (j - zb.dot(j) * zb)
    wx, wy = -h.dot(yb), h.dot(xb)
    _, th, ph = euler321(R)
    dth = (wy - np.sin(ph) * np.cos(th) * dpsi) / np.cos(ph)
    wz = -np.sin(ph) * dth + np.cos(ph) * np.cos(th) * dpsi
    return R, T, m * zb.dot(j), np.array([wx, wy, wz])


def circle_derivs(t, rho=2.0, period=10.0, z=1.0):
    """Окружность примера 7.7: положение, скорость, ускорение и рывок."""
    w = 2 * np.pi / period; c, s = np.cos(w * t), np.sin(w * t)
    return (np.array([rho * c, rho * s, z]), np.array([-rho * w * s, rho * w * c, 0.0]),
            np.array([-rho * w * w * c, -rho * w * w * s, 0.0]), np.array([rho * w ** 3 * s, -rho * w ** 3 * c, 0.0]))


def ref_circle(rho=2.0, period=10.0):
    """Программная траектория примера 7.7: окружность радиуса rho, высота 1 м. ref(t) -> (p*, v*, a*)."""
    def ref(t):
        p, v, a, _ = circle_derivs(t, rho, period); return p, v, a
    return ref


def ref_square(side=4.0, speed=1.0):
    """Квадрат со стороной side, обход с постоянной скоростью; в углах скорость меняется скачком."""
    P = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], float)

    def ref(t):
        s = (speed * t) % (4 * side); k = int(s // side); r = s - k * side
        e = (P[k + 1] - P[k]) / side
        return np.r_[P[k] + r * e, 1.0], np.r_[speed * e, 0.0], np.zeros(3)
    return ref


def simulate_quad(ref, T=30.0, dt=0.005, Kp=2.0, Kv=2.5, tilt_max=np.radians(20), tau=0.15, psi=0.0, p0=None, Kd=None):
    """Простейший каскад (модель «наклон — ускорение»): ПД-регулятор положения -> линейный пересчёт в заданные
    углы theta*, phi* с ограничением tilt_max -> инерционное звено с постоянной времени tau вместо внутренних
    контуров -> корпус m p'' = -m g z0 + T R z0, T = m (g + z''_cmd) / (z_b)_z. Kd — прежнее имя Kv.
    Возвращает словарь массивов t, p, pref, phi, theta, phi_d, theta_d, tilt (полный наклон оси тяги)."""
    if Kd is not None:
        Kv = Kd
    N = int(round(T / dt))
    pr0, vr0, _ = ref(0.0)
    p = np.array(pr0 if p0 is None else p0, float); v = np.zeros(3); phi = th = 0.0
    out = dict(t=np.zeros(N), p=np.zeros((N, 3)), pref=np.zeros((N, 3)), phi=np.zeros(N), theta=np.zeros(N),
               phi_d=np.zeros(N), theta_d=np.zeros(N), tilt=np.zeros(N))
    for i in range(N):
        t = i * dt
        pr, vr, ar = ref(t)
        ad = ar + Kv * (vr - v) + Kp * (pr - p)
        th_d, phi_d = accel_to_tilt(ad, psi)
        th_d = np.clip(th_d, -tilt_max, tilt_max); phi_d = np.clip(phi_d, -tilt_max, tilt_max)
        out['t'][i] = t; out['p'][i] = p; out['pref'][i] = pr
        out['phi'][i] = phi; out['theta'][i] = th; out['phi_d'][i] = phi_d; out['theta_d'][i] = th_d
        out['tilt'][i] = np.arccos(np.cos(phi) * np.cos(th))
        phi += dt * (phi_d - phi) / tau; th += dt * (th_d - th) / tau
        zb = R321(psi, th, phi)[:, 2]
        thrust = M_Q * (G + ad[2]) / max(zb[2], 0.2)
        acc = thrust / M_Q * zb - np.array([0, 0, G])
        v = v + dt * acc; p = p + dt * v
    return out


def simulate_quad_full(ref, T=20.0, dt=0.004, Kp=2.0, Kv=2.5, K_att=(8.0, 8.0, 3.0), K_rate=(30.0, 30.0, 15.0),
                       tilt_max=np.radians(35), fmax=F_MAX, alloc='priority', M_alloc=None, M_plant=None,
                       psi_ref=None, mapping='exact', I=I_Q, p0=None):
    """Полный каскад раздела 7.6 на модели твёрдого тела:
      положение и скорость (ПД)  ->  p''_cmd;
      пересчёт в тягу и ориентацию: mapping='exact' — z_b* = a_T/|a_T|, T = m a_T . z_b (на текущую ось);
                                    mapping='linear' — малые углы, T = m (g + z''_cmd);
      ориентация (П по e_R = 1/2 (R*^T R - R^T R*)^v)  ->  omega* = -K_att e_R;
      угловая скорость (П)  ->  tau = I K_rate (omega* - omega) + omega x I omega;
      распределение: alloc='priority' (allocate_priority) или 'clip' (обрезание), по матрице M_alloc;
      корпус: m p'' = -m g z0 + T R z0, I omega' = tau - omega x I omega, R <- R exp(dt Lambda(omega)),
      где T и tau считаются по «настоящей» матрице M_plant (по умолчанию она совпадает с M_alloc).
    psi_ref(t) — заданный курс (по умолчанию 0). Шаг dt = 4 мс: при dt = 2 мс результаты совпадают
    до миллиметров, а счёт вдвое дольше. Возвращает словарь массивов."""
    M_alloc = mixer_x() if M_alloc is None else M_alloc
    M_plant = M_alloc if M_plant is None else M_plant
    psi_ref = (lambda t: 0.0) if psi_ref is None else psi_ref
    Ka, Kr = np.asarray(K_att, float), np.asarray(K_rate, float)
    Iinv = np.linalg.inv(I); Mp_inv = np.linalg.pinv(M_alloc)
    N = int(round(T / dt))
    pr0, _, _ = ref(0.0)
    p = np.array(pr0 if p0 is None else p0, float); v = np.zeros(3); R = np.eye(3); w = np.zeros(3)
    keys = dict(t=1, p=3, pref=3, tilt=1, psi=1, T=1, f=M_alloc.shape[1], w=3, eR=1, sat=1, tau_cmd=3, tau_real=3)
    out = {k: np.zeros((N, n)) if n > 1 else np.zeros(N) for k, n in keys.items()}
    for i in range(N):
        t = i * dt
        pr, vr, ar = ref(t)
        acc = ar + Kv * (vr - v) + Kp * (pr - p)
        psi_d = psi_ref(t)
        if mapping == 'exact':
            aT = acc + np.array([0, 0, G])
            # ограничение наклона: горизонтальная часть урезается, вертикальная (тяга на вес) в приоритете
            hor = np.hypot(aT[0], aT[1]); hmax = max(aT[2], 0.1) * np.tan(tilt_max)
            if hor > hmax:
                aT[:2] *= hmax / hor
            Rd = attitude_from_zb(aT / np.linalg.norm(aT), psi_d)
            Tc = M_Q * aT.dot(R[:, 2])
        else:
            th_d, ph_d = accel_to_tilt(acc, psi_d)
            th_d = np.clip(th_d, -tilt_max, tilt_max); ph_d = np.clip(ph_d, -tilt_max, tilt_max)
            Rd = R321(psi_d, th_d, ph_d); Tc = M_Q * (G + acc[2])
        eR = 0.5 * vee(Rd.T.dot(R) - R.T.dot(Rd))
        w_d = -Ka * eR
        tau = I.dot(Kr * (w_d - w)) + _cross3(w, I.dot(w))
        u = np.array([Tc, tau[0], tau[1], tau[2]])
        if alloc == 'priority':
            f, _ = allocate_priority(u, M_alloc, fmax)
        else:
            f = np.clip(Mp_inv.dot(u), 0.0, fmax)
        ureal = M_plant.dot(f)
        out['t'][i] = t; out['p'][i] = p; out['pref'][i] = pr; out['tilt'][i] = np.arccos(np.clip(R[2, 2], -1, 1))
        out['psi'][i] = np.arctan2(R[1, 0], R[0, 0]); out['T'][i] = ureal[0]; out['f'][i] = f; out['w'][i] = w
        out['eR'][i] = np.linalg.norm(eR); out['sat'][i] = np.any((f <= 1e-9) | (f >= fmax - 1e-9))
        out['tau_cmd'][i] = tau; out['tau_real'][i] = ureal[1:]
        # корпус
        a = ureal[0] / M_Q * R[:, 2] - np.array([0, 0, G])
        dw = Iinv.dot(ureal[1:] - _cross3(w, I.dot(w)))
        v = v + dt * a; p = p + dt * v
        R = R.dot(expm_so3(dt * w)); w = w + dt * dw
    return out


# ====================================================================== самолёт
def turn_rate(V, phi):
    """Модуль угловой скорости координированного разворота |psi'| = (g/V) tg|phi|, рад/с.
    Знак в осях лекции: psi' = -(g/V) tg(phi) (положительный крен — разворот вправо, psi убывает)."""
    return G / V * np.tan(np.abs(phi))


def turn_radius(V, phi):
    """Радиус координированного разворота rho = V^2 / (g tg|phi|)."""
    return V ** 2 / (G * np.tan(np.abs(phi)))


def crab_angle(Va, wind, chi=0.0):
    """Курс psi, при котором путевая скорость направлена по путевому углу chi; возвращает (psi, Vg)."""
    wx, wy = wind
    w_cross = -wx * np.sin(chi) + wy * np.cos(chi)          # составляющая ветра поперёк линии пути
    w_along = wx * np.cos(chi) + wy * np.sin(chi)
    psi = chi + np.arcsin(-w_cross / Va)
    return psi, Va * np.cos(psi - chi) + w_along


def simulate_plane(Va, psi_fun, wind=(0.0, 0.0), T=30.0, dt=0.05, start=(0.0, 0.0)):
    """Кинематическая модель самолёта в горизонтальной плоскости: x' = Va cos(psi) + w_x, y' = Va sin(psi) + w_y.
    psi_fun(t, xy) задаёт курс."""
    N = int(round(T / dt)); xy = np.zeros((N + 1, 2)); xy[0] = start
    for i in range(N):
        psi = psi_fun(i * dt, xy[i])
        xy[i + 1] = xy[i] + dt * (Va * np.array([np.cos(psi), np.sin(psi)]) + np.asarray(wind))
    return xy


def simulate_bank(Va, phi, wind=(0.0, 0.0), T=60.0, dt=0.01, psi0=0.0, start=(0.0, 0.0)):
    """Координированный разворот с постоянным креном phi: psi' = -(g/Va) tg(phi) (оси лекции).
    Возвращает (xy над землёй, xy в воздушной массе, psi)."""
    N = int(round(T / dt)); xy = np.zeros((N + 1, 2)); xa = np.zeros((N + 1, 2)); ps = np.zeros(N + 1)
    xy[0] = xa[0] = start; ps[0] = psi0; w_psi = -G / Va * np.tan(phi)
    for i in range(N):
        va = Va * np.array([np.cos(ps[i]), np.sin(ps[i])])
        xa[i + 1] = xa[i] + dt * va; xy[i + 1] = xy[i] + dt * (va + np.asarray(wind)); ps[i + 1] = ps[i] + dt * w_psi
    return xy, xa, ps


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


# ====================================================================== оценка ориентации
def complementary_pitch(T=60.0, dt=0.01, a=0.98, bias=0.01, gyro_noise=0.002, acc_noise=np.radians(2.0), seed=7,
                        theta_true=None):
    """Одномерный комплементарный фильтр для тангажа (§ 7.8, пример 7.11):
    theta_hat <- a (theta_hat + dt omega_m) + (1 - a) theta_acc, omega_m = omega + bias + шум.
    theta_true(t) — истинный тангаж (по умолчанию 0: аппарат висит). Постоянная времени a dt / (1 - a),
    остаточная ошибка в установившемся режиме a dt bias / (1 - a). Возвращает словарь массивов."""
    rng = np.random.default_rng(seed)
    N = int(round(T / dt)); t = np.arange(N + 1) * dt
    th = np.zeros(N + 1) if theta_true is None else np.array([theta_true(x) for x in t])
    om = np.r_[np.diff(th) / dt, 0.0]
    gyro = om + bias + gyro_noise * rng.standard_normal(N + 1)
    acc = th + acc_noise * rng.standard_normal(N + 1)
    integ = np.zeros(N + 1); filt = np.zeros(N + 1)
    for k in range(N):
        integ[k + 1] = integ[k] + dt * gyro[k]
        filt[k + 1] = a * (filt[k] + dt * gyro[k]) + (1 - a) * acc[k + 1]
    return dict(t=t, true=th, gyro_int=integ, acc=acc, filt=filt, tau=a * dt / (1 - a), resid=a * dt * bias / (1 - a))


def mahony_hover(T=60.0, dt=0.01, bias=(0.0, 0.01, 0.0), kP=1.0, kI=0.1, acc_noise=0.05, seed=7, R0=None):
    """Комплементарный фильтр Махони на SO(3) для висящего аппарата с одним опорным вектором (тяжесть):
    R_hat' = R_hat Lambda(omega_m - b_hat + kP omega_mes), b_hat' = -kI omega_mes, omega_mes = v x R_hat^T v0,
    v — измеренное акселерометром направление «вверх» в связанных осях, v0 = z0.
    Курс по одной тяжести ненаблюдаем, поэтому смещение по оси z не оценивается. Возвращает словарь массивов."""
    rng = np.random.default_rng(seed)
    N = int(round(T / dt)); b = np.asarray(bias, float); z0 = np.array([0, 0, 1.0])
    Rh = np.eye(3) if R0 is None else R0.copy(); bh = np.zeros(3); Rg = np.eye(3)
    err = np.zeros(N); bhist = np.zeros((N, 3)); errg = np.zeros(N)
    for k in range(N):
        wm = b                                           # истинная угловая скорость равна нулю
        v = z0 + acc_noise * rng.standard_normal(3); v = v / np.linalg.norm(v)
        wmes = np.cross(v, Rh.T.dot(z0))
        Rh = Rh.dot(expm_so3(dt * (wm - bh + kP * wmes))); bh = bh - dt * kI * wmes
        Rg = Rg.dot(expm_so3(dt * wm))
        tilt = lambda R: np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        err[k] = tilt(Rh); errg[k] = tilt(Rg); bhist[k] = bh
    return dict(t=np.arange(1, N + 1) * dt, tilt_err=err, tilt_err_gyro=errg, b_hat=bhist)


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
    """Самопроверка: числа примеров 7.1–7.11 конспекта."""
    rad = np.radians
    print('7.1:', enu_to_ned([2, 1, 3]), np.degrees(yaw_enu_to_ned(rad(30))))
    print('7.2:', R321(rad(30), rad(10), rad(-20)).dot([5, 0, -1]).round(2))
    print('7.3:', Wmat(rad(20), rad(60)).dot([0.1, 0.2, 0.3]).round(3), Wmat(rad(20), rad(85)).dot([0.1, 0.2, 0.3]).round(2))
    print('7.5:', mixer_x_inv().dot([M_Q * G, 0.05, 0, 0.01]).round(2))
    u = np.array([M_Q * G, 0.05, 0, 0.15])
    print('7.6: clip', allocate(u)[2].round(3), 'scale %.3f' % allocate_scale_moments(u)[0], 'prio', allocate_priority(u)[0].round(3))
    print('turn: %.1f deg/s, %.1f m' % (np.degrees(turn_rate(20, rad(30))), turn_radius(20, rad(30))))
    psi, Vg = crab_angle(20, (0, 6)); print('7.9: %.1f deg, %.1f m/s' % (np.degrees(psi), Vg))
    for st, go in (((0, 0, 0), (400, 250, np.pi)), ((0, 0, np.pi / 2), (90, 0, -np.pi / 2))):
        print('7.10:', [(w, round(Ln)) for Ln, w, _, _, _ in dubins_all(st, go, 70.6)])
    o = integrate_orientation()
    print('7.4 integration: euler %.2f, R %.2f, e %.2f, exp %.2f, exp mid %.4f deg; stretch %.3f %.3f'
          % (o[:, 2].max(), o[:, 3].max(), o[:, 4].max(), o[:, 7].max(), o[:, 8].max(), o[-1, 9], o[-1, 10]))
    _, _, a, j = circle_derivs(1.0); R, T, _, w = flat_state(a, j, 0.0, 0.0)
    print('7.7: tilt %.2f deg, T/mg %.4f, omega' % (np.degrees(np.arccos(R[2, 2])), T / (M_Q * G)), w.round(4))
    Rs, Ts, _ = thrust_vector([5.0, 0, 0]); print('7.8: exact %.1f deg, %.2f N; linear %.1f deg' % (np.degrees(np.arccos(Rs[2, 2])), Ts, np.degrees(accel_to_tilt([5.0, 0, 0])[0])))
    Mq1 = np.delete(mixer_x(), 0, axis=1); xy6, e6 = hexa_geometry(); xy6b, e6b = hexa_geometry(layout='PPNNPN')
    print('fail: quad margin %.3f; PNPNPN %.3f; PPNNPN' % (abs(hover_feasible(Mq1, M_Q * G, F_MAX, rows=(0, 1, 2))[2]),
                                                       abs(hover_feasible(np.delete(mixer_general(xy6, e6), 0, axis=1), M_Q * G, F_MAX)[2])),
          [round(abs(hover_feasible(np.delete(mixer_general(xy6b, e6b), k, axis=1), M_Q * G, F_MAX)[2]), 2) for k in range(6)])
    c = complementary_pitch(); print('7.11: gyro %.1f deg, tau %.2f s, resid %.2f deg' % (np.degrees(c['gyro_int'][-1]), c['tau'], np.degrees(c['resid'])))


if __name__ == '__main__':
    check()
