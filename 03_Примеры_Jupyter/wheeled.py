# -*- coding: utf-8 -*-
"""wheeled — учебный модуль кинематики колёсных мобильных роботов (занятие 6).

Обозначения — как в конспекте лекции 6 и в 00_Паспорт/Обозначения.md:
  ξ = (x, y, θ)          — поза корпуса (точка P и курс) в неподвижной системе; в коде массив q;
  ξ̇_R = (v_x, v_y, ω)    — скорость корпуса в связанных осях (x вперёд, y влево); ξ̇ = R_z(θ) ξ̇_R;
  φ̇                      — скорости колёс (рад/с), r — радиус колеса;
  b                      — колея дифференциальной схемы и автомобиля (каждое колесо удалено от P на l = b/2);
  ρ, ρ_min               — радиус поворота центра оси (со знаком: ρ > 0 — МЦВ слева) и наименьший радиус;
  l, α, β                — расстояние |PA| и углы установки колеса (раздел 6.3); у омни-платформы l — радиус
                           окружности колёс;
  d, δ                   — база и угол руля; l_x, l_y — полубаза и полуколея меканум-платформы;
  Δt, Δs, Δθ             — шаг интегрирования, путь и поворот корпуса за шаг; e_θ — ошибка курса;
  Σ, F_ξ, F_Δ, k_s       — ковариация позы, якобианы шага одометрии, коэффициент шума путей колёс (м);
  H, F                   — матрицы обратной и прямой задач о скорости: φ̇ = H ξ̇_R, ξ̇_R = F φ̇ (F — не сила);
  l_B                    — вынос точки B вперёд от P (отслеживание траектории);
  D                      — диагональная матрица весов взвешенной псевдообратной (мобильный манипулятор).
Старые имена параметров сохранены там, где их вызывают анимации (anim/eng_xytheta.py): все функции
вызываются позиционно, поэтому переименование l → b в сигнатурах совместимо. DD_L оставлен
псевдонимом колеи DD_B, OMNI_LL — псевдонимом радиуса OMNI_L.

Состав:
  rotz, body_to_world                                   — связь скоростей в двух системах;
  dd_fk, dd_ik, dd_icc, dd_G, dd_saturate, dd_diamond   — дифференциальная схема (примеры 6.1, 6.2);
  step_exact, step_euler, step_mid, integrate, odo_update — шаг одометрии и интегрирование (пример 6.3);
  se2_matrix, se2_from_matrix, se2_twist, se2_exp       — шаг как экспонента на SE(2) (в том числе v_y ≠ 0);
  square_run, umbmark, umbmark_model                    — систематические ошибки и калибровка UMBmark;
  odo_mid_jacobians, odo_cov_update, odo_cov_path,
  odo_monte_carlo, encoder_increments                   — неопределённость одометрии;
  tri_front, tri_rear, ackermann, rear_wheel_speeds,
  bicycle_cm                                            — трицикл, принцип Аккермана, модель центра масс;
  omni_H, omni_H_lines                                  — омни-платформа (пример 6.5), матрица по линиям тяги;
  mec_H, mec_Hpinv, mec_fk, mec_vmax, mec_H_rollers     — меканум-платформа (примеры 6.6, 6.7);
  roll_row, slide_row, swedish_row, wheel_at,
  maneuverability, swerve_ik                            — связи колёс, манёвренность, поворотные модули;
  lie_maneuver, g1, g2, lie_bracket_num, parking_path,
  car_corners                                           — неголономность и параллельная парковка (раздел 6.10);
  flat_outputs, lookahead_control, track_lookahead,
  figure_eight, pure_pursuit_kappa, track_pure_pursuit  — обратная задача на траектории (раздел 6.10);
  planar2r_fk, planar2r_J, mobile_manip_J, weighted_pinv,
  base_twist_J6                                         — мобильный манипулятор (раздел 6.11, пример 6.8);
  simulate_body                                         — интегрирование движения корпуса по команде ξ̇_R(t);
  draw_diffdrive, draw_mecanum, draw_omni3, draw_car, draw_path, draw_cov_ellipse — рисование;
  check                                                 — самопроверка (печать чисел примеров 6.1–6.8).
Только numpy и matplotlib.
"""
import numpy as np

NAVY, AMB, TEAL, RED, GREY, ICE, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#CADCFC', '#2C7A5A'
BG = '#F3F6FB'

# ---- параметры примеров конспекта, метры
DD_R, DD_B = 0.05, 0.30                 # дифференциальная схема: радиус колеса r, колея b
DD_L = DD_B                             # прежнее имя колеи (псевдоним)
OMNI_R, OMNI_L = 0.05, 0.20             # омни-платформа: радиус колеса r, радиус окружности колёс l
OMNI_LL = OMNI_L                        # прежнее имя (псевдоним)
OMNI_ALPHA = np.radians([0.0, 120.0, 240.0])
MEC_R, MEC_LX, MEC_LY = 0.05, 0.235, 0.15   # меканум-платформа (учебные размеры, близкие к youBot)
ACK_D, ACK_B = 2.7, 1.6                 # автомобиль примера 6.4: база d, колея b
ACK_L = ACK_B                           # прежнее имя колеи автомобиля (псевдоним)


# ====================================================================== системы координат
def rotz(th):
    """Матрица R_z(θ): ξ̇ = R_z(θ) ξ̇_R."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_world(u, th):
    """ξ̇ = R_z(θ) ξ̇_R."""
    return rotz(th).dot(np.asarray(u, float))


# ====================================================================== дифференциальная схема
def dd_fk(phr, phl, r=DD_R, b=DD_B):
    """Прямая задача о скорости: (φ̇_r, φ̇_l), рад/с -> (v, ω); v = (V_r + V_l)/2, ω = (V_r − V_l)/b.
    Порядок входов курса: сначала ПРАВОЕ колесо (у roboticstoolbox.DiffSteer — сначала левое)."""
    Vr, Vl = r * phr, r * phl
    return (Vr + Vl) / 2.0, (Vr - Vl) / b


def dd_ik(v, w, r=DD_R, b=DD_B):
    """Обратная задача о скорости: (v, ω) -> (φ̇_r, φ̇_l) = ((v + bω/2)/r, (v − bω/2)/r)."""
    return (v + b / 2.0 * w) / r, (v - b / 2.0 * w) / r


def dd_icc(q, v, w):
    """Радиус поворота ρ = v/ω (со знаком, ρ > 0 — МЦВ слева) и координаты МЦВ
    x_c = x − ρ sin θ, y_c = y + ρ cos θ; при ω = 0 — (inf, None)."""
    if abs(w) < 1e-12:
        return np.inf, None
    R = v / w
    return R, np.array([q[0] - R * np.sin(q[2]), q[1] + R * np.cos(q[2])])


def dd_G(th, r=DD_R, b=DD_B):
    """Матрица G(θ) = R_z(θ) F размера 3×2: ξ̇ = G (φ̇_r, φ̇_l); F = [[r/2, r/2], [0, 0], [r/b, −r/b]]."""
    F = np.array([[r / 2, r / 2], [0.0, 0.0], [r / b, -r / b]])
    return rotz(th).dot(F)


def dd_saturate(phr, phl, pmax, mode='scale'):
    """Ограничение скоростей колёс: 'clip' — срезание каждого колеса, 'scale' — общий коэффициент
    (сохраняет кривизну 1/ρ)."""
    ph = np.array([phr, phl], float)
    if mode == 'clip':
        return np.clip(ph, -pmax, pmax)
    k = min(1.0, pmax / np.abs(ph).max())
    return ph * k


def dd_diamond(pmax=20.0, r=DD_R, b=DD_B):
    """Вершины ромба |v| + (b/2)|ω| ≤ r φ̇_max: (v_max, ω_max) = (r φ̇_max, 2 r φ̇_max / b)."""
    return r * pmax, 2.0 * r * pmax / b


# ====================================================================== шаг одометрии
def step_exact(q, v, w, dt):
    """Точный шаг по дуге (поворот вокруг МЦВ); при ω → 0 — прямолинейный шаг."""
    x, y, th = q
    if abs(w) < 1e-9:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th])
    return np.array([x + v / w * (np.sin(th + w * dt) - np.sin(th)),
                     y - v / w * (np.cos(th + w * dt) - np.cos(th)), th + w * dt])


def step_euler(q, v, w, dt):
    """Метод Эйлера: весь шаг с курсом начала шага."""
    x, y, th = q
    return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th + w * dt])


def step_mid(q, v, w, dt):
    """Метод средней точки: курс середины шага θ + Δθ/2."""
    x, y, th = q
    return np.array([x + v * dt * np.cos(th + w * dt / 2), y + v * dt * np.sin(th + w * dt / 2), th + w * dt])


STEPS = {'euler': step_euler, 'mid': step_mid, 'exact': step_exact}


def integrate(q0, v, w, dt, n, method='exact'):
    """n шагов интегрирования при постоянных v, ω. Возвращает массив (n+1)×3."""
    f = STEPS[method]; q = np.array(q0, float); P = [q.copy()]
    for i in range(n):
        q = f(q, v, w, dt); P.append(q.copy())
    return np.array(P)


def odo_update(q, dphr, dphl, r=DD_R, b=DD_B, method='exact'):
    """Один такт одометрии по приращениям углов колёс (рад): Δs = (Δs_r + Δs_l)/2, Δθ = (Δs_r − Δs_l)/b."""
    dsr, dsl = r * dphr, r * dphl
    return STEPS[method](q, (dsr + dsl) / 2.0, (dsr - dsl) / b, 1.0)


# ====================================================================== SE(2)
def se2_matrix(q):
    """Поза ξ = (x, y, θ) -> однородная матрица T 3×3."""
    c, s = np.cos(q[2]), np.sin(q[2])
    return np.array([[c, -s, q[0]], [s, c, q[1]], [0.0, 0.0, 1.0]])


def se2_from_matrix(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def se2_twist(vx, vy, w):
    """Матрица X со строками (0, −ω, v_x), (ω, 0, v_y), (0, 0, 0): T' = T exp(Δt X)."""
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]])


def se2_exp(q, vx, vy, w, dt):
    """Шаг одометрии как экспонента на SE(2) в конечном виде (конспект, раздел 6.5):
    p' = p + R(θ) (Δt/Δθ) [[sin Δθ, −(1 − cos Δθ)], [1 − cos Δθ, sin Δθ]] (v_x, v_y), θ' = θ + Δθ."""
    x, y, th = q; dth = w * dt
    if abs(dth) < 1e-9:
        M = dt * np.eye(2)
    else:
        M = dt / dth * np.array([[np.sin(dth), -(1 - np.cos(dth))], [1 - np.cos(dth), np.sin(dth)]])
    c, s = np.cos(th), np.sin(th)
    d = np.array([[c, -s], [s, c]]).dot(M).dot([vx, vy])
    return np.array([x + d[0], y + d[1], th + dth])


# ====================================================================== систематические ошибки, UMBmark
def square_run(rr, rl, base, r_nom=0.05, b_nom=0.30, side=4.0, ccw=True, n=200, rr_nom=None, rl_nom=None):
    """Робот отрабатывает квадрат side×side по своей модели; реальные радиусы rr, rl и колея base.
    Модель контроллера: радиусы rr_nom, rl_nom (по умолчанию оба r_nom) и колея b_nom.
    Одометрия по той же модели считает, что робот вернулся в начало. Возвращает истинный путь (4·2n+1)×3."""
    rrn = r_nom if rr_nom is None else rr_nom
    rln = r_nom if rl_nom is None else rl_nom
    q = np.array([0.0, 0.0, 0.0]); path = [q.copy()]
    sgn = 1.0 if ccw else -1.0
    for k in range(4):
        dphr, dphl = side / rrn / n, side / rln / n                       # прямая: равные пути колёс
        for i in range(n):
            vr, vl = rr * dphr, rl * dphl
            q = step_exact(q, (vr + vl) / 2, (vr - vl) / base, 1.0); path.append(q.copy())
        dphr = sgn * (np.pi / 2) * (b_nom / 2) / rrn / n                  # поворот на месте на 90°
        dphl = -sgn * (np.pi / 2) * (b_nom / 2) / rln / n
        for i in range(n):
            vr, vl = rr * dphr, rl * dphl
            q = step_exact(q, (vr + vl) / 2, (vr - vl) / base, 1.0); path.append(q.copy())
    return np.array(path)


def umbmark(end_cw, end_ccw, side=4.0, b=0.30):
    """Формулы UMBmark (Borenstein, Feng, 1996) в осях маршрута square_run (старт в начале координат
    курсом вдоль X; обход против часовой стрелки идёт в y > 0). По абсциссам конечных точек двух обходов:
      α = (x_cw + x_ccw)/(−4L) — недоворот на каждом повороте из-за ошибки колеи (рад);
      β = (x_cw − x_ccw)/(−4L) — поворот курса на каждой прямой из-за разных диаметров (рад);
      ρ_β = (L/2)/sin(β/2), E_d = D_R/D_L = (ρ_β + b/2)/(ρ_β − b/2), E_b = (π/2)/(π/2 − α).
    Формулы линейны по ошибкам, поэтому при больших ошибках тест повторяют с поправками (итерации).
    Возвращает словарь alpha, beta, E_d, E_b."""
    xw, xc = end_cw[0], end_ccw[0]; L = side
    a = (xw + xc) / (-4.0 * L); be = (xw - xc) / (-4.0 * L)
    if abs(be) > 1e-12:
        R = (L / 2.0) / np.sin(be / 2.0); Ed = (R + b / 2.0) / (R - b / 2.0)
    else:
        Ed = 1.0
    return {'alpha': a, 'beta': be, 'E_d': Ed, 'E_b': (np.pi / 2) / (np.pi / 2 - a)}


def umbmark_model(E_d, E_b, r_nom=0.05, b_nom=0.30):
    """Поправленная модель контроллера: радиусы r_R = c_R r, r_L = c_L r при c_L = 2/(E_d + 1),
    c_R = 2/(1/E_d + 1) (средний радиус не меняется) и колея b = E_b b_nom. Возвращает (r_R, r_L, b)."""
    cL = 2.0 / (E_d + 1.0); cR = 2.0 / (1.0 / E_d + 1.0)
    return r_nom * cR, r_nom * cL, b_nom * E_b


# ====================================================================== неопределённость одометрии
def odo_mid_jacobians(q, dsr, dsl, b=DD_B):
    """Якобианы шага средней точки ξ' = f(ξ, Δs_r, Δs_l): F_ξ (3×3) и F_Δ (3×2), приложение 6.Б."""
    ds = (dsr + dsl) / 2.0; dth = (dsr - dsl) / b; tm = q[2] + dth / 2.0
    c, s = np.cos(tm), np.sin(tm)
    Fx = np.array([[1.0, 0.0, -ds * s], [0.0, 1.0, ds * c], [0.0, 0.0, 1.0]])
    k = ds / (2.0 * b)
    Fd = np.array([[0.5 * c - k * s, 0.5 * c + k * s],
                   [0.5 * s + k * c, 0.5 * s - k * c],
                   [1.0 / b, -1.0 / b]])
    return Fx, Fd


def odo_cov_update(q, Sigma, dsr, dsl, b=DD_B, ks=1e-4):
    """Шаг одометрии (средняя точка) с переносом ковариации:
    Σ' = F_ξ Σ F_ξᵀ + F_Δ Σ_Δ F_Δᵀ, Σ_Δ = diag(k_s|Δs_r|, k_s|Δs_l|). Возвращает (ξ', Σ')."""
    Fx, Fd = odo_mid_jacobians(q, dsr, dsl, b)
    Sd = np.diag([ks * abs(dsr), ks * abs(dsl)])
    ds = (dsr + dsl) / 2.0; dth = (dsr - dsl) / b
    return step_mid(q, ds, dth, 1.0), Fx.dot(Sigma).dot(Fx.T) + Fd.dot(Sd).dot(Fd.T)


def odo_cov_path(dsr, dsl, b=DD_B, ks=1e-4, q0=(0.0, 0.0, 0.0), Sigma0=None):
    """Рекурсия ковариации вдоль последовательности путей колёс; возвращает (Q (n+1)×3, S (n+1)×3×3)."""
    q = np.array(q0, float); S = np.zeros((3, 3)) if Sigma0 is None else np.array(Sigma0, float)
    Q, SS = [q.copy()], [S.copy()]
    for a, c in zip(dsr, dsl):
        q, S = odo_cov_update(q, S, a, c, b, ks); Q.append(q.copy()); SS.append(S.copy())
    return np.array(Q), np.array(SS)


def encoder_increments(s_cum, r=DD_R, ticks=1024):
    """Целочисленный энкодер: по накопленному истинному пути колеса s_cum (массив, м) возвращает
    измеренные приращения пути за такт (м). Счётчик накопительный, поэтому ошибка квантования
    ограничена одной меткой и не накапливается."""
    counts = np.floor(np.asarray(s_cum) / r * ticks / (2 * np.pi))
    return np.diff(counts, axis=-1) * 2 * np.pi * r / ticks


def odo_monte_carlo(dsr, dsl, b=DD_B, ks=1e-4, n_runs=2000, seed=0, ticks=None, r=DD_R):
    """Монте-Карло для одометрии по методу средней точки.
    dsr, dsl — пути колёс за такт по их вращению (то, что видит энкодер). Истинные перемещения точек контакта
    по полу отличаются от них на независимые ошибки N(0, k_s|Δs|) (проскальзывание, неровности, пятно контакта).
    Возвращает ошибки позы «истина − оценка» в конце каждого такта, массив (n_runs, n+1, 3).
    Без ticks оценка интегрирует пути dsr, dsl точно; с ticks (меток на оборот) — приращения целочисленного
    накопительного счётчика энкодера, опрашиваемого на каждом такте."""
    rng = np.random.default_rng(seed)
    dsr = np.asarray(dsr, float); dsl = np.asarray(dsl, float); n = len(dsr)
    tr = dsr + rng.standard_normal((n_runs, n)) * np.sqrt(ks * np.abs(dsr))
    tl = dsl + rng.standard_normal((n_runs, n)) * np.sqrt(ks * np.abs(dsl))
    if ticks is None:
        mr, ml = dsr, dsl
    else:
        mr = encoder_increments(np.concatenate([[0.0], np.cumsum(dsr)]), r, ticks)
        ml = encoder_increments(np.concatenate([[0.0], np.cumsum(dsl)]), r, ticks)
    mr = np.broadcast_to(mr, (n_runs, n)); ml = np.broadcast_to(ml, (n_runs, n))

    def run(a, c):
        q = np.zeros((n_runs, 3)); out = np.zeros((n_runs, n + 1, 3))
        for k in range(n):
            ds = (a[:, k] + c[:, k]) / 2.0; dth = (a[:, k] - c[:, k]) / b; tm = q[:, 2] + dth / 2.0
            q = q + np.column_stack([ds * np.cos(tm), ds * np.sin(tm), dth]); out[:, k + 1] = q
        return out
    return run(tr, tl) - run(mr, ml)


# ====================================================================== трицикл и Аккерман
def tri_front(vs, delta, d):
    """Трицикл с приводным рулевым колесом: (v_s, δ) -> (v, ω) = (v_s cos δ, v_s sin δ / d)."""
    return vs * np.cos(delta), vs * np.sin(delta) / d


def tri_rear(v, delta, d):
    """Трицикл с задним приводом (велосипедная модель): (v, δ) -> (v, ω) = (v, v tg δ / d)."""
    return v, v * np.tan(delta) / d


def ackermann(R, d=ACK_D, b=ACK_B):
    """Углы внутреннего и наружного рулевых колёс и эквивалентного трицикла (рад) при радиусе ρ = R:
    tg δ_i = d/(ρ − b/2), tg δ_o = d/(ρ + b/2), tg δ = d/ρ."""
    return np.arctan(d / (R - b / 2.0)), np.arctan(d / (R + b / 2.0)), np.arctan(d / R)


def rear_wheel_speeds(v, R, b):
    """Линейные скорости внутреннего и наружного задних колёс V = ω(ρ ∓ b/2) при ω = v/ρ."""
    w = v / R
    return w * (R - b / 2.0), w * (R + b / 2.0)


def bicycle_cm(v, delta, d_f, d_r):
    """Велосипедная модель центра масс (приложение 6.Д): tg β_c = d_r tg δ/d; возвращает (β_c, ω, ρ_C)
    при скорости центра масс v: ω = v cos β_c tg δ / d, ρ_C = v/ω."""
    d = d_f + d_r; bc = np.arctan(d_r * np.tan(delta) / d); w = v * np.cos(bc) * np.tan(delta) / d
    return bc, w, v / w


# ====================================================================== омни-платформа
def omni_H(r=OMNI_R, l=OMNI_L, alphas=OMNI_ALPHA):
    """Матрица обратной задачи о скорости: φ̇ = H ξ̇_R, строка i = (−sin α_i, cos α_i, l)/r."""
    a = np.asarray(alphas, float)
    return np.column_stack([-np.sin(a), np.cos(a), l * np.ones(len(a))]) / r


def omni_H_lines(points, thrusts, r=OMNI_R):
    """Матрица H по линиям тяги: строка i = (t_x, t_y, p_x t_y − p_y t_x)/r — направление тяги и его момент
    относительно P (плюккеровы координаты линии тяги). points, thrusts — списки (x, y) центров и ортов тяги."""
    rows = [(t[0], t[1], p[0] * t[1] - p[1] * t[0]) for p, t in zip(points, thrusts)]
    return np.array(rows, float) / r


# ====================================================================== меканум-платформа
def mec_H(r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Матрица 4×3 обратной задачи о скорости; колёса: 1 — ПЛ, 2 — ПП, 3 — ЗЛ, 4 — ЗП."""
    k = lx + ly
    return np.array([[1, -1, -k], [1, 1, k], [1, 1, -k], [1, -1, k]], float) / r


def mec_H_rollers(gammas=np.radians([-45.0, 45.0, 45.0, -45.0]), r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """H из геометрии роликов: r φ̇_i = v_ix + v_iy tg γ_i при v_ix = v_x − ω y_i, v_iy = v_y + ω x_i,
    то есть строка (1, tg γ_i, x_i tg γ_i − y_i)/r. По умолчанию — правильная установка «X сверху»
    (γ₁ = γ₄ = −45°, γ₂ = γ₃ = +45°); со сменой знаков γ получается установка «X снизу»."""
    pts = ((lx, ly), (lx, -ly), (-lx, ly), (-lx, -ly))
    rows = [(1.0, np.tan(g), x * np.tan(g) - y) for g, (x, y) in zip(gammas, pts)]
    return np.array(rows) / r


def mec_Hpinv(r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Псевдообратная F = H⁺ = (HᵀH)⁻¹Hᵀ в явном виде."""
    k = lx + ly
    return r / 4.0 * np.array([[1, 1, 1, 1], [-1, 1, 1, -1], [-1 / k, 1 / k, -1 / k, 1 / k]], float)


def mec_fk(phi, r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Прямая задача о скорости по наименьшим квадратам.
    Возвращает (ξ̇_R, невязка φ̇ − H ξ̇_R, левая часть условия совместности φ̇₁ + φ̇₂ − φ̇₃ − φ̇₄)."""
    phi = np.asarray(phi, float)
    u = mec_Hpinv(r, lx, ly).dot(phi)
    return u, phi - mec_H(r, lx, ly).dot(u), phi[0] + phi[1] - phi[2] - phi[3]


def mec_vmax(direction, omega=0.0, pmax=20.0, r=MEC_R, lx=MEC_LX, ly=MEC_LY):
    """Наибольшая скорость в направлении direction (рад) при заданной ω: граница октаэдра
    |v_x| + |v_y| + (l_x + l_y)|ω| ≤ r φ̇_max."""
    rest = r * pmax - (lx + ly) * abs(omega)
    if rest <= 0:
        return 0.0
    return rest / (abs(np.cos(direction)) + abs(np.sin(direction)))


# ====================================================================== связи колёс и манёвренность
def roll_row(alpha, beta, l):
    """Строка условия качения фиксированного (рулевого) колеса: row · ξ̇_R = r φ̇."""
    return np.array([np.sin(alpha + beta), -np.cos(alpha + beta), -l * np.cos(beta)])


def slide_row(alpha, beta, l):
    """Строка условия отсутствия бокового скольжения: row · ξ̇_R = 0 (скорость центра колеса вдоль его оси)."""
    return np.array([np.cos(alpha + beta), np.sin(alpha + beta), l * np.sin(beta)])


def swedish_row(alpha, beta, gamma, l):
    """Строка связи шведского колеса: row · ξ̇_R = r φ̇ cos γ."""
    return np.array([np.sin(alpha + beta + gamma), -np.cos(alpha + beta + gamma), -l * np.cos(beta + gamma)])


def wheel_at(x, y, heading=0.0):
    """Параметры (α, β, l) колеса с центром (x, y) в связанных осях, плоскость которого (направление качения)
    повёрнута на угол heading от оси X_R: α = atan2(y, x), l = √(x² + y²), β = π/2 + heading − α."""
    a = np.arctan2(y, x)
    return a, np.pi / 2 + heading - a, np.hypot(x, y)


def maneuverability(fixed=(), steer=()):
    """fixed, steer — списки колёс (α, β, l). Возвращает C₁, rank C₁, δ_m, δ_s, δ_M."""
    rf = [slide_row(*w) for w in fixed]; rs = [slide_row(*w) for w in steer]
    C1 = np.array(rf + rs).reshape(-1, 3)
    rank = int(np.linalg.matrix_rank(C1, tol=1e-9)) if len(C1) else 0
    ds = int(np.linalg.matrix_rank(np.array(rs), tol=1e-9)) if rs else 0
    dm = 3 - rank
    return C1, rank, dm, ds, dm + ds


def swerve_ik(u, modules, r=0.05):
    """Поворотные приводные модули (приложение 6.Д): для ξ̇_R = (v_x, v_y, ω) и модулей в точках (x_i, y_i)
    v_i = (v_x − ω y_i, v_y + ω x_i), β_i = atan2(v_iy, v_ix), φ̇_i = |v_i|/r.
    Возвращает массивы (β_i, рад; |v_i|, м/с; φ̇_i, рад/с)."""
    vx, vy, w = u; M = np.asarray(modules, float)
    V = np.column_stack([vx - w * M[:, 1], vy + w * M[:, 0]])
    sp = np.hypot(V[:, 0], V[:, 1])
    return np.arctan2(V[:, 1], V[:, 0]), sp, sp / r


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
    """Поле «ехать вперёд» g₁ = (cos θ, sin θ, 0)."""
    return np.array([np.cos(q[2]), np.sin(q[2]), 0.0])


def g2(q):
    """Поле «поворачиваться» g₂ = (0, 0, 1)."""
    return np.array([0.0, 0.0, 1.0])


def lie_bracket_num(f, g, q, h=1e-6):
    """Скобка Ли [f, g] = (∂g/∂ξ) f − (∂f/∂ξ) g численным дифференцированием."""
    q = np.asarray(q, float); n = len(q)
    Df = np.zeros((n, n)); Dg = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        Df[:, j] = (f(q + e) - f(q - e)) / (2 * h); Dg[:, j] = (g(q + e) - g(q - e)) / (2 * h)
    return Dg.dot(f(q)) - Df.dot(g(q))


def parking_path(Rm=4.0, psi=np.radians(20.0), ncyc=4, n=40):
    """Параллельная парковка велосипедной модели (центр задней оси): задний ход по дуге радиуса ρ_min = Rm
    с рулём вправо до упора, пока курс не изменится на центральный угол ψ, затем так же с рулём влево,
    возврат вперёд по прямой на 2ρ_min sin ψ. Возвращает (траектория, узловые положения)."""
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


def car_corners(q, Lf=3.4, Lr=0.8, W=1.7):
    """Четыре угла кузова (свес назад Lr от задней оси, вперёд Lf, ширина W; длина Lf + Lr) в мировых осях."""
    c, s = np.cos(q[2]), np.sin(q[2])
    pts = np.array([[Lf, W / 2], [-Lr, W / 2], [-Lr, -W / 2], [Lf, -W / 2]])
    return np.column_stack([q[0] + c * pts[:, 0] - s * pts[:, 1], q[1] + s * pts[:, 0] + c * pts[:, 1]])


# ====================================================================== обратная задача на траектории
def flat_outputs(xd, yd, xdd, ydd):
    """Плоские выходы уницикла: по (ẋ, ẏ, ẍ, ÿ) возвращает (v, θ, ω, κ); v = √(ẋ² + ẏ²),
    θ = atan2(ẏ, ẋ), ω = (ẋÿ − ẏẍ)/(ẋ² + ẏ²) = vκ."""
    v2 = xd ** 2 + yd ** 2; v = np.sqrt(v2); w = (xd * ydd - yd * xdd) / v2
    return v, np.arctan2(yd, xd), w, w / v


def figure_eight(t, A=1.0, T=20.0):
    """Восьмёрка (лемниската Жероно) p(t) = (A sin Ωt, (A/2) sin 2Ωt), Ω = 2π/T; возвращает (p, ṗ)."""
    W = 2 * np.pi / T
    p = np.array([A * np.sin(W * t), A / 2 * np.sin(2 * W * t)])
    pd = np.array([A * W * np.cos(W * t), A * W * np.cos(2 * W * t)])
    return p, pd


def lookahead_control(q, pB_ref, pB_dot_ref, lB=0.2, K=1.0):
    """Управление по вынесенной точке B = P + l_B (cos θ, sin θ):
    (v, ω) = J_B⁻¹(θ) (ṗ_B* + K (p_B* − p_B)), J_B = [[cos θ, −l_B sin θ], [sin θ, l_B cos θ]], det J_B = l_B."""
    c, s = np.cos(q[2]), np.sin(q[2])
    pB = np.array([q[0] + lB * c, q[1] + lB * s])
    rhs = np.asarray(pB_dot_ref, float) + K * (np.asarray(pB_ref, float) - pB)
    JB = np.array([[c, -lB * s], [s, lB * c]])
    v, w = np.linalg.solve(JB, rhs)
    return v, w


def track_lookahead(ref, q0, lB=0.2, K=1.0, T=20.0, dt=0.01):
    """Моделирование: уницикл ведёт точку B по опорной траектории ref(t) -> (p*, ṗ*), точный шаг по дуге.
    Возвращает t, Q (поза), PB (точка B), U (v, ω)."""
    n = int(round(T / dt)); q = np.array(q0, float)
    ts = np.arange(n + 1) * dt; Q = [q.copy()]; PB = []; U = []
    for k in range(n + 1):
        p, pd = ref(ts[k]); v, w = lookahead_control(q, p, pd, lB, K)
        PB.append([q[0] + lB * np.cos(q[2]), q[1] + lB * np.sin(q[2])]); U.append([v, w])
        if k < n:
            q = step_exact(q, v, w, dt); Q.append(q.copy())
    return ts, np.array(Q), np.array(PB), np.array(U)


def pure_pursuit_kappa(alpha, Ld):
    """Кривизна дуги преследования точки упреждения: κ = 2 sin η / L_d.

    Аргумент alpha — угол η между курсом и направлением на точку упреждения
    (в конспекте η, чтобы не путать с углом установки колеса α; имя аргумента
    сохранено ради совместимости)."""
    return 2.0 * np.sin(alpha) / Ld


def track_pure_pursuit(path, q0, v=0.3, Ld=0.5, T=20.0, dt=0.01):
    """Pure pursuit по ломаной path (N×2): точка упреждения — первая точка пути впереди на расстоянии ≥ L_d
    от робота; ω = v κ. Возвращает Q (поза) и W (ω)."""
    path = np.asarray(path, float); q = np.array(q0, float); i0 = 0; Q = [q.copy()]; Ws = []
    for k in range(int(round(T / dt))):
        d = np.hypot(path[i0:, 0] - q[0], path[i0:, 1] - q[1])
        i0 += int(np.argmin(d[:max(1, min(len(d), 200))]))
        j = i0 + np.argmax(np.hypot(path[i0:, 0] - q[0], path[i0:, 1] - q[1]) >= Ld)
        a = np.arctan2(path[j, 1] - q[1], path[j, 0] - q[0]) - q[2]
        w = v * pure_pursuit_kappa(np.arctan2(np.sin(a), np.cos(a)), Ld)
        q = step_exact(q, v, w, dt); Q.append(q.copy()); Ws.append(w)
    return np.array(Q), np.array(Ws)


# ====================================================================== мобильный манипулятор
def planar2r_fk(q, a=(0.3, 0.3)):
    """Плоский двухзвенник на платформе: положение схвата r_Pe в связанных осях."""
    return np.array([a[0] * np.cos(q[0]) + a[1] * np.cos(q[0] + q[1]), a[0] * np.sin(q[0]) + a[1] * np.sin(q[0] + q[1])])


def planar2r_J(q, a=(0.3, 0.3)):
    """Якобиан 2×2 плоского двухзвенника (скорость схвата относительно платформы, связанные оси)."""
    s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-a[0] * s1 - a[1] * s12, -a[1] * s12], [a[0] * c1 + a[1] * c12, a[1] * c12]])


def mobile_manip_J(r_Pe, J_arm):
    """Общая матрица Якоби плоской части мобильного манипулятора в связанных осях:
    J = [J_base | J_arm], J_base = [[1, 0, −r_Pe,y], [0, 1, r_Pe,x]]; ṗ_e = R(θ) J (ξ̇_R; q̇)."""
    Jb = np.array([[1.0, 0.0, -r_Pe[1]], [0.0, 1.0, r_Pe[0]]])
    return np.hstack([Jb, np.asarray(J_arm, float)])


def weighted_pinv(J, D):
    """Взвешенная псевдообратная J_D⁺ = D⁻¹Jᵀ(J D⁻¹ Jᵀ)⁻¹ (минимум zᵀDz при J z = ṗ)."""
    Di = np.linalg.inv(np.atleast_2d(np.diag(D) if np.ndim(D) == 1 else D))
    return Di.dot(J.T).dot(np.linalg.inv(J.dot(Di).dot(J.T)))


def base_twist_J6(p_e):
    """Вклад плоской скорости платформы ξ̇_R = (v_x, v_y, ω) в скорость схвата [v; ω] (порядок kinlib),
    p_e — положение схвата в осях платформы: v = (v_x − ω p_y, v_y + ω p_x, 0), ω_e = (0, 0, ω)."""
    J = np.zeros((6, 3)); J[0, 0] = 1.0; J[1, 1] = 1.0; J[0, 2] = -p_e[1]; J[1, 2] = p_e[0]; J[5, 2] = 1.0
    return J


# ====================================================================== движение корпуса по команде
def simulate_body(u_fun, q0=(0.0, 0.0, 0.0), T=5.0, n=500):
    """Интегрирование ξ̇ = R_z(θ) ξ̇_R(t) методом средней точки. u_fun(t) -> (v_x, v_y, ω)."""
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


def draw_diffdrive(ax, q, b=DD_B, color=NAVY, alpha=1.0, scale=1.0):
    """Робот с дифференциальной схемой: круглый корпус диаметра b·scale, два колеса, метка носа."""
    a = b * scale / 2.0; t = np.linspace(0, 2 * np.pi, 60)
    ax.fill(q[0] + a * np.cos(t), q[1] + a * np.sin(t), facecolor=BG, edgecolor=color, lw=1.2, alpha=alpha, zorder=3)
    for sy in (1, -1):
        _wheel(ax, q, 0.0, sy * a, 0.8 * a, 0.25 * a, fc=color, ec=color, alpha=alpha)
    n = _b2w(q, a, 0.0); ax.plot([q[0], n[0]], [q[1], n[1]], color=color, lw=1.5, alpha=alpha, zorder=5)


def draw_mecanum(ax, q, lx=MEC_LX, ly=MEC_LY, color=NAVY, alpha=1.0, scale=1.0, top=(1, -1, -1, 1)):
    """Меканум-платформа, вид сверху: штриховка — верхние ролики (по умолчанию рисунок X), треугольник — нос.
    top=(-1, 1, 1, -1) рисует неправильную установку (X образуют нижние ролики)."""
    lx, ly = lx * scale, ly * scale; wl, ww = 0.75 * lx, 0.42 * ly
    bl, bw = lx + 0.35 * wl, ly - ww / 2 - 0.05 * ly
    _poly(ax, q, [(bl, bw), (-bl, bw), (-bl, -bw), (bl, -bw)], BG, color, alpha=alpha)
    _poly(ax, q, [(bl, 0.0), (bl - 0.3 * lx, 0.3 * bw), (bl - 0.3 * lx, -0.3 * bw)], color, color, alpha=alpha, z=4)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        _wheel(ax, q, sx * lx, sy * ly, wl, ww, fc=ICE, ec=color, alpha=alpha)
        for s in np.linspace(-wl / 2 + ww / 2, wl / 2 - ww / 2, 3):
            p0 = _b2w(q, sx * lx + s - top[i] * 0.4 * ww, sy * ly - 0.4 * ww)
            p1 = _b2w(q, sx * lx + s + top[i] * 0.4 * ww, sy * ly + 0.4 * ww)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.4, alpha=alpha, zorder=6)


def draw_omni3(ax, q, l=OMNI_L, color=NAVY, alpha=1.0, scale=1.0, wheel_ang=None):
    """Трёхколёсная омни-платформа: круглый корпус, три колеса (по умолчанию плоскость колеса по касательной),
    метка носа. wheel_ang — углы плоскостей колёс в связанных осях (для опыта «тяга по радиусам»)."""
    a = l * scale; t = np.linspace(0, 2 * np.pi, 60)
    ax.fill(q[0] + a * np.cos(t), q[1] + a * np.sin(t), facecolor=BG, edgecolor=color, lw=1.2, alpha=alpha, zorder=3)
    for k, al in enumerate(OMNI_ALPHA):
        ang = al + np.pi / 2 if wheel_ang is None else wheel_ang[k]
        _wheel(ax, q, a * np.cos(al), a * np.sin(al), 0.55 * a, 0.2 * a, ang=ang, fc=(AMB if k == 0 else color), ec=color, alpha=alpha)
    n = _b2w(q, 0.7 * a, 0.0); ax.plot([q[0], n[0]], [q[1], n[1]], color=color, lw=1.5, alpha=alpha, zorder=5)


def draw_car(ax, q, d=ACK_D, b=ACK_B, di=0.0, do=0.0, color=NAVY, fc=ICE, alpha=1.0, fill=True, wheels=True):
    """Автомобиль: точка q — центр задней оси; di, do — углы левого и правого передних колёс."""
    ov = 0.18 * d
    pts = [(d + 2 * ov, b / 2 + 0.05), (-2 * ov, b / 2 + 0.05), (-2 * ov, -b / 2 - 0.05), (d + 2 * ov, -b / 2 - 0.05)]
    if fill:
        _poly(ax, q, pts, fc, color, alpha=alpha, z=2)
    else:
        P = np.array([_b2w(q, bx, by) for bx, by in pts + pts[:1]]); ax.plot(P[:, 0], P[:, 1], color=color, lw=0.9, alpha=alpha, zorder=2)
    if not wheels:
        return
    wl, ww = 0.24 * d, 0.08 * d
    for sy in (1, -1):
        _wheel(ax, q, 0.0, sy * b / 2, wl, ww, fc=color, ec=color, alpha=alpha)
    _wheel(ax, q, d, b / 2, wl, ww, ang=di, fc=AMB, ec=AMB, alpha=alpha)
    _wheel(ax, q, d, -b / 2, wl, ww, ang=do, fc=AMB, ec=AMB, alpha=alpha)


def draw_cov_ellipse(ax, mean, S2, nsig=3.0, color=NAVY, lw=1.8, ls='-', label=None):
    """Эллипс nsig·σ по ковариации 2×2 положения."""
    w, V = np.linalg.eigh(np.asarray(S2, float)); t = np.linspace(0, 2 * np.pi, 200)
    E = V.dot(np.diag(nsig * np.sqrt(np.maximum(w, 0.0)))).dot(np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(mean[0] + E[0], mean[1] + E[1], color=color, lw=lw, ls=ls, label=label, zorder=6)


# ====================================================================== самопроверка
def check():
    v, w = dd_fk(12.0, 8.0); print('6.1: v = %.3f, w = %.3f, rho = %.3f' % (v, w, v / w))
    pr, pl = dd_ik(0.8, 2.0); print('6.2: phi = %.1f, %.1f' % (pr, pl))
    for m in ('clip', 'scale'):
        s = dd_saturate(pr, pl, 20.0, m); v, w = dd_fk(s[0], s[1]); print('     %s: v = %.3f, w = %.3f, rho = %.3f' % (m, v, w, v / w))
    n = 13; qt = step_exact(np.zeros(3), 0.5, 0.5, n * 0.5)
    print('6.3:', ['%s %.4f' % (m, np.hypot(*(integrate(np.zeros(3), 0.5, 0.5, 0.5, n, m)[-1, :2] - qt[:2]))) for m in ('euler', 'mid', 'exact')])
    di, do, de = ackermann(6.0); print('6.4: %.1f %.1f %.1f, ratio %.2f' % (np.degrees(di), np.degrees(do), np.degrees(de), 6.8 / 5.2))
    print('6.5:', omni_H().dot([0.2, 0.1, 0.5]).round(2))
    print('6.6:', mec_H().dot([0.3, 0.2, 0.5]).round(2))
    u, res, c = mec_fk([10.0, 10.4, 9.6, 10.2]); print('6.7: u =', u.round(4), 'res =', res.round(3), 'compat = %.1f' % c)
    P = square_run(0.05 * 1.005, 0.05 * 0.995, 0.30)[-1]; print('square, radii: %.3f %.3f %.1f deg' % (P[0], P[1], np.degrees(P[2])))
    P = square_run(0.05, 0.05, 0.30 * 1.03)[-1]; print('square, track: %.3f %.3f %.1f deg' % (P[0], P[1], np.degrees(P[2])))
    print('parking: %.3f m per cycle' % (2 * 4.0 * (1 - np.cos(np.radians(20)))))
    print('Lie eps = 0.6:', lie_maneuver(0.6)[1][-1].round(4))
    J = mobile_manip_J(planar2r_fk([0.0, np.pi / 2]), planar2r_J([0.0, np.pi / 2]))
    print('6.8: z =', weighted_pinv(J, np.ones(5)).dot([0.1, 0.0]).round(4))


if __name__ == '__main__':
    check()
