# -*- coding: utf-8 -*-
"""ДЗ2 без симулятора: окружность по трём точкам -> закон движения во времени -> ОЗК в узлах ->
сплайн в пространстве сочленений -> проверка пределов скоростей (с растяжением времени) -> trajectory.csv.

Запуск (Python 3, numpy, scipy, matplotlib):  python run_offline.py
Результат: trajectory.csv (столбцы t, q1…qN, qd1…qdN; углы в рад, скорости в рад/с) и offline_check.png.
Замените робота, пределы и три точки на свои. В примере — UR5e (kinlib.make_ur5e, официальная таблица ДХ
Universal Robots) и точки варианта 1 задания.

Конвейер (лекция 4):
  1) окружность через P1, P2, P3: центр C, радиус ρ, орты плоскости u, w, нормаль n (circle3p.py);
  2) закон движения: угол на окружности φ(t) = 2π·λ5(t/T), λ5(τ) = 10τ³ − 15τ⁴ + 6τ⁵ — нулевые скорость
     и ускорение схвата в начале и в конце обхода;
  3) число узлов по допустимой ошибке хорды ε_х: соседние узлы должны отстоять по углу не больше чем на
     Δφ = 2·arccos(1 − ε_х/ρ); узлы берутся равномерно по времени, а наибольший шаг по углу приходится на
     середину обхода, где dλ5/dτ = 15/8;
  4) ОЗК в узлах «цепочкой близких целей» (старт — решение в предыдущем узле), полный шаг Ньютона;
  5) кубический сплайн по времени для каждого qᵢ с нулевыми скоростями на концах (trajlib.spline_m);
  6) проверка max|q̇ᵢ| ≤ q̇ᵢ^max и, если заданы пределы ускорений, max|q̈ᵢ| ≤ q̈ᵢ^max; при нарушении время
     обхода растягивается в κ = max(maxᵢ max|q̇ᵢ|/q̇ᵢ^max, √(maxᵢ max|q̈ᵢ|/q̈ᵢ^max)) раз: скорости делятся на κ,
     ускорения — на κ²;
  7) контроль: отклонение пути схвата после сплайна от окружности; близость к особым конфигурациям —
     σ_min отдельно для J_v (м/рад) и J_ω (безразмерный блок), а также σ_min безразмерной матрицы
     J_n = diag(E/R_max, E)·J (σ_min полной 6×N матрицы J зависит от того, в метрах или миллиметрах
     записаны длины, и сравнивать её между отчётами нельзя); сравнение q̇ сплайна с J⁺·ṡ в узлах.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '03_Примеры_Jupyter'))
sys.path.insert(0, HERE)
from kinlib import RobotSerial, Frame, UR5E_DH, UR5E_QLIM, UR5E_QD_MAX, UR5E_REACH
from trajlib import lam_poly5, spline_m, spline_eval, spline_extrema
from circle3p import circle_through_points, tool_orientation

# ------------------------------------------------------------------ ВАШИ ДАННЫЕ
DH = UR5E_DH                                 # таблица ДХ вашего робота | d | a | alpha | theta0 |
JOINT_TYPES = 'RRRRRR'
Q_LIMITS = UR5E_QLIM                         # пределы сочленений из паспорта, рад (UR5e: ±360°)
QD_MAX = UR5E_QD_MAX                         # пределы скоростей сочленений из паспорта, рад/с (UR5e: 180°/с)
QDD_MAX = None                               # пределы ускорений, рад/с²; в паспорте UR5e их нет — None = не проверять
R_MAX = UR5E_REACH                           # паспортная досягаемость R_max, м (UR5e: 0,85 м)
VARIANT_POINTS = ((0.45, -0.20, 0.35), (0.65, 0.00, 0.50), (0.45, 0.20, 0.35))   # вариант 1, в долях R_max
P1, P2, P3 = (R_MAX * np.array(p) for p in VARIANT_POINTS)
T_CIRCLE = 6.0                               # начальное время обхода окружности, с
EPS_CHORD = 1e-4                             # допустимая ошибка хорды ε_х, м (0,1 мм)
DT_OUT = 0.01                                # шаг по времени в trajectory.csv, с
Q_START = np.radians([180, -90, 90, -90, -90, 0])   # начальное приближение ОЗК для первой точки
TOL_POS, TOL_ROT = 1e-6, 1e-6                # допуски ОЗК: м и рад
# ------------------------------------------------------------------


def circle_point(c, rho, u, w, phi):
    return c + rho * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), w))


def distance_to_circle(p, c, rho, n):
    """Расстояние от точек p [k,3] до окружности (центр c, радиус ρ, нормаль n)."""
    d = p - c
    h = d @ n                                        # отклонение от плоскости
    r_in = np.linalg.norm(d - np.outer(h, n), axis=1)
    return np.hypot(r_in - rho, h)


def main():
    robot = RobotSerial(DH, joint_types=JOINT_TYPES, plot_xlim=(-0.9, 0.9), plot_ylim=(-0.9, 0.9), plot_zlim=(-0.3, 1.1))
    N = robot.num_axis
    c, rho, u, w, n = circle_through_points(P1, P2, P3)
    R_tool = tool_orientation(n)
    print('окружность: центр %s м, радиус rho = %.4f м, нормаль %s' % (c.round(4), rho, n.round(4)))
    far = max(np.linalg.norm(c) + rho, max(np.linalg.norm(p) for p in (P1, P2, P3)))
    print('грубая проверка зоны: наибольшее удаление точек окружности от основания <= %.3f м, R_max = %.3f м' % (far, R_MAX))

    # 1) число узлов по ошибке хорды и узлы по времени
    dphi_max = 2 * np.arccos(1 - EPS_CHORD / rho)
    n_nodes = int(np.ceil(2 * np.pi * (15.0 / 8.0) / dphi_max))
    tau = np.linspace(0.0, 1.0, n_nodes + 1)
    lam, dlam, _ = lam_poly5(tau)
    phi = 2 * np.pi * lam
    pts = circle_point(c, rho, u, w, phi)
    print('ошибка хорды eps = %.2g мм -> шаг по углу <= %.3f°, узлов N_у = %d (равномерно по времени)'
          % (EPS_CHORD * 1e3, np.degrees(dphi_max), n_nodes))

    # 2) ОЗК в узлах цепочкой близких целей
    Q = np.zeros((len(pts), N)); ok = np.zeros(len(pts), dtype=bool); iters = []
    q = Q_START.copy()
    for k, p in enumerate(pts):
        robot.inverse_numerical(Frame.from_r_3_3(R_tool, p), q0=q, tol_pos=TOL_POS, tol_rot=TOL_ROT, max_iter=200)
        q = robot.axis_values.copy()
        if k > 0:                                        # без скачков на 2π между соседними узлами
            q = Q[k - 1] + (q - Q[k - 1] + np.pi) % (2 * np.pi) - np.pi
        inside = np.all((q >= Q_LIMITS[:, 0]) & (q <= Q_LIMITS[:, 1]))
        Q[k], ok[k] = q, robot.is_reachable_inverse and inside
        iters.append(len(robot.inv_history) - 1)
    print('достижимо узлов: %d из %d; итераций Ньютона: первый узел %d, далее в среднем %.1f, наибольшее %d'
          % (ok.sum(), len(pts), iters[0], np.mean(iters[1:]), max(iters[1:])))
    if not ok.all():
        print('НЕДОСТИЖИМЫЕ узлы (индексы):', np.where(~ok)[0].tolist(),
              '— окружность выходит за рабочую зону или пределы сочленений; дальше результат не имеет смысла')
    jump = np.abs(np.diff(Q, axis=0)).max()
    print('наибольшее изменение координаты между соседними узлами: %.2f° (десятки градусов — смена ветви)' % np.degrees(jump))

    # 3) сплайн по времени и растяжение
    def splines(T):
        t = tau * T
        return t, [spline_m(t, Q[:, i], 'clamped', 0.0, 0.0) for i in range(N)]

    T = T_CIRCLE
    t_nodes, M = splines(T)
    vpk = np.array([spline_extrema(t_nodes, Q[:, i], M[i])[0] for i in range(N)])
    apk = np.array([spline_extrema(t_nodes, Q[:, i], M[i])[1] for i in range(N)])
    kappa_v = np.max(vpk / QD_MAX)
    kappa_a = np.sqrt(np.max(apk / np.asarray(QDD_MAX))) if QDD_MAX is not None else 0.0
    kappa = max(kappa_v, kappa_a)
    print('при T = %.2f с: max_i max|qd_i|/qd_max_i = %.3f%s; наименьшее допустимое время обхода T_min = %.2f с'
          % (T, kappa_v, (', sqrt(max_i max|qdd_i|/qdd_max_i) = %.3f' % kappa_a) if QDD_MAX is not None else '', kappa * T))
    if kappa > 1:
        T = kappa * T
        t_nodes, M = splines(T)
        vpk /= kappa; apk /= kappa ** 2
        print('пределы нарушены -> время растянуто в kappa = %.3f раза: T = %.2f с' % (kappa, T))
    else:
        print('пределы выполняются, растяжение не нужно (kappa = %.3f <= 1)' % kappa)
    for i in range(N):
        print('  q%d: max|qd| = %6.1f °/с (%4.0f %% предела), max|qdd| = %7.1f °/с^2'
              % (i + 1, np.degrees(vpk[i]), 100 * vpk[i] / QD_MAX[i], np.degrees(apk[i])))

    # 4) плотная выборка, контроль пути и запись trajectory.csv
    t_out = np.arange(0.0, T + 1e-9, DT_OUT)
    if t_out[-1] < T - 1e-12:
        t_out = np.append(t_out, T)
    q_out = np.zeros((len(t_out), N)); qd_out = np.zeros_like(q_out); qdd_out = np.zeros_like(q_out)
    for i in range(N):
        q_out[:, i], qd_out[:, i], qdd_out[:, i] = spline_eval(t_nodes, Q[:, i], M[i], t_out)
    p_out = np.array([robot.forward(qk).t_3_1.ravel() for qk in q_out])
    dev = distance_to_circle(p_out, c, rho, n)
    ang = np.array([robot.forward(qk).angle_to(Frame.from_r_3_3(R_tool, np.zeros(3))) for qk in q_out])
    print('путь схвата после сплайна: наибольшее отклонение от окружности %.3f мм (eps_х = %.2f мм), '
          'наибольшее отклонение ориентации %.3f°' % (dev.max() * 1e3, EPS_CHORD * 1e3, np.degrees(ang.max())))

    # 5) близость к особым конфигурациям и сравнение q̇ сплайна с J⁺ṡ в узлах.
    # σ_min полной матрицы 6×N зависит от единиц длины (строки v — в м/рад, строки ω — безразмерные),
    # поэтому считаем σ_min(J_v) и σ_min(J_ω) отдельно и σ_min безразмерной матрицы J_n = diag(E/R_max, E)·J:
    # она обращается в нуль во всех особых конфигурациях, в том числе кистевой, которую блоки по отдельности
    # могут не заметить.
    s_v, s_w, s_n, qd_err = [], [], [], 0.0
    for k, qk in enumerate(Q):
        robot.forward(qk)
        J = robot.jacobian                                  # порядок блоков [v; ω]
        s_v.append(np.linalg.svd(J[0:3], compute_uv=False).min())
        s_w.append(np.linalg.svd(J[3:6], compute_uv=False).min())
        s_n.append(np.linalg.svd(np.vstack([J[0:3] / R_MAX, J[3:6]]), compute_uv=False).min())
        v = 2 * np.pi * dlam[k] / T * rho * (-np.sin(phi[k]) * u + np.cos(phi[k]) * w)   # ṡ = (v; ω = 0)
        qd_ref = np.linalg.pinv(J) @ np.concatenate([v, np.zeros(3)])
        qd_spl = np.array([spline_eval(t_nodes, Q[:, i], M[i], t_nodes[k:k + 1])[1][0] for i in range(N)])
        qd_err = max(qd_err, np.abs(qd_spl - qd_ref).max())
    s_v, s_w, s_n = np.array(s_v), np.array(s_w), np.array(s_n)
    print('наименьшие сингулярные числа вдоль траектории: sigma_min(J_v) = %.3f м/рад, sigma_min(J_w) = %.3f, '
          'sigma_min(J_n) = %.3f (J_n: строки v поделены на R_max)' % (s_v.min(), s_w.min(), s_n.min()))
    print('контроль скоростей: max|qd_сплайн - J+ * ds/dt| в узлах = %.2e рад/с (наибольшая скорость %.1f °/с)'
          % (qd_err, np.degrees(np.abs(qd_out).max())))

    hdr = 't,' + ','.join('q%d' % (i + 1) for i in range(N)) + ',' + ','.join('qd%d' % (i + 1) for i in range(N))
    np.savetxt(os.path.join(HERE, 'trajectory.csv'), np.column_stack([t_out, q_out, qd_out]), delimiter=',',
               header=hdr, comments='', fmt='%.8f')

    # 6) графики
    fig = plt.figure(figsize=(13, 8.6))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    for k in (0, n_nodes // 3, 2 * n_nodes // 3):
        robot.forward(Q[k]); robot.draw(ax=ax, alpha=0.5, cylinders=False)
    ax.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2], color='#D97706', lw=2)
    ax.scatter(*np.array([P1, P2, P3]).T, color='#B85042', s=40)
    ax.set_title('окружность и три конфигурации')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_out, np.degrees(q_out)); ax2.set_xlabel('t, с'); ax2.set_ylabel('q, °'); ax2.grid(alpha=.3)
    ax2.legend(['q%d' % (i + 1) for i in range(N)], fontsize=7, ncol=2); ax2.set_title('обобщённые координаты')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_out, 100 * qd_out / QD_MAX); ax3.axhline(100, color='#B85042', ls='--', lw=1); ax3.axhline(-100, color='#B85042', ls='--', lw=1)
    ax3.set_xlabel('t, с'); ax3.set_ylabel(r'$\dot q_i\,/\,\dot q_i^{\max}$, %'); ax3.grid(alpha=.3)
    ax3.legend(['q%d' % (i + 1) for i in range(N)], fontsize=7, ncol=2); ax3.set_title('скорости сочленений в долях паспортного предела')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t_out, dev * 1e3, color='#1E2761', label='отклонение от окружности, мм')
    ax4.axhline(EPS_CHORD * 1e3, color='#1E2761', ls=':', lw=1)
    ax4.set_xlabel('t, с'); ax4.set_ylabel('мм'); ax4.grid(alpha=.3)
    ax5 = ax4.twinx()
    ax5.plot(t_nodes, s_v, color='#167D8D', label='σ_min(J_v), м/рад'); ax5.plot(t_nodes, s_w, color='#D97706', label='σ_min(J_ω)')
    ax5.plot(t_nodes, s_n, color='#B85042', label='σ_min(J_n), J_n = diag(E/R_max, E)·J')
    ax5.set_ylim(0, None)
    h1, l1 = ax4.get_legend_handles_labels(); h2, l2 = ax5.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right'); ax4.set_title('точность пути и близость к особым конфигурациям')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'offline_check.png'), dpi=110)
    print('сохранено: trajectory.csv (%d строк, T = %.2f с), offline_check.png' % (len(t_out), T))


if __name__ == '__main__':
    main()
