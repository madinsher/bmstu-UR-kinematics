# -*- coding: utf-8 -*-
"""ДЗ2 без симулятора: окружность по трём точкам -> проверка рабочей зоны -> ОЗК вдоль траектории -> графики.

Запуск:  python run_offline.py
Результат: trajectory.csv (обобщённые координаты по точкам окружности) и offline_check.png.
Замените таблицу ДХ и три точки на свои. В примере — шестизвенник с размерами типа UR10 из kinlib.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '03_Примеры_Jupyter'))
sys.path.insert(0, HERE)
from kinlib import RobotSerial, Frame, UR_DH
from circle3p import circle_through_points, circle_points, tool_orientation

# ------------------------------------------------------------------ ВАШИ ДАННЫЕ
DH = UR_DH                                   # таблица ДХ вашего робота | d | a | alpha | theta0 |
JOINT_TYPES = 'RRRRRR'
Q_LIMITS = np.radians([[-360, 360]] * 6)     # пределы сочленений из паспорта робота
P1, P2, P3 = (0.55, -0.25, 0.45), (0.80, 0.00, 0.65), (0.55, 0.25, 0.45)
NUM = 72                                     # число точек на окружности
Q_START = np.radians([0, -60, 90, -30, 90, 0])
# ------------------------------------------------------------------


def main():
    robot = RobotSerial(DH, joint_types=JOINT_TYPES, plot_xlim=(-1.3, 1.3), plot_ylim=(-1.3, 1.3), plot_zlim=(-0.3, 1.5))
    c, r, u, w, n = circle_through_points(P1, P2, P3)
    pts, s = circle_points(P1, P2, P3, NUM)
    R_tool = tool_orientation(n)
    print('окружность: центр %s, радиус %.4f м, нормаль %s' % (c.round(4), r, n.round(4)))

    # 1) грубая проверка рабочей зоны: досягаемость по радиусу (уточняется решением ОЗК ниже)
    reach = np.abs(DH[:, 0]).sum() + np.abs(DH[:, 1]).sum()
    far = np.linalg.norm(pts, axis=1).max()
    print('максимальное удаление точки окружности от основания %.3f м, сумма длин звеньев %.3f м' % (far, reach))

    # 2) ОЗК вдоль окружности «цепочкой близких целей»: старт каждой точки — решение предыдущей
    Q = np.zeros((len(pts), robot.num_axis)); ok = np.zeros(len(pts), dtype=bool); iters = []
    q = Q_START.copy()
    for k, p in enumerate(pts):
        robot.inverse_numerical(Frame.from_r_3_3(R_tool, p), q0=q, max_iter=200)
        q = robot.axis_values.copy()
        inside = np.all((q >= Q_LIMITS[:, 0]) & (q <= Q_LIMITS[:, 1]))
        Q[k], ok[k] = q, robot.is_reachable_inverse and inside
        iters.append(len(robot.inv_history))
    print('достижимо точек: %d из %d; итераций на точку: первая %d, далее в среднем %.1f' % (ok.sum(), len(pts), iters[0], np.mean(iters[1:])))
    if not ok.all():
        print('НЕДОСТИЖИМЫЕ точки (индексы):', np.where(~ok)[0].tolist(), '— окружность выходит за рабочую зону или пределы сочленений')

    # 3) контроль: ПЗК по найденным q и скачки обобщённых координат (признак смены ветви или особой точки)
    fk = np.array([robot.forward(qk).t_3_1.ravel() for qk in Q])
    err = np.linalg.norm(fk - pts, axis=1)
    jump = np.abs(np.diff(Q, axis=0)).max()
    smin = [np.linalg.svd(robot.forward(qk) and robot.jacobian, compute_uv=False).min() for qk in Q]
    print('ошибка положения: макс. %.2e м; наибольший скачок координаты между соседними точками %.1f°; мин. сингулярное число J: %.3f' % (err.max(), np.degrees(jump), min(smin)))

    np.savetxt(os.path.join(HERE, 'trajectory.csv'), np.column_stack([s, Q]), delimiter=',', header='s,' + ','.join('q%d' % (i + 1) for i in range(robot.num_axis)), comments='')

    fig = plt.figure(figsize=(13, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    for k in (0, NUM // 3, 2 * NUM // 3):
        robot.forward(Q[k]); robot.draw(ax=ax, alpha=0.5, cylinders=False)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#D97706', lw=2)
    ax.scatter(*np.array([P1, P2, P3]).T, color='#B85042', s=40)
    ax.set_title('окружность и три конфигурации')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(np.degrees(s), np.degrees(Q)); ax2.set_xlabel('параметр окружности s, °'); ax2.set_ylabel('q, °'); ax2.grid(alpha=.3)
    ax2.legend(['q%d' % (i + 1) for i in range(robot.num_axis)], fontsize=7, ncol=2); ax2.set_title('обобщённые координаты')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(np.degrees(s), np.maximum(err, 1e-12), label='ошибка положения, м'); ax3.plot(np.degrees(s), smin, label='σ_min(J)')
    ax3.set_xlabel('s, °'); ax3.grid(alpha=.3); ax3.legend(fontsize=8); ax3.set_title('точность и близость к особым точкам')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'offline_check.png'), dpi=110)
    print('сохранено: trajectory.csv, offline_check.png')


if __name__ == '__main__':
    main()
