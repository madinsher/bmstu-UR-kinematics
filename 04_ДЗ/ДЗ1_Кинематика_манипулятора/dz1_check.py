# -*- coding: utf-8 -*-
"""Самопроверка ДЗ1: таблица Денавита — Хартенберга по известным позам, затем ваши аналитические формулы
против численного расчёта по этой таблице.

ЧТО ИМЕННО ПРОВЕРЯЕТСЯ.
  0. Таблица ДХ — по известным позам (блок KNOWN_POSES): в трёх конфигурациях, где положение схвата
     очевидно из рис. 1 задания, скрипт сравнивает расстояние схвата от оси Z0 и его высоту с тем, что
     даёт ваша таблица, и проверяет, что при увеличении q1 рука поворачивается вокруг Z0 против часовой
     стрелки (если смотреть сверху). Азимут нулевого положения q1 НЕ проверяется: он зависит от того,
     куда вы направили ось x0, и любой выбор правилен.
  1–4. Формулы прямой и обратной задач о положении и скорости — против произведения матриц ДХ
     ВАШЕЙ таблицы. Эти проверки ловят ошибки в выкладках, но НЕ ошибки в самой таблице: ошибочная, но
     самосогласованная таблица их проходит. Поэтому проверка 0 обязательна.
  Кроме того, скрипт рисует системы координат звеньев (оси x_i и z_i) в двух конфигурациях и сохраняет
  рисунок dz1_frames.png — сравните его со своей схемой и приложите к отчёту.

Что сделать:
  1. В блоке «ВАШИ ДАННЫЕ» задать номер схемы, длины, таблицу ДХ, типы сочленений, пределы и три функции
     с ВАШИМИ формулами.
  2. Запустить:  python dz1_check.py
  3. Снимок вывода и рисунок dz1_frames.png приложить к отчёту.

Сейчас блок заполнен примером из файла «Пример_решения.md» (манипулятор типа SCARA, SCHEME = 0) —
скрипт запускается как есть. Библиотека kinlib лежит в ../../03_Примеры_Jupyter.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '03_Примеры_Jupyter'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kinlib import RobotSerial, jacobian_numeric

pi = np.pi

# ======================================================================= ВАШИ ДАННЫЕ
SCHEME = 0                             # номер вашей схемы на рис. 1 задания: 1, 2 или 3; 0 — пример SCARA
L1, A1, A2 = 0.8, 0.6, 0.5            # параметры варианта, м
D3_MAX = 0.5                           # предельный ход телескопического звена
LENGTHS = (L1, A1, A2)                 # (l1, l2, l3) из табл. 1 вашего варианта; для примера SCARA — (l1, a1, a2)

# таблица ДХ: строки | d | a | alpha | theta | — постоянные части; переменная добавляется сама:
# для 'R' к theta, для 'P' к d
DH = np.array([[L1, A1, 0.0, 0.0],
               [0.0, A2, pi, 0.0],
               [0.0, 0.0, 0.0, 0.0]])
JOINT_TYPES = 'RRP'
Q_LIMITS = [(-pi, pi), (-pi, pi), (0.0, D3_MAX)]

P_TARGET = np.array([0.7, 0.4, 0.5])   # табл. 2
V_TARGET = np.array([0.2, -0.1, 0.05]) # табл. 3


def my_q_from_figure(q_fig):
    """Перевод обобщённых координат «как на рис. 1» в координаты ВАШЕЙ таблицы.
    На рис. 1: телескопические ходы d отсчитываются от полностью втянутого звена; в схеме 2 угол q3
    отсчитывается от вертикали вниз, положительный — звено l3 отклоняется от колонны наружу; в схеме 3
    угол q2 отсчитывается от горизонтали, положительный — плечо поднимается. Если в вашей таблице
    нуль или направление отсчёта другие (например, ваш угол равен q3 − 90°), впишите связь здесь."""
    return np.asarray(q_fig, dtype=float)


def my_fk(q):
    """ВАША формула прямой задачи: положение схвата (x, y, z)."""
    q1, q2, d3 = q
    return np.array([A1*np.cos(q1) + A2*np.cos(q1 + q2),
                     A1*np.sin(q1) + A2*np.sin(q1 + q2),
                     L1 - d3])


def my_ik(p):
    """ВАШЕ решение обратной задачи: список всех решений q (без проверки пределов)."""
    x, y, z = p
    c2 = (x*x + y*y - A1*A1 - A2*A2) / (2*A1*A2)
    if abs(c2) > 1:
        return []
    sols = []
    for sgn in (1, -1):
        q2 = sgn*np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(A2*np.sin(q2), A1 + A2*np.cos(q2))
        sols.append(np.array([q1, q2, L1 - z]))
    return sols


def my_jacobian(q):
    """ВАША матрица Якоби линейных скоростей J_v (3×3): v = J_v · q̇."""
    q1, q2, d3 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-A1*s1 - A2*s12, -A2*s12, 0.0],
                     [A1*c1 + A2*c12, A2*c12, 0.0],
                     [0.0, 0.0, -1.0]])
# ======================================================================= КОНЕЦ БЛОКА


# ======================================================================= KNOWN_POSES (не менять)
# Для каждой схемы — три конфигурации «как на рис. 1» (q1 = 0, остальные координаты заданы в долях длин)
# и ожидаемые расстояние схвата от оси Z0, r = sqrt(x^2 + y^2), и высота z. Значения сверены с эталонным
# решением преподавателя для всех вариантов табл. 1.
KNOWN_POSES = {
    0: [  # пример SCARA: q = (q1, q2, d3), l = (l1, a1, a2)
        ('рука вытянута, телескоп втянут', lambda l: (0.0, 0.0, 0.0), lambda l: (l[1] + l[2], l[0])),
        ('локоть под 90°, телескоп выдвинут на 0,2 м', lambda l: (0.0, pi/2, 0.2), lambda l: (np.hypot(l[1], l[2]), l[0] - 0.2)),
        ('рука сложена', lambda l: (0.0, pi, 0.1), lambda l: (abs(l[1] - l[2]), l[0] - 0.1)),
    ],
    1: [  # схема 1 (ВПП): q = (q1, d2, d3)
        ('горизонтальный ход 0,5*l2, вертикальный 0,25*l3', lambda l: (0.0, 0.5*l[1], 0.25*l[2]), lambda l: (0.5*l[1], l[0] - 0.25*l[2])),
        ('горизонтальный ход l2, вертикальный 0', lambda l: (0.0, l[1], 0.0), lambda l: (l[1], l[0])),
        ('горизонтальный ход 0,2*l2, вертикальный l3', lambda l: (0.0, 0.2*l[1], l[2]), lambda l: (0.2*l[1], l[0] - l[2])),
    ],
    2: [  # схема 2 (ВПВ): q = (q1, d2, q3)
        ('ход 0,5*l2, звено l3 висит вниз (q3 = 0)', lambda l: (0.0, 0.5*l[1], 0.0), lambda l: (0.5*l[1], l[0] - l[2])),
        ('ход 0,5*l2, звено l3 горизонтально наружу (q3 = 90°)', lambda l: (0.0, 0.5*l[1], pi/2), lambda l: (0.5*l[1] + l[2], l[0])),
        ('ход 0,2*l2, звено l3 вертикально вверх (q3 = 180°)', lambda l: (0.0, 0.2*l[1], pi), lambda l: (0.2*l[1], l[0] + l[2])),
    ],
    3: [  # схема 3 (ВВП): q = (q1, q2, d3)
        ('плечо горизонтально, телескоп втянут', lambda l: (0.0, 0.0, 0.0), lambda l: (l[1], l[0])),
        ('плечо вертикально вверх (q2 = 90°), ход 0,5*l3', lambda l: (0.0, pi/2, 0.5*l[2]), lambda l: (0.0, l[0] + l[1] + 0.5*l[2])),
        ('плечо опущено на 30° (q2 = -30°), ход l3', lambda l: (0.0, -pi/6, l[2]), lambda l: ((l[1] + l[2])*np.cos(pi/6), l[0] - 0.5*(l[1] + l[2]))),
    ],
}


def check_known_poses(robot, verdict):
    """Проверка 0: таблица ДХ по известным позам. Возвращает список конфигураций для рисунка."""
    poses = KNOWN_POSES[SCHEME]
    qs = []
    for name, qf, pe in poses:
        q_fig = np.array(qf(LENGTHS)); q = my_q_from_figure(q_fig)
        p = robot.forward(q).t_3_1.ravel()
        r_exp, z_exp = pe(LENGTHS)
        err = max(abs(np.hypot(p[0], p[1]) - r_exp), abs(p[2] - z_exp))
        print('  %s' % name)
        print('     ожидается r = %.4f, z = %.4f; по вашей таблице r = %.4f, z = %.4f' % (r_exp, z_exp, np.hypot(p[0], p[1]), p[2]))
        verdict('  поза «%s»' % name[:30], err)
        qs.append(q)
    # направление отсчёта q1: поворот на +90° должен повернуть схват вокруг Z0 против часовой стрелки
    q_fig = np.array(poses[0][1](LENGTHS)); q_fig2 = q_fig.copy(); q_fig2[0] += pi/2
    p1 = robot.forward(my_q_from_figure(q_fig)).t_3_1.ravel(); p2 = robot.forward(my_q_from_figure(q_fig2)).t_3_1.ravel()
    if np.hypot(p1[0], p1[1]) > 1e-9:
        daz = np.degrees(np.arctan2(p1[0]*p2[1] - p1[1]*p2[0], p1[0]*p2[0] + p1[1]*p2[1]))
        print('  при q1 -> q1 + 90° схват поворачивается вокруг Z0 на %.1f° (ожидается +90°, против часовой стрелки)' % daz)
        verdict('  направление отсчёта q1', abs(daz - 90.0) / 90.0)
    return qs


def draw_frames(robot, configs, fname):
    """Системы координат звеньев: оси x_i (красные) и z_i (синие) в нескольких конфигурациях."""
    fig = plt.figure(figsize=(6*len(configs), 6))
    span = max(LENGTHS) * 1.2
    for k, q in enumerate(configs):
        ax = fig.add_subplot(1, len(configs), k + 1, projection='3d')
        robot.forward(q)
        frames = [np.eye(4)] + [f.t_4_4 for f in robot.axis_frames]
        o = np.array([F[0:3, 3] for F in frames])
        ax.plot(o[:, 0], o[:, 1], o[:, 2], color='#1E2761', lw=3)
        L = 0.25 * max(LENGTHS)
        for i, F in enumerate(frames):
            p = F[0:3, 3]
            for col, axis, name in (('#B85042', 0, 'x'), ('#167D8D', 2, 'z')):
                d = F[0:3, axis] * L
                ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], color=col, lw=1.5, arrow_length_ratio=0.15)
                ax.text(*(p + 1.12*d), '%s%d' % (name, i), color=col, fontsize=9)
            ax.text(*(p - 0.08*L), 'O%d' % i, fontsize=8, color='#5A6478')
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-0.6*span, 1.4*span)
        try:
            ax.set_box_aspect((1, 1, 1))           # одинаковый масштаб осей (matplotlib 3.3+)
        except AttributeError:
            pass
        ax.set_xlabel('X0'); ax.set_ylabel('Y0'); ax.set_zlabel('Z0')
        ax.set_title('q = %s' % np.round(q, 3), fontsize=9)
    fig.suptitle('Системы координат звеньев по вашей таблице ДХ: x_i — красные, z_i — синие', fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def main():
    np.set_printoptions(precision=5, suppress=True)
    robot = RobotSerial(DH, joint_types=JOINT_TYPES)
    rng = np.random.RandomState(0)
    ok_all = True

    def verdict(name, err, tol=1e-6):
        nonlocal ok_all
        good = err < tol
        ok_all &= good
        print('  [%s] %-46s расхождение %.2e' % ('OK' if good else 'ОШИБКА', name, err))

    print('0. Таблица ДХ по известным позам рис. 1 (схема %s)' % (SCHEME if SCHEME else 'примера SCARA'))
    known_q = check_known_poses(robot, verdict)

    print('1. Прямая задача о положении (20 случайных конфигураций)')
    err = 0.0
    for _ in range(20):
        q = np.array([rng.uniform(a, b) for a, b in Q_LIMITS])
        err = max(err, np.abs(my_fk(q) - robot.forward(q).t_3_1.ravel()).max())
    verdict('my_fk против произведения матриц ДХ', err)

    print('2. Обратная задача о положении, цель', P_TARGET)
    sols = my_ik(P_TARGET)
    if not sols:
        print('  решений нет — проверьте достижимость точки аналитически')
    good_sols = []
    for k, q in enumerate(sols, 1):
        inside = all(a - 1e-9 <= v <= b + 1e-9 for v, (a, b) in zip(q, Q_LIMITS))
        e = np.abs(robot.forward(q).t_3_1.ravel() - P_TARGET).max()
        print('  решение %d: q = %s  в пределах: %s' % (k, q, 'да' if inside else 'НЕТ'))
        verdict('  ПЗК(решение %d) = цель' % k, e)
        if inside:
            good_sols.append(q)
    # сколько решений находит итерационный метод из разных начальных приближений
    found = []
    for _ in range(30):
        q0 = np.array([rng.uniform(a, b) for a, b in Q_LIMITS])
        q = robot.inverse_position(P_TARGET, q0=q0).copy()
        if robot.is_reachable_inverse:
            q = np.array([(v + pi) % (2*pi) - pi if t == 'R' else v for v, t in zip(q, JOINT_TYPES)])
            if not any(np.allclose(q, f, atol=1e-4) for f in found):
                found.append(q)
    print('  итерационный метод из 30 случайных стартов нашёл различных решений: %d' % len(found))
    for q in found:
        known = any(np.allclose(q, s, atol=1e-4) for s in sols)
        print('    q = %s %s' % (q, '' if known else '  <-- этого решения нет в my_ik'))

    print('3. Матрица Якоби (20 случайных конфигураций)')
    e1 = e2 = 0.0
    for _ in range(20):
        q = np.array([rng.uniform(a, b) for a, b in Q_LIMITS])
        robot.forward(q)
        e1 = max(e1, np.abs(my_jacobian(q) - robot.jacobian[0:3]).max())   # строки 0..2 — блок J_v
        e2 = max(e2, np.abs(my_jacobian(q) - jacobian_numeric(robot, q)[0:3]).max())
    verdict('my_jacobian против столбцов (z x p; z)', e1)
    verdict('my_jacobian против численного дифференцирования', e2, tol=1e-4)

    print('4. Обратная задача о скорости, v =', V_TARGET)
    for k, q in enumerate(good_sols or sols, 1):
        J = my_jacobian(q)
        d = np.linalg.det(J)
        if abs(d) < 1e-9:
            print('  конфигурация %d вырождена: det J = %.2e' % (k, d)); continue
        qd = np.linalg.solve(J, V_TARGET)
        print('  конфигурация %d: det J = %.5f, dq/dt = %s' % (k, d, qd))
        verdict('  J * dq/dt = v', np.abs(J @ qd - V_TARGET).max())

    fname = os.path.join(HERE, 'dz1_frames.png')
    draw_frames(robot, [known_q[0], (good_sols or sols or known_q)[0]], fname)
    print('\nРисунок систем координат: %s' % fname)
    print('ИТОГ:', 'все проверки пройдены' if ok_all else
          'ЕСТЬ РАСХОЖДЕНИЯ — ошибка в проверке 0 указывает на таблицу ДХ (или на связь в my_q_from_figure), '
          'в проверках 1–4 — на формулы')


if __name__ == '__main__':
    main()
