# -*- coding: utf-8 -*-
"""Самопроверка ДЗ1: ваши аналитические формулы против численного расчёта по таблице Денавита — Хартенберга.

Что сделать:
  1. В блоке «ВАШИ ДАННЫЕ» задать таблицу ДХ, типы сочленений, пределы и три функции с ВАШИМИ формулами.
  2. Запустить:  python dz1_check.py
  3. Снимок вывода приложить к отчёту.

Сейчас блок заполнен примером из файла «Пример_решения.md» (манипулятор типа SCARA) — скрипт запускается как есть.
Библиотека kinlib лежит в ../../03_Примеры_Jupyter.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '03_Примеры_Jupyter'))
import matplotlib
matplotlib.use('Agg')
from kinlib import RobotSerial, jacobian_numeric

pi = np.pi

# ======================================================================= ВАШИ ДАННЫЕ
L1, A1, A2 = 0.8, 0.6, 0.5            # параметры варианта, м
D3_MAX = 0.5                           # предельный ход телескопического звена

# таблица ДХ: строки | d | a | alpha | theta | — постоянные части; переменная добавляется сама:
# для 'R' к theta, для 'P' к d
DH = np.array([[L1, A1, 0.0, 0.0],
               [0.0, A2, pi, 0.0],
               [0.0, 0.0, 0.0, 0.0]])
JOINT_TYPES = 'RRP'
Q_LIMITS = [(-pi, pi), (-pi, pi), (0.0, D3_MAX)]

P_TARGET = np.array([0.7, 0.4, 0.5])   # табл. 2
V_TARGET = np.array([0.2, -0.1, 0.05]) # табл. 3


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
        e1 = max(e1, np.abs(my_jacobian(q) - robot.jacobian[0:3]).max())
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

    print('\nИТОГ:', 'все проверки пройдены' if ok_all else 'ЕСТЬ РАСХОЖДЕНИЯ — ищите ошибку в формулах или в таблице ДХ')


if __name__ == '__main__':
    main()
