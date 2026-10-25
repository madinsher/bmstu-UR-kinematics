# -*- coding: utf-8 -*-
"""ДЗ2: исполнение траектории из trajectory.csv в CoppeliaSim через ZeroMQ remote API.

Уставки подаются на КАЖДОМ шаге симуляции и вычисляются по модельному времени: q*(t) — интерполяция
таблицы trajectory.csv (столбцы t, q1…qN, qd1…qdN, которую пишет run_offline.py). Поэтому траектория
исполняется с тем законом движения, который вы рассчитали (разгон и торможение без скачков скорости),
а не ступеньками «точка — пауза — точка».

Подготовка:
  pip install coppeliasim-zmqremoteapi-client
  1) открыть в CoppeliaSim свою сцену с манипулятором (симуляция остановлена); шаг симуляции сцены
     желательно уменьшить до 5–10 мс — уставки обновляются один раз за шаг;
  2) в JOINT_PATHS указать пути к сочленениям сцены в порядке q1…qN, в TIP_PATH — путь к dummy в
     характерной точке инструмента;
  3) режим сочленений: динамический с управлением по положению (SET_MODE = 'target') или кинематический,
     когда положение задаётся напрямую (SET_MODE = 'position');
  4) python run_offline.py   ->  trajectory.csv
  5) python coppelia_run.py  ->  tip_trace.csv

ВНИМАНИЕ: нулевые положения и направления вращения сочленений в модели симулятора могут не совпадать с вашей
таблицей ДХ. Сдвиги и знаки задаются массивами Q_OFFSET и Q_SIGN; подберите их по шагу 5 инструкции.
Скрипт написан для ZeroMQ remote API (client.require('sim'), sim.setStepping, sim.step); со старым
remoteApi (sim.py, simConst.py) он не работает. Проверьте его на своей версии CoppeliaSim до сдачи.
"""
import os
import numpy as np

# пути — пример; точные имена посмотрите в иерархии своей сцены (выделите объект — путь виден в его свойствах)
JOINT_PATHS = ['/UR5/joint', '/UR5/link/joint', '/UR5/link/joint/link/joint',
               '/UR5/link/joint/link/joint/link/joint', '/UR5/link/joint/link/joint/link/joint/link/joint',
               '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint']
TIP_PATH = '/UR5/tip'                       # dummy в характерной точке инструмента (создайте, если его нет)
Q_SIGN = np.ones(6)                         # +1 / -1: направление вращения сочленения сцены относительно ДХ
Q_OFFSET = np.zeros(6)                      # рад: q_сцены = Q_SIGN * q_ДХ + Q_OFFSET
SET_MODE = 'target'                         # 'target' — sim.setJointTargetPosition, 'position' — sim.setJointPosition
APPROACH_QD, APPROACH_QDD = 0.5, 1.0        # рад/с и рад/с^2: пределы для подвода к первой точке
N_LAPS = 3                                  # число обходов окружности (для оценки повторяемости траектории)
DWELL = 0.5                                 # с: выдержка в неподвижности перед каждым обходом


def poly5_path(q_from, q_to, vmax, amax):
    """Переход из покоя в покой по многочлену 5-й степени: функция q(t) и длительность.
    Время выбирается так, чтобы пики 15/8·D/T и 10/√3·D/T² не превышали vmax и amax (лекция 4)."""
    D = np.abs(q_to - q_from).max()
    T = max(15.0 / 8.0 * D / vmax, np.sqrt(10.0 / np.sqrt(3.0) * D / amax), 1e-3)

    def q_of_t(t):
        tau = min(max(t / T, 0.0), 1.0)
        lam = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        return q_from + lam * (q_to - q_from)
    return q_of_t, T


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.loadtxt(os.path.join(here, 'trajectory.csv'), delimiter=',', skiprows=1)
    n = (data.shape[1] - 1) // 2
    t_tab, Q_tab = data[:, 0], data[:, 1:1 + n]
    T_lap = t_tab[-1]

    def q_ref(t):
        """Уставка по таблице в момент t от начала обхода (линейная интерполяция между строками)."""
        t = min(max(t, 0.0), T_lap)
        return np.array([np.interp(t, t_tab, Q_tab[:, i]) for i in range(n)])

    client = RemoteAPIClient()
    sim = client.require('sim')
    joints = [sim.getObject(p) for p in JOINT_PATHS]
    tip = sim.getObject(TIP_PATH)

    def command(q_dh):
        for j, v in zip(joints, Q_SIGN[:n] * q_dh + Q_OFFSET[:n]):
            if SET_MODE == 'position':
                sim.setJointPosition(j, float(v))
            else:
                sim.setJointTargetPosition(j, float(v))

    sim.setStepping(True)                   # шагами симуляции управляет эта программа
    sim.startSimulation()
    trace = []
    try:
        # подвод к первой точке: из текущего положения в покое в Q_tab[0] по многочлену 5-й степени
        q_now = (np.array([sim.getJointPosition(j) for j in joints]) - Q_OFFSET[:n]) * Q_SIGN[:n]
        approach, T_app = poly5_path(q_now, Q_tab[0], APPROACH_QD, APPROACH_QDD)
        t0 = sim.getSimulationTime()
        while sim.getSimulationTime() - t0 < T_app + DWELL:
            command(approach(sim.getSimulationTime() - t0))
            sim.step()
        # обходы окружности: уставка — функция модельного времени
        for lap in range(N_LAPS):
            t0 = sim.getSimulationTime()
            while True:
                t = sim.getSimulationTime() - t0
                command(q_ref(t))
                sim.step()
                p = sim.getObjectPosition(tip, sim.handle_world)
                q_act = (np.array([sim.getJointPosition(j) for j in joints]) - Q_OFFSET[:n]) * Q_SIGN[:n]
                trace.append([lap + 1, sim.getSimulationTime() - t0] + list(p) + list(q_act))
                if t > T_lap:
                    break
            t1 = sim.getSimulationTime()          # выдержка в конечной точке
            while sim.getSimulationTime() - t1 < DWELL:
                command(Q_tab[-1]); sim.step()
    finally:
        sim.stopSimulation()
    trace = np.array(trace)
    hdr = 'lap,t,x,y,z,' + ','.join('q%d' % (i + 1) for i in range(n))
    np.savetxt(os.path.join(here, 'tip_trace.csv'), trace, delimiter=',', header=hdr, comments='')
    print('записано строк: %d (обходов: %d, T = %.2f с) -> tip_trace.csv' % (len(trace), N_LAPS, T_lap))
    print('сравните x, y, z с окружностью (отклонение пути) и обходы между собой (повторяемость пути)')


if __name__ == '__main__':
    main()
