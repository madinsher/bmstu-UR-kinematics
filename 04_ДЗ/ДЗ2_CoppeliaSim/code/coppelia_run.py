# -*- coding: utf-8 -*-
"""ДЗ2: воспроизведение траектории из trajectory.csv в CoppeliaSim через ZeroMQ remote API.

Подготовка:
  pip install coppeliasim-zmqremoteapi-client
  1) открыть в CoppeliaSim свою сцену с манипулятором (симуляция остановлена);
  2) в JOINT_PATHS указать пути к сочленениям сцены в порядке q1…qN, в TIP_PATH — путь к dummy на фланце/инструменте;
  3) сочленения перевести в режим «Dynamic / Position control» (или «Kinematic mode» для чисто кинематической сцены);
  4) python run_offline.py   ->  trajectory.csv
  5) python coppelia_run.py

ВНИМАНИЕ: нулевые положения и направления вращения сочленений в модели симулятора могут не совпадать с вашей
таблицей ДХ. Сдвиги и знаки задаются массивами Q_OFFSET и Q_SIGN; подберите их по шагу 5 инструкции.
Скрипт написан по документации ZeroMQ remote API (CoppeliaSim 4.5+); со старым remoteApi (sim.py, simConst.py)
он не работает.
"""
import os, time
import numpy as np

JOINT_PATHS = ['/UR10/joint', '/UR10/link/joint', '/UR10/link/joint/link/joint',
               '/UR10/link/joint/link/joint/link/joint', '/UR10/link/joint/link/joint/link/joint/link/joint',
               '/UR10/link/joint/link/joint/link/joint/link/joint/link/joint']
TIP_PATH = '/UR10/tip'                      # dummy в характерной точке инструмента (создайте, если его нет)
Q_SIGN = np.ones(6)                         # +1 / -1: направление вращения сочленения сцены относительно ДХ
Q_OFFSET = np.zeros(6)                      # рад: q_сцены = Q_SIGN * q_ДХ + Q_OFFSET
DT_POINT = 0.08                             # с модельного времени на одну точку окружности


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    here = os.path.dirname(os.path.abspath(__file__))
    data = np.loadtxt(os.path.join(here, 'trajectory.csv'), delimiter=',', skiprows=1)
    Q = data[:, 1:]

    client = RemoteAPIClient()
    sim = client.require('sim')
    joints = [sim.getObject(p) for p in JOINT_PATHS]
    tip = sim.getObject(TIP_PATH)

    sim.setStepping(True)                   # шагами симуляции управляет эта программа
    sim.startSimulation()
    trace = []
    try:
        # плавный подвод к первой точке
        q_now = np.array([sim.getJointPosition(j) for j in joints])
        q_first = Q_SIGN * Q[0] + Q_OFFSET
        for a in np.linspace(0, 1, 60):
            for j, v in zip(joints, (1 - a) * q_now + a * q_first):
                sim.setJointTargetPosition(j, float(v))
            sim.step()
        # окружность
        for q in Q:
            for j, v in zip(joints, Q_SIGN * q + Q_OFFSET):
                sim.setJointTargetPosition(j, float(v))
            t_end = sim.getSimulationTime() + DT_POINT
            while sim.getSimulationTime() < t_end:
                sim.step()
            trace.append(sim.getObjectPosition(tip, sim.handle_world))
    finally:
        sim.stopSimulation()
    trace = np.array(trace)
    np.savetxt(os.path.join(here, 'tip_trace.csv'), trace, delimiter=',', header='x,y,z', comments='')
    print('записано точек: %d -> tip_trace.csv (сравните с окружностью в отчёте)' % len(trace))


if __name__ == '__main__':
    main()
