# -*- coding: utf-8 -*-
"""kinlib — учебная библиотека кинематики манипуляторов для демонстраций на занятиях 2–3.

Основа — классы Frame / Robot / RobotSerial / RobotTrajectory из ноутбука
«ПЗК_и_ОЗК_на_Python.ipynb» (визуализация на matplotlib, без внешних робототехнических
пакетов). Добавлены помощники для плоских 2R/3R, рабочих зон, коэффициента сервиса,
численного якобиана, эллипсов и многоугольников допустимых скоростей.

Обозначения по учебнику Зенкевича — Ющенко: q — обобщённые координаты, A_i — матрицы
Денавита — Хартенберга, T_i = A_1…A_i, J — матрица Якоби.
ВНИМАНИЕ: в классе RobotSerial якобиан хранится в порядке [v; ω] (сначала линейная,
затем угловая скорость), а в книге — [ω; v]. См. функцию jacobian_book().
"""
import logging
from abc import abstractmethod
from math import sin as s, cos as c, pi

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

NAVY, AMB, TEAL, RED, GREY, ICE, GREEN = '#1E2761', '#D97706', '#167D8D', '#B85042', '#5A6478', '#CADCFC', '#2C7A5A'


# ============================================================== Frame
class Frame:
    """Однородная матрица 4×4 положения системы координат."""

    def __init__(self, t_4_4):
        self.t_4_4 = np.asarray(t_4_4, dtype=float)

    def __mul__(self, other):
        if isinstance(other, Frame):
            return Frame(np.dot(self.t_4_4, other.t_4_4))
        return Frame(np.dot(self.t_4_4, other))

    def __getitem__(self, key):
        return self.t_4_4[key]

    def __str__(self):
        return np.array2string(self.t_4_4, precision=3, suppress_small=True)

    __repr__ = __str__

    @property
    def inv(self):
        t = self.t_4_4.copy()
        t[0:3, 3:4] = -self.t_4_4[0:3, 0:3].T.dot(self.t_4_4[0:3, 3:4])
        t[0:3, 0:3] = self.t_4_4[0:3, 0:3].T
        return Frame(t)

    @property
    def copy(self):
        return Frame(self.t_4_4.copy())

    @property
    def z_3_1(self):
        return self.t_4_4[0:3, 2:3]

    @property
    def t_3_1(self):
        return self.t_4_4[0:3, 3:4]

    @property
    def r_3_3(self):
        return self.t_4_4[0:3, 0:3]

    @property
    def q_4(self):
        return Rotation.from_matrix(self.r_3_3).as_quat()

    @property
    def r_3(self):
        return Rotation.from_matrix(self.r_3_3).as_rotvec()

    @property
    def euler_3(self):
        return Rotation.from_matrix(self.r_3_3).as_euler("ZYX", degrees=False)

    @staticmethod
    def from_r_3_3(r_3_3, t_3_1):
        t = np.eye(4)
        t[0:3, 0:3] = r_3_3
        t[0:3, 3:4] = np.reshape(t_3_1, (3, 1))
        return Frame(t)

    @staticmethod
    def from_q_4(q_4, t_3_1):
        return Frame.from_r_3_3(Rotation.from_quat(q_4).as_matrix(), t_3_1)

    @staticmethod
    def from_r_3(r_3, t_3_1):
        return Frame.from_r_3_3(Rotation.from_rotvec(r_3).as_matrix(), t_3_1)

    @staticmethod
    def from_euler_3(euler_3, t_3_1):
        return Frame.from_r_3_3(Rotation.from_euler("ZYX", euler_3, degrees=False).as_matrix(), t_3_1)

    @staticmethod
    def from_dh(dh_params):
        """Классические параметры ДХ (d, a, alpha, theta) — формула (2.8) учебника."""
        d, a, alpha, theta = dh_params
        return Frame(np.array([[c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
                               [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
                               [0., s(alpha), c(alpha), d],
                               [0., 0., 0., 1.]]))

    @staticmethod
    def from_dh_modified(dh_params):
        d, a, alpha, theta = dh_params
        return Frame(np.array([[c(theta), -s(theta), 0, a],
                               [s(theta) * c(alpha), c(theta) * c(alpha), -s(alpha), -s(alpha) * d],
                               [s(theta) * s(alpha), c(theta) * s(alpha), c(alpha), c(alpha) * d],
                               [0., 0., 0., 1.]]))

    @staticmethod
    def i_4_4():
        return Frame(np.eye(4))

    def distance_to(self, other):
        return np.linalg.norm(self.t_3_1 - other.t_3_1, ord=2)

    def angle_to(self, other):
        return np.linalg.norm(Rotation.from_matrix(self.r_3_3.T.dot(other.r_3_3)).as_rotvec(), ord=2)


# ============================================================== Robot
class Robot(object):
    def __init__(self, params, initial_offset, plot_xlim=(-0.5, 0.5), plot_ylim=(-0.5, 0.5), plot_zlim=(0.0, 1.0),
                 ws_lim=None, ws_division=5):
        self.params = params
        self.initial_offset = initial_offset
        self.axis_values = np.zeros(initial_offset.shape, dtype=np.float64)
        self.is_reachable_inverse = True
        self.plot_xlim, self.plot_ylim, self.plot_zlim = plot_xlim, plot_ylim, plot_zlim
        self.figure = None
        self.ax = None
        self.ws_lim = np.array([[-pi, pi]] * self.num_axis) if ws_lim is None else np.asarray(ws_lim, dtype=float)
        self.ws_division = ws_division

    # фигура создаётся лениво — иначе в ноутбуке появляется пустой холст
    def _ensure_fig(self, figsize=(6, 6)):
        if self.figure is None or not plt.fignum_exists(self.figure.number):
            self.figure = plt.figure(figsize=figsize)
            self.ax = self.figure.add_subplot(111, projection="3d")
        return self.ax

    @property
    def num_axis(self):
        return self.initial_offset.shape[0]

    @property
    @abstractmethod
    def end_frame(self):
        pass

    @property
    @abstractmethod
    def jacobian(self):
        pass

    def forward(self, theta_x):
        self.axis_values = np.asarray(theta_x, dtype=float).copy()
        return self.end_frame

    @abstractmethod
    def inverse(self, end_frame):
        pass

    def workspace(self):
        """Прямая задача в узлах равномерной сетки внутри S_q (с. 84 учебника)."""
        num_points = self.ws_division ** self.num_axis
        lower = self.ws_lim[:, 0]
        intervals = (self.ws_lim[:, 1] - self.ws_lim[:, 0]) / (self.ws_division - 1)
        points = np.zeros([num_points, 3])
        idx = np.zeros([self.num_axis], dtype=np.int32)
        for i in range(num_points):
            points[i] = self.forward(lower + idx * intervals).t_3_1.flatten()
            idx[0] += 1
            for k in range(self.num_axis):
                if idx[k] >= self.ws_division:
                    if k >= self.num_axis - 1:
                        break
                    idx[k] = 0
                    idx[k + 1] += 1
                else:
                    break
        return points

    def workspace_random(self, n=20000, seed=0):
        """Метод Монте-Карло: n случайных q внутри ограничений."""
        rng = np.random.RandomState(seed)
        Q = rng.uniform(self.ws_lim[:, 0], self.ws_lim[:, 1], size=(n, self.num_axis))
        pts = np.array([self.forward(q).t_3_1.flatten() for q in Q])
        return pts, Q

    def plot_settings(self):
        self.ax.set_xlim(self.plot_xlim)
        self.ax.set_ylim(self.plot_ylim)
        self.ax.set_zlim(self.plot_zlim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    @abstractmethod
    def draw(self):
        pass

    def draw_ws(self, n=5000, s=2):
        self._ensure_fig()
        self.plot_settings()
        points, _ = self.workspace_random(n)
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="red", marker=".", s=s, alpha=0.3)

    def show(self, body=True, ws=False):
        self._ensure_fig()
        if body:
            self.draw()
        if ws:
            self.draw_ws()
        plt.show()


def simplify_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def simplify_angles(angles):
    for i in range(angles.shape[0]):
        angles[i] = simplify_angle(angles[i])
    return angles


# ============================================================== RobotSerial
class RobotSerial(Robot):
    """Разомкнутая цепь по таблице ДХ dh_params [n,4] = | d | a | alpha | theta0 |."""

    def __init__(self, dh_params, dh_type="normal", analytical_inv=None,
                 plot_xlim=(-0.5, 0.5), plot_ylim=(-0.5, 0.5), plot_zlim=(0.0, 1.0),
                 ws_lim=None, ws_division=5,
                 inv_m="jac_pinv", step_size=5e-1, max_iter=300, final_loss=1e-4, joint_types=None):
        dh_params = np.asarray(dh_params, dtype=float)
        # типы сочленений: 'R' — вращательное (переменная q_i = theta_i), 'P' — телескопическое (переменная d_i)
        self.joint_types = list(joint_types) if joint_types is not None else ['R'] * dh_params.shape[0]
        if len(self.joint_types) != dh_params.shape[0] or any(t not in 'RP' for t in self.joint_types):
            raise Exception("joint_types: строка или список из 'R' и 'P' по числу сочленений")
        Robot.__init__(self, dh_params[:, 0:3], dh_params[:, 3], plot_xlim, plot_ylim, plot_zlim, ws_lim, ws_division)
        if dh_type not in ("normal", "modified"):
            raise Exception('dh_type can only be "normal" or "modified"')
        self.dh_type = dh_type
        self.analytical_inv = analytical_inv
        if inv_m not in ("jac_t", "jac_pinv", "jac_dls"):
            raise Exception('inv_m can only be "jac_t", "jac_pinv" or "jac_dls"')
        self.inv_m = inv_m
        self.step_size = step_size
        self.max_iter = max_iter
        self.final_loss = final_loss
        self.dls_lambda = 0.1
        self.inv_history = []

    @property
    def dh_params(self):
        dh = np.hstack((self.params, self.initial_offset.reshape([self.num_axis, 1]))).astype(float)
        for i, t in enumerate(self.joint_types):
            if t == 'R':
                dh[i, 3] += self.axis_values[i]      # обобщённая координата — угол theta_i
            else:
                dh[i, 0] += self.axis_values[i]      # обобщённая координата — смещение d_i
        return dh

    @property
    def ts(self):
        dh = self.dh_params
        f = Frame.from_dh if self.dh_type == "normal" else Frame.from_dh_modified
        return [f(dh[i]) for i in range(self.num_axis)]

    @property
    def axis_frames(self):
        """Список T_1 … T_N (положения звеньев в базовой системе)."""
        fs = []
        f = Frame.i_4_4()
        for t in self.ts:
            f = f * t
            fs.append(f)
        return fs

    @property
    def end_frame(self):
        return self.axis_frames[-1]

    @property
    def jacobian(self):
        """Якобиан 6×N в порядке [v; ω] — столбцы (3.62) учебника."""
        fs = self.axis_frames
        jac = np.zeros([6, self.num_axis])
        if self.dh_type == "normal":
            pN = fs[-1].t_3_1.reshape(3)
            for i in range(self.num_axis):
                z = np.array([0., 0., 1.]) if i == 0 else fs[i - 1].z_3_1.reshape(3)
                o = np.zeros(3) if i == 0 else fs[i - 1].t_3_1.reshape(3)
                if self.joint_types[i] == 'R':
                    jac[0:3, i] = np.cross(z, pN - o)     # вращательное: (z × p; z)
                    jac[3:6, i] = z
                else:
                    jac[0:3, i] = z                        # телескопическое: (z; 0)
        else:
            for i in range(self.num_axis):
                z = fs[i].z_3_1.reshape(3)
                if self.joint_types[i] == 'R':
                    jac[0:3, i] = np.cross(z, (fs[-1].t_3_1 - fs[i].t_3_1).reshape(3))
                    jac[3:6, i] = z
                else:
                    jac[0:3, i] = z
        return jac

    def jacobian_book(self):
        """Якобиан в порядке учебника: сначала ω, потом v."""
        J = self.jacobian
        return np.vstack([J[3:6], J[0:3]])

    def pose_error(self, end_frame):
        """Невязка e = (Δp; δφ) между текущим положением схвата и целью (в базовой СК)."""
        end = self.end_frame
        dx = np.zeros([6, 1])
        dx[0:3, 0] = (end_frame.t_3_1 - end.t_3_1).reshape(3)
        diff = end.inv * end_frame
        dx[3:6, 0] = end.r_3_3.dot(diff.r_3.reshape([3, 1])).reshape(3)
        return dx

    def inverse(self, end_frame, q0=None):
        if self.analytical_inv is not None:
            return self.inverse_analytical(end_frame, self.analytical_inv)
        return self.inverse_numerical(end_frame, q0=q0)

    def inverse_analytical(self, end_frame, method):
        self.is_reachable_inverse, theta_x = method(self.dh_params, end_frame)
        self.forward(theta_x)
        return theta_x

    def inverse_numerical(self, end_frame, q0=None, method=None, step_size=None, max_iter=None, verbose=False):
        """Итерационная ОЗК: q_{k+1} = q_k + α·M(q_k)·e_k, где M = J⁺, Jᵀ или демпфированная J⁺.
        История норм невязки сохраняется в self.inv_history."""
        method = method or self.inv_m
        step = self.step_size if step_size is None else step_size
        iters = self.max_iter if max_iter is None else max_iter
        if q0 is not None:
            self.forward(q0)
        self.inv_history = []
        last_dx = np.zeros([6, 1])
        for k in range(iters):
            J = self.jacobian
            if method == "jac_t":
                M = J.T
            elif method == "jac_dls":
                M = J.T @ np.linalg.inv(J @ J.T + self.dls_lambda ** 2 * np.eye(6))
            else:
                M = np.linalg.pinv(J)
            dx = self.pose_error(end_frame)
            self.inv_history.append(float(np.linalg.norm(dx)))
            if np.linalg.norm(dx) < self.final_loss or np.linalg.norm(dx - last_dx) < 0.1 * self.final_loss:
                for i, t in enumerate(self.joint_types):
                    if t == 'R':
                        self.axis_values[i] = simplify_angle(self.axis_values[i])
                self.is_reachable_inverse = np.linalg.norm(dx) < self.final_loss
                if verbose:
                    print("итераций: %d, невязка %.2e, достижимо: %s" % (k + 1, np.linalg.norm(dx), self.is_reachable_inverse))
                return self.axis_values
            dq = step * M.dot(dx)
            self.forward(self.axis_values + dq.reshape(self.num_axis))
            last_dx = dx
        logging.error("Pose cannot be reached!")
        self.is_reachable_inverse = False
        return self.axis_values

    def inverse_position(self, p_target, q0=None, step_size=0.5, max_iter=300, tol=1e-8):
        """Итерационная ОЗК только по положению схвата: q ← q + α·J_v⁺·(p* − p). История — в inv_history."""
        if q0 is not None:
            self.forward(q0)
        p_target = np.asarray(p_target, dtype=float).reshape(3)
        self.inv_history = []
        for _ in range(max_iter):
            e = p_target - self.end_frame.t_3_1.reshape(3)
            self.inv_history.append(float(np.linalg.norm(e)))
            if np.linalg.norm(e) < tol:
                break
            dq = step_size * np.linalg.pinv(self.jacobian[0:3]) @ e
            self.forward(self.axis_values + dq)
        self.is_reachable_inverse = self.inv_history[-1] < max(tol, 1e-6)
        return self.axis_values

    def draw(self, ax=None, color=NAVY, alpha=1.0, frame_len=0.2, cylinders=True):
        if ax is None:
            ax = self._ensure_fig()
            ax.clear()
        else:
            self.ax = ax
        self.plot_settings()
        fs = self.axis_frames
        x, y, z = [0.], [0.], [0.]
        for f in fs:
            x.append(f.t_3_1[0, 0]); y.append(f.t_3_1[1, 0]); z.append(f.t_3_1[2, 0])
        ax.plot(x, y, z, color=color, lw=3, alpha=alpha)
        ax.scatter(x[1:], y[1:], z[1:], c="red", marker="o", alpha=alpha)
        if cylinders:
            cy_radius = np.amax(self.params[:, 0:2]) * 0.05
            cy_len = cy_radius * 4.
            cy_div = 5
            theta = np.linspace(0, 2 * np.pi, cy_div)
            cx = np.array([cy_radius * np.cos(theta)])
            cz = np.array([-0.5 * cy_len, 0.5 * cy_len])
            cx, cz = np.meshgrid(cx, cz)
            cy = np.array([cy_radius * np.sin(theta)] * 2)
            points = np.zeros([3, cy_div * 2])
            points[0], points[1], points[2] = cx.flatten(), cy.flatten(), cz.flatten()
            ax.plot_surface(points[0].reshape(2, cy_div), points[1].reshape(2, cy_div), points[2].reshape(2, cy_div),
                            color="pink", rstride=1, cstride=1, linewidth=0, alpha=0.4 * alpha)
            for i in range(self.num_axis - 1):
                f = fs[i]
                pf = f.r_3_3.dot(points) + f.t_3_1
                ax.plot_surface(pf[0].reshape(2, cy_div), pf[1].reshape(2, cy_div), pf[2].reshape(2, cy_div),
                                color="pink", rstride=1, cstride=1, linewidth=0, alpha=0.4 * alpha)
        f = fs[-1].t_4_4
        for k, col in enumerate(("red", "green", "blue")):
            ax.plot([f[0, 3], f[0, 3] + frame_len * f[0, k]], [f[1, 3], f[1, 3] + frame_len * f[1, k]],
                    [f[2, 3], f[2, 3] + frame_len * f[2, k]], color=col, alpha=alpha)
        return ax


# ============================================================== RobotTrajectory
class RobotTrajectory(object):
    def __init__(self, robot, frames, time_points=None, rot_tran_ratio=2.0):
        self.robot = robot
        if len(frames) < 2:
            raise Exception("trajectory must include at least 2 frames")
        if time_points is not None and len(frames) != len(time_points):
            raise Exception("time_points should have same length as frames")
        self.frames = frames
        self.time_points = None if time_points is None else np.asarray(time_points, dtype=float)
        self.rot_tran_ratio = rot_tran_ratio

    def __len__(self):
        return len(self.frames)

    @property
    def len_segs_tran(self):
        return np.array([self.frames[i].distance_to(self.frames[i + 1]) for i in range(len(self) - 1)])

    @property
    def len_segs_rot(self):
        return np.array([self.frames[i].angle_to(self.frames[i + 1]) for i in range(len(self) - 1)])

    @property
    def len_segs(self):
        return self.len_segs_rot * self.rot_tran_ratio / 2. / np.pi + self.len_segs_tran

    def interpolate(self, num_segs, motion="p2p", method="linear"):
        lengths = self.len_segs if self.time_points is None else self.time_points[1:] - self.time_points[:-1]
        length_total = np.sum(lengths)
        tra_array = np.zeros([len(self), max(self.robot.num_axis, 6)])
        for i in range(len(self)):
            if motion == "p2p":
                self.robot.inverse(self.frames[i])
                tra_array[i, 0:self.robot.num_axis] = self.robot.axis_values
            if motion == "lin":
                tra_array[i, 0:3] = self.frames[i].t_3_1.reshape(3)
                tra_array[i, 3:6] = self.frames[i].euler_3
        inter_values = np.zeros([num_segs + 1, self.robot.num_axis])
        inter_time = np.zeros([num_segs + 1])
        for progress in range(num_segs + 1):
            index = 0
            p_temp = progress * length_total / num_segs
            for i in range(lengths.shape[0]):
                if p_temp - lengths[i] > 1e-5:
                    p_temp -= lengths[i]
                    index += 1
                else:
                    break
            p_temp /= lengths[index]
            if motion == "p2p":
                inter_values[progress] = tra_array[index, 0:self.robot.num_axis] * (1 - p_temp) + tra_array[index + 1, 0:self.robot.num_axis] * p_temp
            if motion == "lin":
                xyzabc = tra_array[index, 0:6] * (1 - p_temp) + tra_array[index + 1, 0:6] * p_temp
                self.robot.inverse(Frame.from_euler_3(xyzabc[3:6], xyzabc[0:3].reshape([3, 1])))
                inter_values[progress] = self.robot.axis_values
            inter_time[progress] = np.sum(lengths[0:index]) + lengths[index] * p_temp
        return inter_values, inter_time

    def show(self, num_segs=100, motion="p2p", method="linear", snapshots=(0.0, 0.5, 1.0)):
        """Статическая версия для ноутбука: несколько «кадров» движения и след схвата."""
        inter_values, inter_time = self.interpolate(num_segs, motion, method)
        x, y, z = [], [], []
        for i in range(num_segs + 1):
            self.robot.forward(inter_values[i])
            p = self.robot.end_frame.t_3_1
            x.append(p[0, 0]); y.append(p[1, 0]); z.append(p[2, 0])
        ax = self.robot._ensure_fig()
        ax.clear()
        for k, sfrac in enumerate(snapshots):
            self.robot.forward(inter_values[int(round(sfrac * num_segs))])
            self.robot.draw(ax=ax, alpha=0.35 + 0.65 * k / max(1, len(snapshots) - 1), cylinders=(k == len(snapshots) - 1))
        ax.plot(x, y, z, color="lightblue", lw=2)
        plt.show()
        return inter_values, inter_time


# ============================================================== вспомогательные функции
def skew(w):
    """Λ(ω) — кососимметрическая матрица (3.11): skew(w) @ r == np.cross(w, r)."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(S):
    """Обратная к skew: вектор из кососимметрической матрицы (усреднение по симметричным элементам)."""
    S = np.asarray(S, dtype=float)
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2.0


def jacobian_numeric(robot, q, eps=1e-6):
    """Якобиан [v; ω] конечными разностями по прямой задаче — проверка формулы (3.62)."""
    q = np.asarray(q, dtype=float)
    T0 = robot.forward(q.copy())
    p0, R0 = T0.t_3_1.reshape(3).copy(), T0.r_3_3.copy()
    J = np.zeros((6, robot.num_axis))
    for i in range(robot.num_axis):
        q1 = q.copy(); q1[i] += eps
        T1 = robot.forward(q1)
        J[0:3, i] = (T1.t_3_1.reshape(3) - p0) / eps
        dR = (T1.r_3_3 - R0) / eps
        J[3:6, i] = vee(dR @ R0.T)
    robot.forward(q)
    return J


# ----- плоские механизмы
def fk2r(q, l=(1.0, 1.0)):
    """Плоский 2R: положение схвата и угол ориентации (пример 2.1)."""
    q1, q2 = q
    x = l[0] * np.cos(q1) + l[1] * np.cos(q1 + q2)
    y = l[0] * np.sin(q1) + l[1] * np.sin(q1 + q2)
    return np.array([x, y]), q1 + q2


def ik2r(x, y, l=(1.0, 1.0)):
    """Плоский 2R: обе ветви (локоть вверх/вниз); пустой список, если цель недостижима (пример 2.10)."""
    l1, l2 = l
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(c2) > 1:
        return []
    sols = []
    for sgn in (1, -1):
        q2 = sgn * np.arccos(np.clip(c2, -1, 1))
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        sols.append(np.array([q1, q2]))
    return sols


def jac2r(q, l=(1.0, 1.0)):
    """Блок J_v плоского 2R (пример 3.1)."""
    q1, q2 = q
    return np.array([[-l[0] * np.sin(q1) - l[1] * np.sin(q1 + q2), -l[1] * np.sin(q1 + q2)],
                     [l[0] * np.cos(q1) + l[1] * np.cos(q1 + q2), l[1] * np.cos(q1 + q2)]])


def fk_planar(q, l):
    """Плоская цепь из n вращательных звеньев: точки сочленений [n+1, 2] и угол ориентации."""
    q = np.asarray(q, dtype=float); l = np.asarray(l, dtype=float)
    a = np.cumsum(q)
    pts = np.zeros((len(q) + 1, 2))
    for i in range(len(q)):
        pts[i + 1] = pts[i] + l[i] * np.array([np.cos(a[i]), np.sin(a[i])])
    return pts, a[-1]


def jac_planar(q, l):
    """Якобиан плоской цепи: строки (ẋ, ẏ, φ̇)."""
    q = np.asarray(q, dtype=float); l = np.asarray(l, dtype=float)
    a = np.cumsum(q)
    n = len(q)
    J = np.zeros((3, n))
    for j in range(n):
        J[0, j] = -np.sum(l[j:] * np.sin(a[j:]))
        J[1, j] = np.sum(l[j:] * np.cos(a[j:]))
        J[2, j] = 1.0
    return J


def draw_arm2d(ax, q, l, color=NAVY, lw=4, alpha=1.0, ls='-', label=None, joints=True):
    pts, _ = fk_planar(q, l)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha, ls=ls, solid_capstyle='round', label=label)
    if joints:
        ax.plot(pts[:-1, 0], pts[:-1, 1], 'o', color='white', mec=color, mew=1.5, ms=7, alpha=alpha)
        ax.plot(pts[-1, 0], pts[-1, 1], 's', color=RED, ms=7, alpha=alpha)
    return pts


def workspace_2r(l=(1.0, 0.7), lim1=(-np.pi, np.pi), lim2=(-np.pi, np.pi), n=200):
    """Точки рабочей зоны плоского 2R на сетке q (пример 2.5)."""
    Q1 = np.linspace(lim1[0], lim1[1], n); Q2 = np.linspace(lim2[0], lim2[1], n)
    A, B = np.meshgrid(Q1, Q2)
    X = l[0] * np.cos(A) + l[1] * np.cos(A + B)
    Y = l[0] * np.sin(A) + l[1] * np.sin(A + B)
    return X.ravel(), Y.ravel()


def area_from_points(x, y, delta):
    """Оценка площади зоны по числу занятых ячеек решётки: S ≈ M·δ² (аналог V ≈ Mδ³, с. 84)."""
    ix = np.floor(x / delta).astype(int); iy = np.floor(y / delta).astype(int)
    cells = set(zip(ix.tolist(), iy.tolist()))
    return len(cells) * delta ** 2


def service_coefficient_3r(x, y, l=(1.0, 0.6, 0.4), nphi=360):
    """Коэффициент сервиса Ω_r плоского 3R без ограничений (пример 2.8):
    доля направлений оси схвата φ, при которых точка O2 = r − l3(cos φ, sin φ) лежит в кольце [l1−l2, l1+l2]."""
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    wx = x - l[2] * np.cos(phis); wy = y - l[2] * np.sin(phis)
    rr = np.hypot(wx, wy)
    ok = (rr <= l[0] + l[1] + 1e-12) & (rr >= abs(l[0] - l[1]) - 1e-12)
    return ok.mean()


def service_map_3r(l=(1.0, 0.6, 0.4), N=200, nphi=180):
    R = sum(l) * 1.05
    xs = np.linspace(-R, R, N)
    X, Y = np.meshgrid(xs, xs)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    cnt = np.zeros_like(X)
    for ph in phis:
        rr = np.hypot(X - l[2] * np.cos(ph), Y - l[2] * np.sin(ph))
        cnt += ((rr <= l[0] + l[1]) & (rr >= abs(l[0] - l[1]))).astype(float)
    return xs, cnt / nphi


def service_average_3r(l=(3.0, 2.0, 1.0), N=300, nphi=180):
    """Сервис манипулятора Ω = (1/V)∫Ω_r dV по кругу радиуса l1+l2+l3 (формула (2.19))."""
    xs, Om = service_map_3r(l, N, nphi)
    X, Y = np.meshgrid(xs, xs)
    inside = np.hypot(X, Y) <= sum(l)
    return Om[inside].mean()


# ----- допустимые скорости
def polygon_image(J, C):
    """Образ параллелепипеда |q̇_i| ≤ C_i при отображении v = J q̇ (J — 2×n): вершины выпуклой оболочки."""
    from itertools import product
    from scipy.spatial import ConvexHull
    C = np.asarray(C, dtype=float)
    verts = np.array([J @ (np.array(sg) * C) for sg in product((-1, 1), repeat=len(C))])
    if J.shape[0] == 2:
        hull = ConvexHull(verts)
        return verts[hull.vertices]
    return verts


def ellipse_image(J, C, n=200):
    """Образ круга ‖q̇‖ ≤ C при отображении v = J q̇ (эллипс допустимых скоростей (3.185))."""
    t = np.linspace(0, 2 * np.pi, n)
    if J.shape[1] == 2:
        qd = C * np.vstack([np.cos(t), np.sin(t)])
        return (J @ qd).T
    # для n>2: эллипс — образ единичного шара, берём главные оси через SVD
    U, S, Vt = np.linalg.svd(J[:2], full_matrices=False)
    pts = (U[:, :2] * (C * S[:2])) @ np.vstack([np.cos(t), np.sin(t)])
    return pts.T


def sphere_radius(J, C):
    """Радиус сферы (3.184): √λ*·C, где λ* — наибольшее собственное число JᵀJ."""
    lam = np.linalg.eigvalsh(J.T @ J)
    return np.sqrt(lam.max()) * float(np.linalg.norm(C)), lam


def manipulability(J):
    """Мера манипулятивности Йошикавы √det(JJᵀ) (пропорциональна объёму эллипсоида)."""
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))


# ----- пример робота из ноутбука (6R, размеры типа UR10)
UR_DH = np.array([[0.163, 0., 0.5 * pi, 0.],
                  [0., 0.632, pi, 0.5 * pi],
                  [0., 0.6005, pi, 0.],
                  [0.2013, 0., -0.5 * pi, -0.5 * pi],
                  [0.1025, 0., 0.5 * pi, 0.],
                  [0.094, 0., 0., 0.]])


def make_ur():
    return RobotSerial(UR_DH, plot_xlim=(-1.3, 1.3), plot_ylim=(-1.3, 1.3), plot_zlim=(-0.6, 1.5))
