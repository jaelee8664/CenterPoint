import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
from .EKF_CV import ExtendedKalmanFilter
from .IMM import IMMEstimator
import lap
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix):
    """
    Hungarian Algorithm
    """
    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) *
              (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) *
              (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o

def l2norm_batch(dist1, dist2):
    """
    Args:
        dist: 3D 위치 [N x 3]
    Returns:
        dist1, dist2간의 Bire Eye View에서의 이차원 L2 norm matrix
    """
    dist2 = np.expand_dims(dist2, 0)
    dist1 = np.expand_dims(dist1, 1)
    
    norm_matrix = np.power(dist1[..., 0] - dist2[..., 0], 2) + np.power(dist1[..., 1] - dist2[..., 1], 2)
    norm_matrix = np.sqrt(norm_matrix)
    return norm_matrix

"""
Tracker class
"""
class Constvel():
    def __init__(self, pos, t=0):
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=3)
        # [x,y,z,dot_x,dot_y,dot_z,acc_x,acc_y,acc_z, rot]
        self.kf.F = np.array([[1, 0, 0, t, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, t, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, t, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.R *= 100.  # (uncertainty/noise of measurement)
        self.kf.P[3:, 3:] *= 100.
        self.kf.P *= 10.
        # (uncertainty/noise of propagation)
        self.kf.Q[3:6, 3:6] *= 40.*(60**3*t**3)
        self.kf.Q[6:10, 6:10] *= 0.
        if pos is not None:
            self.kf.x = np.concatenate((pos.reshape((3, 1)),
                                        np.zeros(6).reshape((6, 1)),
                                        np.ones(1).reshape((1, 1))),
                                       axis=0).reshape((10, 1))
        else:
            self.kf.x = np.zeros(10).reshape((10, 1))

    def update(self, measurement):
        self.kf.update(measurement)

    def predict(self, t):
        self.kf.F = np.array([[1, 0, 0, t, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, t, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, t, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.predict(F=self.kf.F)


class Constacc():
    def __init__(self, pos, t=0):
        self.kf = KalmanFilter(dim_x=10, dim_z=3)
        # [x,y,z,dot_x,dot_y,dot_z,acc_x,acc_y,acc_z, rot]
        self.kf.F = np.array([[1, 0, 0, t, 0, 0, 0.5*(t**2), 0, 0, 0],
                              [0, 1, 0, 0, t, 0, 0, 0.5*(t**2), 0, 0],
                              [0, 0, 1, 0, 0, t, 0, 0.5*(t**2), 0, 0],
                              [0, 0, 0, 1, 0, 0, t, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, t, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, t, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.R *= 100.  # (uncertainty/noise of measurement)
        self.kf.P[3:, 3:] *= 100.
        self.kf.P *= 10.
        # (uncertainty/noise of propagation)
        self.kf.Q[3:6, 3:6] *= 40.*(60**3*t**3)
        self.kf.Q[6:9, 6:9] *= 40.*(60*t)
        self.kf.Q[9, 9] *= 0.
        if pos is not None:
            self.kf.x = np.concatenate((pos.reshape((3, 1)),
                                        np.zeros(6).reshape((6, 1)),
                                        np.ones(1).reshape((1, 1))),
                                       axis=0).reshape((10, 1))
        else:
            self.kf.x = np.zeros(10).reshape((10, 1))

    def update(self, measurement):
        self.kf.update(measurement)

    def predict(self, t):
        self.kf.F = np.array([[1, 0, 0, t, 0, 0, 0.5*(t**2), 0, 0, 0],
                              [0, 1, 0, 0, t, 0, 0, 0.5*(t**2), 0, 0],
                              [0, 0, 1, 0, 0, t, 0, 0.5*(t**2), 0, 0],
                              [0, 0, 0, 1, 0, 0, t, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, t, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, t, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.predict(F=self.kf.F)

class Constantturnrate():
    """
    consant turnrate model corresponding to 3d position
    """
    def __init__(self, pos, t=0):
        # CT EKF는 filters폴더의 EKF_CV 참고
        self.kf = ExtendedKalmanFilter(dim_x=10, dim_z=3)
        if pos is not None:
            self.kf.x = np.concatenate((pos.reshape((3, 1)),
                                        np.zeros(6).reshape((6, 1)),
                                        np.ones(1).reshape((1, 1))),
                                       axis=0).reshape((10, 1))
        else:
            self.kf.x = np.zeros(10).reshape((10, 1))
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        self.kf.P[3:, 3:] *= 100.
        self.kf.P *= 10.
        self.kf.Q[3:6, 3:6] *= 40.*(60**3*t**3)
        self.kf.Q[6:9, 6:9] *= 0.
        self.kf.Q[9, 9] *= 0.002

    def predict(self, t):
        """
        prediction step of filtering
        """
        self.kf.predict(t)

    def update(self, measurement):
        """
        update step of filtering
        """
        self.kf.update(measurement)


class IMM_Combiner():
    """
    IMM corresponding to position
    """
    def __init__(self, filters, id_):
        self.num_filts = len(filters)
        self.mu = np.array([0.34, 0.33, 0.33])
        self.trans = np.array([[0.9, 0.08, 0.05],
                               [0.15, 0.7, 0.15],
                               [0.04, 0.16, 0.8]])
        self.id = id_
        self.immest = IMMEstimator(filters, self.mu,
                                         self.trans)

    def predict(self, t):
        """
        prediction step of IMM
        """
        self.immest.predict(t)

    def update(self, z):
        """
        update step of IMM
        """
        self.immest.update(z)

    def get_state(self):
        """
        return immstate
        """
        return self.immest.x.flatten()


class Track():
    count = 0

    def __init__(self, pos, t=0):
        """
        Args:
            pos (np.array): np.array([x, y, z]) measurement
            t (int, optional): 현재 타임스텝
        """
        self.time_since_update = 0 # 마지막 update step 이후 prediction step이 진행된 횟수 
        self.id = Track.count # ID 부여
        Track.count += 1
        self.history = deque() # 과거 state
        self.hit_cnt = 8 # 보관할 과거 state 개수
        self.hits = 0 # 매칭된 총 횟수
        self.hit_streak = 0 # measurement가 끊기지 않고 연속해서 들어온 횟수
        self.age = 0 # 트랙의 나이
        self.predicted_state = None # IMM의 prediction step에서 예측된 state
        self.last_observation = pos # 마지막으로 매칭된 measurement

        self.pos_filter1 = Constvel(pos, t) # 등속도 모델
        self.pos_filter2 = Constacc(pos, t) # 등가속도 모델
        self.pos_filter3 = Constantturnrate(pos, t) # 등각속도 모델
        self.pos_filters = [self.pos_filter1,
                            self.pos_filter2, self.pos_filter3]
        self.pos_imm = IMM_Combiner(self.pos_filters, self.id) # IMM 관리 Class

    def update(self, pos):
        self.pos = pos
        if pos is not None:
            self.last_observation = pos
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.pos_imm.update(pos)
        else:
            self.last_observation = None
            self.pos_imm.update(None)

        # history 관리
        self.history.append(self.get_state())
        if len(self.history) > self.hit_cnt:
            self.history.popleft()

        return self.get_state()

    def predict(self, t):
        self.pos_imm.predict(t=t)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.predicted_state = self.get_state()
        return self.predicted_state

    def get_state(self):
        return self.pos_imm.get_state()
    
    def get_history(self):
        return self.history