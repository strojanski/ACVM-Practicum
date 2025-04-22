import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from ex4_utils import kalman_step, gaussian_prob

def run_kalman_rw(x, y, q, r):
    state = np.array([x[0], y[0]], dtype=np.float32)
    A = np.eye(2, dtype=np.float32)
    C = np.eye(2, dtype=np.float32)
    Q = q * np.eye(2, dtype=np.float32)
    R = r * np.eye(2, dtype=np.float32)
    
    sx = np.zeros((x.size, 1) , dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1) , dtype=np.float32).flatten()
    sx[0] = x[0]
    sy[0] = y[0]
    covariance = np.eye(A.shape[0] , dtype=np.float32)
    
    sx, sy = [x[0]], [y[0]]
    
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q , R, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
        # sx[j] = state[0]
        # sy[j] = state[1]
        
        sx.append(state[0,0])
        sy.append(state[1,0])
        
    return sx, sy

def run_kalman_ncv(x, y, q, r):
    dt = 1
    # q = 0.1
    # r = 1.0
    
    A = np.eye(4, dtype=np.float32)
    A[0, 2] = dt
    A[1, 3] = dt
    
    C = np.zeros((2, 4), dtype=np.float32)
    C[0, 0] = 1
    C[1, 1] = 1    
    
    Q = q * np.array([
        [dt**3/3,        0, dt**2/2,        0],
        [       0, dt**3/3,        0, dt**2/2],
        [dt**2/2,        0,      dt,        0],
        [       0, dt**2/2,        0,      dt],
    ], dtype=float)
    
    F = np.zeros((4, 4), dtype=np.float32)
    F[0, 2] = 1
    F[1, 3] = 1
    
    R = r * np.eye(2, dtype=np.float32)
    
    sx = np.zeros((x.size , 1) , dtype=np.float32) .flatten()
    sy = np.zeros((y.size , 1) , dtype=np.float32) .flatten()
    sx [0] = x[0]
    sy [0] = y[0]
    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state [0] = x[0]
    state [1] = y[0]
    covariance = np.eye(A.shape[0] , dtype=np.float32)
    
    sx, sy = [x[0]], [y[0]]
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q, R, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
        # sx[j] = state[0]
        # sy[j] = state[1]
        
        P = covariance
        sx.append(state[0,0])
        sy.append(state[1,0])
        
    return sx, sy


import numpy as np
from ex4_utils import kalman_step

def run_kalman_nca(x, y, q=0.1, r=1.0):
    """
    x,y : 1D arrays of measurements
    dt   : timestep
    q    : process‐noise intensity (on jerk)
    r    : measurement variance
    returns: (sx, sy) filtered estimates of x,y
    """
    dt = 1.0
    # 1) State‐transition A (6×6)
    A = np.array([
        [1, 0, dt,  0,  dt*dt/2,       0],
        [0, 1,  0, dt,        0, dt*dt/2],
        [0, 0,  1,  0,       dt,       0],
        [0, 0,  0,  1,        0,      dt],
        [0, 0,  0,  0,        1,       0],
        [0, 0,  0,  0,        0,       1],
    ], dtype=np.float32)

    # 2) Observation C (2×6)
    C = np.zeros((2,6), dtype=np.float32)
    C[0,0] = 1
    C[1,1] = 1

    # 3) Process‐noise Q (6×6), white jerk model
    Q = q * np.array([
        [ dt**5/20,        0,  dt**4/8,        0,  dt**3/6,        0],
        [        0,  dt**5/20,        0,  dt**4/8,        0,  dt**3/6],
        [ dt**4/8,        0,  dt**3/3,        0,  dt**2/2,        0],
        [        0,  dt**4/8,        0,  dt**3/3,        0,  dt**2/2],
        [ dt**3/6,        0,  dt**2/2,        0,       dt,        0],
        [        0,  dt**3/6,        0,  dt**2/2,        0,       dt],
    ], dtype=np.float32)

    # 4) Measurement‐noise R (2×2)
    R = r * np.eye(2, dtype=np.float32)

    # 5) Initial state and covariance
    state = np.zeros((6,1), dtype=np.float32)
    state[0,0] = x[0]
    state[1,0] = y[0]
    P = np.diag([100,100,10,10,1,1]).astype(np.float32)

    # 6) Run filter
    sx, sy = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        y_t = np.array([xi, yi], dtype=np.float32).reshape(2,1)
        state, P, _, _ = kalman_step(A, C, Q, R, y_t, state, P)
        sx.append(float(state[0]))
        sy.append(float(state[1]))

    return np.array(sx), np.array(sy)


if __name__ == "__main__":
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    qs = [100, 5, 1, 1, 1]
    rs = [1,   1,  1, 5, 100]

    fig, axes = plt.subplots(3, 5, figsize=(15,9), sharex=True, sharey=True)
    models = [run_kalman_rw, run_kalman_ncv, run_kalman_nca]
    names = ['RW', 'NCV', 'NCA']
    for i in range(3):
        fn = models[i]
        for j, (q, r) in enumerate(zip(qs, rs)):
            ax = axes[i][j]
            sx, sy = fn(x, y, q, r)
            ax.plot(x,  y,  'o-', mfc='none', color='red',  label='measurement')
            ax.plot(sx, sy, 'o-', mfc='none', color='blue', label='filtered')
            ax.set_title(f"{names[i]}: q={q}, r={r}")
            ax.set_xlim(-20, 20); ax.set_ylim(-15, 15)
axes[0][-1].legend(loc='upper right')
fig.tight_layout()
plt.show()
        