import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from ex4_utils import kalman_step, gaussian_prob, sample_gauss
from ex2_utils import extract_histogram, create_epanechnik_kernel, get_patch
from utils.tracker import Tracker

np.random.seed(0)

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
            [1/3, 0,   1/2, 0],
            [0,   1/3, 0,   1/2],
            [1/2, 0,   1,   0],
            [0,   1/2, 0,   1],
        ], dtype=np.float32)
    
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


def run_kalman_nca(x, y, q=0.1, r=1.0):
    """
    x,y : 1D arrays of measurements
    dt   : timestep
    q    : process‐noise intensity (on jerk)
    r    : measurement variance
    returns: (sx, sy) filtered estimates of x,y
    """
    A = np.array([
        [1, 0, 1, 0, 1/2, 0],
        [0, 1, 0, 1, 0, 1/2],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

    C = np.zeros((2,6), dtype=np.float32)
    C[0,0] = 1
    C[1,1] = 1

    Q = q * np.array([
                [1/20, 0, 1/8, 0, 1/6, 0],
                [0, 1/20, 0,  1/8, 0, 1/6],
                [1/8, 0, 1/3, 0, 1/2, 0],
                [0, 1/8, 0,  1/3,0, 1/2],
                [1/6, 0,  1/2, 0, 1, 0],
                [0, 1/6, 0, 1/2, 0, 1],
            ], dtype=np.float32)

    R = r * np.eye(2, dtype=np.float32)

    state = np.zeros((6,1), dtype=np.float32)
    state[0,0] = x[0]
    state[1,0] = y[0]
    P = np.diag([100,100,10,10,1,1]).astype(np.float32)

    sx, sy = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        y_t = np.array([xi, yi], dtype=np.float32).reshape(2,1)
        state, P, _, _ = kalman_step(A, C, Q, R, y_t, state, P)
        sx.append(float(state[0]))
        sy.append(float(state[1]))

    return np.array(sx), np.array(sy)

class ParticleFilter(Tracker):
    def __init__(self, n_particles=100, alpha=1e-3, q_scale=.01, mmodel="ncv", color="ycrcb"):
        self.n_particles = n_particles
        self.alpha = alpha  
        self.particles = None  
        self.mmodel = mmodel
        self.q_scale = q_scale
        self.weights = np.ones(n_particles, dtype=np.float32) / n_particles
        self.color = color
        
    def name(self):
        return f"ParticleFilter_ncv_ycrcb"

    def initialize(self, img, region):
        region = np.array(region, dtype=np.int32)
        self.x, self.y, self.w, self.h = region
        
        if self.color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.color == "lab":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif self.color == "ycrcb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
        
        # Get visual model
        patch, _ = get_patch(img, (self.x + self.w/2, self.y + self.h/2), (self.w, self.h)) 
        self.kernel = create_epanechnik_kernel(self.w, self.h, 2)
        self.visual_model = extract_histogram(patch, 16, weights=self.kernel)
        
        self.q = self.q_scale * np.minimum(self.w, self.h)

        if self.mmodel == "rw":
            self.A = np.eye(2, dtype=np.float32)
            self.Q = self.q * np.eye(2, dtype=np.float32)
            
        elif self.mmodel == "ncv":
            self.Q = self.q * np.array([
                [1/3, 0,   1/2, 0],
                [0,   1/3, 0,   1/2],
                [1/2, 0,   1,   0],
                [0,   1/2, 0,   1],
            ], dtype=np.float32)
            
            
            self.A = np.array([
                [1, 0, 1, 0],  
                [0, 1, 0, 1],  
                [0, 0, 1, 0],  
                [0, 0, 0, 1],  
            ], dtype=np.float32)
        
        
        elif self.mmodel == "nca":
            self.A = np.array([
            [1, 0, 1, 0, 1/2, 0],
            [0, 1, 0, 1, 0, 1/2],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            ], dtype=np.float32)

            self.Q = self.q * np.array([
                [1/20, 0, 1/8, 0, 1/6, 0],
                [0, 1/20, 0,  1/8, 0, 1/6],
                [1/8, 0, 1/3, 0, 1/2, 0],
                [0, 1/8, 0,  1/3,0, 1/2],
                [1/6, 0,  1/2, 0, 1, 0],
                [0, 1/6, 0, 1/2, 0, 1],
            ], dtype=np.float32)
            
        position = [self.x + self.w/2, self.y + self.h/2]
        mu = np.zeros_like(self.A[0])
        mu[0] = position[0]
        mu[1] = position[1]        

        self.particles = sample_gauss(mu, self.Q, self.n_particles)

    def track(self, img):
        if self.color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.color == "lab":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif self.color == "ycrcb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
        # 1) Replace existing particles by sampling n new particles based on weight distribution of the old particles
        w = self.weights
        w_norm = w / w.sum()
        w_cumsum = np.cumsum(w_norm)
        w_cumsum[-1] = 1.0  # Ensure the last value is exactly 1.0
        
        rand_samples = np.random.rand(self.n_particles, 1)
        # sampled_idxs = np.digitize(rand_samples, w_cumsum)
        sampled_idxs = np.searchsorted(w_cumsum, rand_samples.flatten())

        particles = self.particles[sampled_idxs.flatten(), :]

        # 2) Move each particle using the dynamic model (also apply noise)
        new_p = (self.A @ particles.T).T
        noise = sample_gauss(
            np.zeros(len(self.A), dtype=np.float32),
            self.Q,
            self.n_particles
        )
        self.particles = new_p + noise
        
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, img.shape[1])
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, img.shape[0])

        # 3)  Update weights of particles based on visual model similarity.
        new_w = np.zeros(self.n_particles, dtype=np.float32)
        for i, p in enumerate(self.particles):
            patch, _ = get_patch(img, (p[0], p[1]), (self.w, self.h))
            # plt.imshow(patch)
            # plt.show()
            hist = extract_histogram(patch, 16, weights=self.kernel)
            
            # Normalize histograms and add epsilon to avoid NaN
            hist = hist / hist.sum()
            target_hist = self.visual_model / self.visual_model.sum()
            
            # Compute Bhattacharyya coefficient (clipped to [0, 1])
            dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(hist) - np.sqrt(target_hist))**2))
            

            new_w[i] = np.exp(-1/2 * (dist**2 /0.1**2))

        self.weights = new_w / new_w.sum()

        # 4) Compute new state of the object as a weighted sum of particle states. Use the normalized particle weights as weights in the sum.)
        x = np.sum(self.particles[:, 0] * self.weights)
        y = np.sum(self.particles[:, 1] * self.weights)
        
        
        # Update visual model with the new target position
        patch, _ = get_patch(img, (x, y), (self.w, self.h))
        
        hist_new = extract_histogram(patch, 16, weights=self.kernel)
        hist_new = hist_new / hist_new.sum()
        self.visual_model = (1 - self.alpha) * self.visual_model + self.alpha * hist_new

        return (x-self.w//2, y-self.h//2, self.w, self.h)#, self.particles, self.weights

if __name__ == "__main__":
    N = 40
    v = np.linspace(16, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    
    # x = (np.array([0, 1, 1, 0, 0, 3/4, 3/4, 1/4, 1/2]) - np.ones(9) * 1/2) * 15
    # y = (np.array([1, 1, 0, 0, 3/4, 3/4, 1/4, 1/4, 1/2]) - np.ones(9) * 1/2) * 15

    # x = v - np.mean(v)
    # y = np.sin(v) * v# + np.random.normal(0, 2, N)

    # x = v - np.mean(v)
    # y = v**3 / 100 + v**2/10 - 2 * v - 2  


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
            