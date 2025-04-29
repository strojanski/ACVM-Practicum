import numpy as np
import cv2
from ex1_utils import *
from ex2_utils import *
import matplotlib.pyplot as plt

class MeanShiftTracker(Tracker):
    def __init__(self, params):
        super().__init__(params)
    
    # Get ground truth patch and its histogram
    def initialize(self, image, region):
        # 1: Extract ground truth patch
        self.convergence_criteria = self.parameters.convergence_criteria
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3])

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (np.floor(region[2]), np.floor(region[3]))
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        
        self.q = extract_histogram(self.template, self.parameters.n_bins, self.kernel).astype(np.float64)
        self.q /= sum(self.q)
        

        if self.parameters.resize:
            cx, cy = self.parameters.size // 2, self.parameters.size // 2
            x_coords = np.arange(self.parameters.size) - cx
            y_coords = np.arange(self.parameters.size) - cy
        else:
            cx, cy = self.size[1] // 2, self.size[0] // 2  # Center coordinates
            x_coords = np.arange(self.size[0]) - cx
            y_coords = np.arange(self.size[1]) - cy
        
        self.X, self.Y = np.meshgrid(x_coords, y_coords)

        self.eps = self.parameters.eps
    
    def _bg_correction(self, patch, q, p):            
        cu = np.argsort(p)
        cu = [c for c in cu if p[c] != 0]  
        cu = cu[0] if len(cu) > 0 else 1
        
        cu = np.array([min(p[cu], 1)] * len(p)) / (p + self.eps * np.ones_like(p))
        cu = cu.clip(1e-4, 1)
        
                
        p = cu * p
        q = cu * q
        q /= np.sum(q)
        
        return q, p
    
    def track(self, image):
        x_prev, y_prev = 0, 0
        n_iter = 0
        for i in range(20):
            patch, mask = get_patch(image, self.position, self.size)

            if self.parameters.resize:
                patch = cv2.resize(patch, (self.parameters.size, self.parameters.size), interpolation=cv2.INTER_LANCZOS4)

            q = self.q
            p = extract_histogram(patch, self.parameters.n_bins, self.kernel).astype(np.float64)
            
            # p /= np.sum(p)
            if self.parameters.correction:
                bg_patch, _ = get_patch(image, self.position, (self.size[0] * 2, self.size[1]*2))
                # Compute the position of the original patch within the background patch
                bg_h, bg_w = bg_patch.shape[:2]
                orig_h, orig_w = patch.shape[:2]
                start_x = (bg_w - orig_w) // 2
                start_y = (bg_h - orig_h) // 2

                # Set the pixels of the original patch area to zero
                bg_patch[start_y:start_y + orig_h, start_x:start_x + orig_w] = 0
                q, p = self._bg_correction(bg_patch, self.q, p)
            p /= np.sum(p)        


            # Compute weights
            V = np.sqrt(q / (p + self.eps * np.ones_like(p)))
            
            # Backproject
            W = backproject_histogram(patch, V, self.parameters.n_bins)
            if np.any(np.isnan(W)):
                print("Warning: NaN values found in weights, skipping iteration.")
                continue 

            # Compute movement vectors
            denom = np.sum(W)
            if denom == 0:
                denom = 1
            x_pos = (np.sum(self.X * W) / denom) + self.position[0]
            y_pos = (np.sum(self.Y * W) / denom) + self.position[1] 
            # print((np.sum(self.X * W) / denom), (np.sum(self.Y * W) / denom))
            self.position = (x_pos, y_pos)
            
            # Compute top-left corner of bounding box
            x_tl = int(x_pos - self.size[0] / 2)
            y_tl = int(y_pos - self.size[1] / 2)
            
            n_iter = i
            
            # If we move for less than a pixel
            if (x_prev - x_pos) ** 2 + (y_prev - y_pos) ** 2 < self.convergence_criteria:
                # print(f"{i} iterations")
                break
            
            x_prev = x_pos
            y_prev = y_pos
            
        self.template, _ = get_patch(image, self.position, self.size)
        
        q_tilda = extract_histogram(self.template, self.parameters.n_bins, self.kernel)
        q_tilda /= np.sum(q_tilda)
        
        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * q_tilda
        
        return (x_tl, y_tl, self.size[0], self.size[1]), n_iter

    
class MSParams():
    def __init__(self):
        self.convergence_criteria = 1
        self.eps = 1e-4
        self.resize = True
        self.size = 50
        self.correction = True
        self.n_bins = 16
        self.alpha = 1e-2
        self.sigma = 2
