import numpy as np
import cv2
import matplotlib.pyplot as plt

from sequence_utils import VOTSequence
from ex3_utils import create_gauss_peak, create_cosine_window

data_path = "vot2013/"

def init(size, gauss_sigma):
    gauss = create_gauss_peak(size, sigma=gauss_sigma)
    cosine = create_cosine_window(size)
    
    return gauss, cosine

def read_grayscale(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def convert_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def center_on_target(frame, region):
    x, y, w, h = region
    # Clamp region coordinates to frame boundaries:
    x = max(0, x)
    y = max(0, y)
    x_end = min(frame.shape[1], x + w)
    y_end = min(frame.shape[0], y + h)
    if x_end <= x or y_end <= y:
        return np.array([])  # Return empty if region is invalid
    return frame[y:y_end, x:x_end]


def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.show()

class CorrelationFilter:
    def __init__(self, alpha, lmbd=0.01):
        self.alpha = alpha  # Update rate (0.01-0.1 works well)
        self.lmbd = lmbd    # Regularization term

    def get_next_frame(self):
        return read_grayscale(self.sequence.frame(self.frame_ix))

    def define_search_region(self):
        """Returns search region matching target size"""
        x, y, w, h = self.bbox
        cx, cy = x + w//2, y + h//2
        return [cx - w//2, cy - h//2, w, h]  # Same size as target

    def initialize_sequence(self, sequence_name):
        self.sequence = VOTSequence(data_path, sequence_name)
        self.initial_frame = read_grayscale(self.sequence.frame(0))
        self.bbox = [int(i) for i in self.sequence.get_annotation(0, type="rectangle")]
        self.target_size = (self.bbox[2], self.bbox[3])
        self.frame_ix = 1
        self.search_region = self.define_search_region()

    def initialize_filter(self, gauss_sigma=3):
        """Initialize filter with Gaussian target"""
        self.G, self.cosine = init(self.target_size, gauss_sigma)
        F = center_on_target(self.initial_frame, self.bbox)
        F = self.cosine * F
        # Pre-compute Fourier terms
        self.G_hat = np.fft.fft2(self.G)[:-1, :-1]
        
        F_hat = np.fft.fft2(F)
        F_hat_conj = F_hat.conj()
        
        # Initial filter (Eq.1)
        self.H_hat_conj = (self.G_hat * F_hat_conj) / (F_hat * F_hat_conj + self.lmbd)
        show_img(np.abs(self.H_hat_conj))

    def track(self):
        while self.frame_ix < self.sequence.length():
            frame = self.get_next_frame()
            
            # 1. Localization (Eq.2)
            F = center_on_target(frame, self.search_region)
            F = self.cosine * F # Apply cosine window
            
            F_hat = np.fft.fft2(F)
            R = np.fft.ifft2(self.H_hat_conj * F_hat)
            
            
            # Find peak response
            max_pos = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            dy, dx = max_pos[0] - R.shape[0]//2, max_pos[1] - R.shape[1]//2
            
            # Update bounding box
            self.bbox[0] += dx
            self.bbox[1] += dy
            
            # 2. Online Update (Eq.3)
            F_new = center_on_target(frame, self.bbox)
            F_new_hat = np.fft.fft2(F_new)
            F_new_hat_conj = F_new_hat.conj()
            
            F_new_hat__ = np.zeros_like(self.G_hat)
            F_new_hat__[:F_new_hat.shape[0], :F_new_hat.shape[1]] = F_new_hat
            F_new_hat = F_new_hat__
            
            F_new_hat_conj__ = np.zeros_like(self.G_hat)
            F_new_hat_conj__[:F_new_hat_conj.shape[0], :F_new_hat_conj.shape[1]] = F_new_hat_conj
            F_new_hat_conj = F_new_hat_conj__
            
            print(F_new_hat_conj.shape, self.G_hat.shape, F_new_hat.shape)
            
            H_new_hat_conj = (self.G_hat * F_new_hat_conj) / (F_new_hat * F_new_hat_conj + self.lmbd)
            self.H_hat_conj = (1 - self.alpha) * self.H_hat_conj + self.alpha * H_new_hat_conj
            
            # Visualization
            self.sequence.draw_region(frame, self.bbox, (0,0,255), 2)
            self.sequence.show_image(frame, 20)
            self.frame_ix += 1

    # def track(self):
    #     while self.frame_ix < self.sequence.length():
    #         frame = self.get_next_frame()
            
    #         # 1. Localization (Eq.2)
    #         F = center_on_target(frame, self.search_region)
    #         F_hat = np.fft.fft2(F)
    #         R = np.fft.ifft2(self.H_hat_conj * F_hat)
            
    #         # Find peak response
    #         max_pos = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    #         dy, dx = max_pos[0] - R.shape[0]//2, max_pos[1] - R.shape[1]//2
            
    #         # Update bounding box
    #         self.bbox[0] += dx
    #         self.bbox[1] += dy
            
    #         # 2. Online Update (Eq.3)
    #         F_new = center_on_target(frame, self.bbox)
    #         self.sequence.draw_region(frame, self.bbox, (0,0,255), 2)
    #         F_new_hat = np.fft.fft2(F_new)
    #         F_new_hat_conj = F_new_hat.conj()
            
    #         H_new_hat_conj = (self.G_hat * F_new_hat_conj) / (F_new_hat * F_new_hat_conj + self.lmbd)
    #         self.H_hat_conj = (1 - self.alpha)*self.H_hat_conj + self.alpha*H_new_hat_conj
            
    #         # Visualization
    #         self.sequence.draw_region(frame, self.bbox, (0,0,255), 2)
    #         self.frame_ix += 1
    #         self.search_region = self.bbox

        
        
        
if __name__ == "__main__":
    cfilter = CorrelationFilter(alpha=0.02, lmbd=1e-3)
    cfilter.initialize_sequence("bicycle")
    cfilter.initialize_filter()
    cfilter.track()
        


