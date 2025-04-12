import numpy as np
import cv2
import matplotlib.pyplot as plt

# from sequence_utils import VOTSequence
from ex3_utils import create_gauss_peak, create_cosine_window, gausssmooth,get_patch
from utils.tracker import Tracker
import time

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
    x = max(0, min(x, frame.shape[1] - w)) 
    y = max(0, min(y, frame.shape[0] - h)) 
    x_end = x + w
    y_end = y + h
    if x_end <= x or y_end <= y:
        return np.array([])
    return frame[y:y_end, x:x_end]


def show_img(img):
    plt.imshow(img)# cmap="gray")
    plt.show()

class CorrelationFilter(Tracker):
    def __init__(self, alpha=0.2, lmbd=1e-8, sigma=np.sqrt(3), increase_factor=1.3):
        self.alpha = alpha  
        self.lmbd = lmbd    
        self.sigma = sigma
        self.increase_factor = increase_factor
        print(self.alpha, self.lmbd)
        
    def name(self):
        return f"CorrelationFilter"#_{self.sigma:.2f}_{self.alpha}_{self.lmbd}"

    def initialize(self, image, bbox):
        start = time.time()
        if len(bbox) == 8:
            x_ = np.array(bbox[::2])
            y_ = np.array(bbox[1::2])
            bbox = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        bbox = [int(i) for i in bbox]
        inc_w = int(np.floor(bbox[2] * self.increase_factor))
        inc_h = int(np.floor(bbox[3] * self.increase_factor))
        bbox[0] = int(bbox[0] - (inc_w - bbox[2]) / 2)
        bbox[1] = int(bbox[1] - (inc_h - bbox[3]) / 2)
        bbox[2] = inc_w
        bbox[3] = inc_h
        
        if bbox[2] % 2 == 0:
            bbox[2] += 1
        if bbox[3] % 2 == 0:
            bbox[3] += 1
        
        
        self.bbox = bbox
        frame = convert_grayscale(image)

        self.target_size = (bbox[2], bbox[3])
        self.frame_ix = 1
        
        
        self.G, self.cosine = init(self.target_size, gauss_sigma=self.sigma)
        F,_ = get_patch(frame, self.bbox[:2], self.bbox[2:])
        F = self.cosine * F
        
        
        self.G_hat = np.fft.fft2(self.G)
        
        F = (F - np.mean(F)) / np.std(F)
        F_hat = np.fft.fft2(F)
        F_hat_conj = F_hat.conj()
        
        
        self.H_hat_conj = (self.G_hat * F_hat_conj) / (F_hat * F_hat_conj + self.lmbd)
        end = time.time()
        
        # print("Initialization took", end - start)

    def track(self, frame):
        start = time.time()
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        F,_ = get_patch(frame, self.bbox[:2], self.bbox[2:])
        F = self.cosine * F # Apply cosine window
        F = (F - np.mean(F)) / (np.std(F))
        
        F_hat = np.fft.fft2(F)
        R = np.fft.ifft2(F_hat * self.H_hat_conj)
        
        max_pos = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        
        w, h = R.shape[1], R.shape[0]
        
        x, y = max_pos[1], max_pos[0]
        if x > w / 2:
            x = x - w
        if y > h / 2:
            y = y - h
        
        self.bbox[0] += x
        self.bbox[1] += y
        
        self.bbox[0] = self.bbox[0].clip(0, frame.shape[1] - w)
        self.bbox[1] = self.bbox[1].clip(0, frame.shape[0] - h)
    
        F_new,_ = get_patch(frame, self.bbox[:2], self.bbox[2:])
        F_new = self.cosine * F_new
        F_new = (F_new - np.mean(F_new)) / (np.std(F_new))
        F_new_hat = np.fft.fft2(F_new)
        F_new_hat_conj = F_new_hat.conj()
        
        
        H_new_hat_conj = (self.G_hat * F_new_hat_conj) / (F_new_hat * F_new_hat_conj + self.lmbd)
        self.H_hat_conj = (1 - self.alpha) * self.H_hat_conj + self.alpha * H_new_hat_conj
        self.H_hat_conj = self.H_hat_conj

        end = time.time()
        
        # print("Tracking the frame took", end- start)

        return (self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])#(self.bbox[0], self.bbox[1], self.target_size[0], self.target_size[1])
        

        
if __name__ == "__main__":
    pass        


