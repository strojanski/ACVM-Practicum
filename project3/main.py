import numpy as np
import cv2
import matplotlib.pyplot as plt

from sequence_utils import VOTSequence
from ex3_utils import create_gauss_peak, create_cosine_window, gausssmooth

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

class CorrelationFilter:
    def __init__(self, alpha, lmbd=0.01):
        self.alpha = alpha  # Update rate (0.01-0.1 works well)
        self.lmbd = lmbd    # Regularization term

    def initialize(self, image, bbox):
        
        if len(bbox) == 8:
            x_ = np.array(bbox[::2])
            y_ = np.array(bbox[1::2])
            bbox = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        bbox = [int(i) for i in bbox]
        
        if bbox[2] % 2 == 0:
            bbox[2] += 1
        if bbox[3] % 2 == 0:
            bbox[3] += 1
        
        
        self.bbox = bbox
        frame = convert_grayscale(image)

        self.target_size = (bbox[2], bbox[3])
        self.frame_ix = 1
        
        # image = gausssmooth(image, 2)
        
        # Initialize filter with Gaussian target
        self.G, self.cosine = init(self.target_size, gauss_sigma=np.sqrt(2))
        F = center_on_target(frame, self.bbox)
        F = self.cosine * F
        F /= 255.0
        
        self.G_hat = np.fft.fft2(self.G)#[:-1, :-1]
        
        F = (F - np.mean(F)) / np.std(F)
        F_hat = np.fft.fft2(F)
        F_hat_conj = F_hat.conj()
        
        
        self.H_hat_conj = (self.G_hat * F_hat_conj) / (F_hat * F_hat_conj + self.lmbd)
        

    def track(self, frame):
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frame = gausssmooth(frame, 2)
        
        # 1. Localization (Eq.2)
        F = center_on_target(frame, self.bbox)
        F = self.cosine * F # Apply cosine window
        F = (F - np.mean(F)) / (np.std(F))
        
        F_hat = np.fft.fft2(F)
        R = np.fft.ifft2(F_hat * self.H_hat_conj)
        
        # Find peak response
        max_pos = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        
        w, h = R.shape[1], R.shape[0]
        
        x, y = max_pos[1], max_pos[0]
        if x > w / 2:
            x = x - w
        if y > h / 2:
            y = y - h
        
        # Update bounding box
        self.bbox[0] += x
        self.bbox[1] += y
        
        self.bbox[0] = self.bbox[0].clip(0, frame.shape[1] - w)
        self.bbox[1] = self.bbox[1].clip(0, frame.shape[0] - h)
        
        # Update 
        F_new = center_on_target(frame, self.bbox)
        F_new = self.cosine * F_new
        F_new = (F_new - np.mean(F_new)) / (np.std(F_new))
        F_new_hat = np.fft.fft2(F_new)
        F_new_hat_conj = F_new_hat.conj()
        
        
        H_new_hat_conj = (self.G_hat * F_new_hat_conj) / (F_new_hat * F_new_hat_conj + self.lmbd)
        self.H_hat_conj = (1 - self.alpha) * self.H_hat_conj + self.alpha * H_new_hat_conj
        self.H_hat_conj /= np.max(np.abs(self.H_hat_conj))  
        self.H_hat_conj = self.H_hat_conj.real

        return self.bbox
        
        
if __name__ == "__main__":
    pass        


