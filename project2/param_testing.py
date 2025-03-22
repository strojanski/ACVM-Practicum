import time

import cv2

from sequence_utils import VOTSequence
from ncc_tracker_example import NCCTracker, NCCParams
from main import MeanShiftTracker, MSParams
import matplotlib.pyplot as plt
import numpy as np


def compute_iou(box1, box2):
    # box = (x_tl, y_tl, width, height)
    
    # Coordinates of the intersection box
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])
    
    # If there's no intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Area of intersection
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Area of both bounding boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    # Area of union
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area
    return iou

# set the path to directory where you have the sequences
dataset_path = 'data/' # TODO: set to the dataet path on your disk
sequence_ = 'basketball'  # choose the sequence you want to test
total_fails = 0
# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 10
font = cv2.FONT_HERSHEY_PLAIN

# create sequence object
# create parameters and tracker objects
# parameters = NCCParams()
# tracker = NCCTracker(parameters)
parameters = MSParams()

hist_bins = [4, 8, 16, 32]

res = {}

for n_bins in hist_bins:
    sequence = VOTSequence(dataset_path, sequence_)
    init_frame = 0
    n_failures = 0
    iters = []
    ious = []
    parameters.n_bins = n_bins
    tracker = MeanShiftTracker(parameters)
    time_all = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox, n_iter = tracker.track(img)
            time_all += time.time() - t_
            iters.append(n_iter)
            

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        ious.append(compute_iou(predicted_bbox, gt_bb))
        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)
        # time.sleep(1)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    res[n_bins] = (n_failures, (sequence.length() / time_all), np.mean(n_iter), np.mean(ious))
    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    print("Mean IOU: ", np.mean(ious))
    total_fails += n_failures
print(res)
