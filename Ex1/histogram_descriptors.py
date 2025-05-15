import numpy as np
import cv2
from itertools import cycle
from collections import deque
from pathlib import Path
from tqdm import tqdm
import os


from dense_points import extract_dense_trajectories




PATCH = 32
HALF = PATCH//2
TRACK_LEN = 15

# fixed bin edges
BIN_HOG = 8
BIN_HOF = 9          # 8 orient + 1 static
BIN_MBH = 8

def crop_volume(frames, flows_u, flows_v, traj):

    L = 15
    H, W = frames[0].shape

    img_vol = np.zeros((L, PATCH, PATCH), np.float32)
    u_vol   = np.zeros_like(img_vol)
    v_vol   = np.zeros_like(img_vol)

    #fill each time dim (15) with the 32x32 patch centered on the trajectory head
    #anything that falls outside the image stays zero

    for t, (x, y) in enumerate(traj):
        x, y = int(round(x)), int(round(y)) #flow tracking gave us float, but to slice the arrays we need int

        #Original full frame bounds of the patch
        x0, x1 = x - HALF, x + HALF #left-right bounds
        y0, y1 = y - HALF, y + HALF #top-bottom bounds

        #Clip these bounds so they stay inside the image
        xs0, xs1 = max(0, x0), min(W, x1)
        ys0, ys1 = max(0, y0), min(H, y1)

        #The patches might bet clipped to not be 32x32 anymore, if they out of image bounds
        #Calulcate the offset for zero padding (the arrays we fill are already full of zeros)
        #So we only fill in those places which have a value from the frame

        dx0, dx1 = xs0-x0, PATCH - (x1 - xs1)
        dy0, dy1 = ys0-y0, PATCH - (y1 - ys1)

        img_vol[t, dy0:dy1, dx0:dx1] = frames[t][ys0:ys1, xs0:xs1]
        u_vol  [t, dy0:dy1, dx0:dx1] = flows_u[t][ys0:ys1, xs0:xs1]
        v_vol  [t, dy0:dy1, dx0:dx1] = flows_v[t][ys0:ys1, xs0:xs1]

        return img_vol, u_vol, v_vol

def tube_generate(vol):

    '''
    yield indices for slicing the 32x32x15 block into 2x2x3 tubes
    '''

    for ti in range(3):
        t0, t1 = ti*5, (ti+1)*5
        for ry in range(2):
            r0, r1 = ry*16, (ry+1)*16
            for cx in range(2):
                c0, c1 = cx*16, (cx+1)*16
                yield (t0, t1, r0, r1, c0, c1)




def hog_hist(patch):
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.int32(ang // (360/BIN_HOG)) % BIN_HOG
    hist = np.bincount(bins.ravel(), mag.ravel(), BIN_HOG).astype(np.float32)
    return hist / (np.linalg.norm(hist)+1e-6)

def hof_hist(u_patch, v_patch, static_thr=1.0):
    mag, ang = cv2.cartToPolar(u_patch, v_patch, angleInDegrees=True)
    static = mag < static_thr
    bins = np.int32(ang // (360/8)) % 8
    bins[static] = 8                      # 9th bin
    hist = np.bincount(bins.ravel(), mag.ravel(), BIN_HOF).astype(np.float32)
    return hist / (np.linalg.norm(hist)+1e-6)

def mbh_hist(comp_patch):
    # comp_patch is u or v flow channel
    gx = cv2.Sobel(comp_patch, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(comp_patch, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.int32(ang // (360/BIN_MBH)) % BIN_MBH
    hist = np.bincount(bins.ravel(), mag.ravel(), BIN_MBH).astype(np.float32)
    return hist / (np.linalg.norm(hist)+1e-6)

def tube_descriptors(img_vol, u_vol, v_vol):
    HOG, HOF, MBHx, MBHy = [], [], [], []

    for t0,t1,r0,r1,c0,c1 in tube_generate(img_vol):
        # slice each 16×16×5 tube
        img_patch = img_vol[t0:t1, r0:r1, c0:c1]
        u_patch   = u_vol  [t0:t1, r0:r1, c0:c1]
        v_patch   = v_vol  [t0:t1, r0:r1, c0:c1]

        HOG .append(hog_hist(img_patch))
        HOF .append(hof_hist(u_patch, v_patch))
        MBHx.append(mbh_hist(u_patch))
        MBHy.append(mbh_hist(v_patch))

    return (np.hstack(HOG),  np.hstack(HOF),
            np.hstack(MBHx), np.hstack(MBHy))      # 96,108,96,96


def descriptor_426(frames, flows_u, flows_v, traj, shape30):
    img_vol, u_vol, v_vol = crop_volume(frames, flows_u, flows_v, traj)
    h_hog, h_hof, h_mbx, h_mby = tube_descriptors(img_vol, u_vol, v_vol)
    return np.hstack((shape30, h_hog, h_hof, h_mbx, h_mby))   # 30+96+108+96+96

def compute_desc426(video):

    finished_dict, shape_dict, desc30 = extract_dense_trajectories(video, return_tracks=True)
    
    cap = cv2.VideoCapture(video)
    ok, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frames_buf, flows_u_buf, flows_v_buf = deque(maxlen=TRACK_LEN), deque(maxlen=TRACK_LEN), deque(maxlen=TRACK_LEN)

    desc426 = []
    frame_idx = 0
    while ok:
        gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        frames_buf.append(gray)
    
        ok, frame = cap.read()
        if not ok:
            break
        flow = cv2.calcOpticalFlowFarneback(gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows_u_buf.append(flow[...,0]); flows_v_buf.append(flow[...,1])
    
        # when a trajectory ends on *this* frame, build its 426-D descriptor
        for tr, shape30 in zip(finished_dict.get(frame_idx, []), shape_dict.get(frame_idx, [])):
            if len(frames_buf) == 15:  # buffers full
                d426 = descriptor_426(list(frames_buf),
                                      list(flows_u_buf),
                                      list(flows_v_buf),
                                      tr, shape30)
                desc426.append(d426)
    
        prev = frame
        frame_idx += 1
    
    cap.release()
    desc426 = np.array(desc426, np.float32)

    return desc426

