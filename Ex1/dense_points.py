import numpy as np
import cv2
from pathlib import Path


CACHE_desc30 = Path("traj_cache")      #storing 30-dim trajectory descriptor
CACHE_desc30.mkdir(exist_ok=True)

def sample_dense_points(gray,
                        step = 5,
                        eig_mult= 1e-3):

    H, W = gray.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]           # regular grid
    grid   = np.vstack((xs.ravel(), ys.ravel())).T  # (N,2)

    saliency = cv2.cornerMinEigenVal(gray, 3, 3)    # Shi–Tomasi λ_min
    thresh   = eig_mult * saliency.max()
    keep     = saliency[grid[:, 1], grid[:, 0]] >= thresh
    return grid[keep].astype(np.float32)            # (x,y) float for sub-pixel

def compute_flow(prev, curr):

    return cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        0.5,
        3,
        15,
        3,
        5, 1.2, 0
    )

def advance_tracks(tracks, flow,
                   jump_frac=0.7,
                   track_len=15,
                   min_total=0.5):
    """
    Move every active track one step forward.
    """
    H, W, _ = flow.shape
    new_active, finished = [], []

    for tr in tracks:
        #print(tr)
        x, y = tr[-1]
        dx, dy = flow[int(y), int(x)]
        nx, ny = x + dx, y + dy

        # out-of-frame?
        if not (0 <= nx < W and 0 <= ny < H):
            continue

        # sanity-check
        
        total_disp = np.hypot(nx - tr[0][0], ny - tr[0][1])
        #print(f'total_disp from the start of the trajectory:{total_disp}')
        #print(f'Current displacement: {np.hypot(dx, dy)}')
        if len(tr) > 1:
            if np.hypot(dx, dy) > jump_frac * (total_disp + 1e-8):
                #print('dropping trajectory due to high displacement.')
                continue                              # drop erratic

        tr.append((nx, ny))
        if len(tr) == track_len:
            if total_disp > min_total:
                finished.append(tr)
        else:
            new_active.append(tr)

    return new_active, finished

#The paper wants one active point per STEP × STEP grid cell at all times.
def reseed(tracks, gray, step):
    """
    Fill every step×step cell not already occupied by a track head.
    Modifies 'tracks' in-place by appending [seed] lists.
    """
    seeds = sample_dense_points(gray, step)
    if tracks:
        heads = np.array([tr[-1] for tr in tracks])
        dist  = np.linalg.norm(seeds[:, None] - heads[None], axis=2)
        seeds = seeds[dist.min(axis=1) >= step / 2] #If the cell already has a head, you discard that seed — 
                 #it’s already covered.If no head is in that cell, you keep the seed and start a new track.
        #print(seeds)
    tracks.extend([tuple(p)] for p in seeds)

def trajectory_shape(track):
    """30-D normalised displacement vector (Eq. 3)."""
    disp = np.diff(np.array(track), axis=0)          # (15,2)
    norm = np.linalg.norm(disp, axis=1).sum() + 1e-8
    return (disp / norm).flatten().astype(np.float32)


def extract_dense_trajectories(video_path,
                               step = 5,
                               track_len = 16,
                               return_tracks = False):
    """
    Track dense points and return trajectory-shape descriptors.

    Parameters:
    
    video_path   : str / Path
    step         : grid spacing  (pixels)
    track_len    : #positions per trajectory (16 → 15 displacements = 30-D)
    return_tracks: if True → also return {frame_idx: [...] } dictionaries

    Returns:
    
    If return_tracks == False:
        desc30 : (N, 30) float32 array
    If return_tracks == True:
        finished_by_frame : dict[int, list[list[(x,y)]]]
        shapes_by_frame   : dict[int, list[np.ndarray shape (30,)]]
        desc30            : (N, 30) float32 array
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"Cannot open {video_path}")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── containers ──────────────────────────────────────────────────────
    tracks        = []       # active, still-growing lists of (x,y)
    descriptors   = []       # 30-D vectors (all finished)
    finished_dict = {}       # key = end frame idx,   value = list of tracks
    shape_dict    = {}       # key = end frame idx,   value = list of shape30

    # ── seed frame 0 ───────────────────────────────────────────────────
    tracks.extend([tuple(p)] for p in sample_dense_points(prev_gray, step))

    frame_idx = 0            # keeps track of *prev* frame’s index
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # A. Farnebäck flow prev → curr
        flow = compute_flow(prev_gray, gray)

        # B. advance tracks, collect those that hit length = track_len
        tracks, done = advance_tracks(tracks, flow, track_len=track_len)

        for tr in done:
            shape30 = trajectory_shape(tr)
            descriptors.append(shape30)

            if return_tracks:                       # store for Task-2
                finished_dict.setdefault(frame_idx, []).append(tr)
                shape_dict   .setdefault(frame_idx, []).append(shape30)

        # C. reseed empty grid cells on current frame
        reseed(tracks, gray, step)

        # slide window
        prev_gray = gray
        frame_idx += 1

    cap.release()

    desc30 = (np.vstack(descriptors)
              if descriptors else np.empty((0, 30), np.float32))

    desc30_file = CACHE_desc30 / (Path(video_path).stem + ".npy")
    np.save(desc30_file, desc30)

    if return_tracks:
        return finished_dict, shape_dict, desc30
    else:
        return desc30