import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cv2
from decimal import Decimal
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['font.family'] = 'liberation serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5

COLORS = ["#d7301f", "#fc8d59", "#fdcc8a"]
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=32)
plt.rc('font', size=16)
plt.rc('axes', titlesize=32)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

K = [[615.8287353515625, 0.0, 320.0995178222656],
     [0.0, 615.9053344726562, 245.37171936035156],
     [0.0, 0.0, 1.0]]
FOV_X = np.arctan(320/K[0][0])*2
FOV_Y = np.arctan(240/K[1][1])*2

VELOCITY_ESTIMATE = "vel_estimate.npy"
V_OVER_D_ESTIMATE = "v_over_d_full.npy"
DATA_FOLDER = "data/straight-gt-1"

def get_timestamps(im_folder):
    times = [float(str(path.name)[:-4]) for path in Path(im_folder).glob("*")]
    return np.sort(np.array(times))

def get_depth_timestamps(depth_folder):
    # Decimal is needed to avoid floating point error causing issues
    # with file loading
    times = [Decimal(str(path.name)[:-4])
             for path in Path(depth_folder).glob("*")]
    return np.sort(np.array(times))

def load_homographies(ang_vel_file, timestamps):
    vel_data = pd.read_csv(ang_vel_file, sep=' ',
                           names=['time', 'vel_x', 'vel_y', 'vel_z'])
    ang_vels = []
    tmp_list = []
    i = 0
    for timestamp in timestamps:
        while vel_data.time[i] < timestamp:
            tmp_list.append(np.array([vel_data.vel_x[i],
                                      vel_data.vel_y[i],
                                      vel_data.vel_z[i]]))
            i += 1
        ang_vels.append(np.average(tmp_list, axis=0))
        tmp_list = [np.array([vel_data.vel_x[i],
                              vel_data.vel_y[i],
                              vel_data.vel_z[i]])]

    rotations = []
    homographies = []
    ang_change = np.array([0,0,0], dtype=np.float64)
    for i in range(len(timestamps)):
        if i == 0:
            time_elapsed = 0
        else:
            time_elapsed = timestamps[i] - timestamps[i-1]
        ang_change += time_elapsed * ang_vels[i]
        rotations.append(np.copy(ang_change))

        R_z = np.array([[np.cos(ang_change[0]), -np.sin(ang_change[0]), 0],
                        [np.sin(ang_change[0]), np.cos(ang_change[0]), 0],
                        [0, 0, 1]])
        R_y = np.array([[np.cos(ang_change[1]), 0, np.sin(ang_change[1])],
                        [0, 1, 0],
                        [-np.sin(ang_change[1]), 0, np.cos(ang_change[1])]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(ang_change[2]), -np.sin(ang_change[2])],
                        [0, np.sin(ang_change[2]), np.cos(ang_change[2])]])
        R = R_z @ R_y @ R_x

        A = K @ R @ np.linalg.inv(K)
        homographies.append(A)

    return np.array(homographies)

def get_nearest_depth(timestamp, depth_folder, depth_times, flo_times, H):
    flo_idx = np.searchsorted(flo_times, timestamp)
    depth_idx = np.searchsorted(depth_times, timestamp)
    if (depth_idx < len(depth_times) - 1 and
            (timestamp - float(depth_times[depth_idx-1])
             < float(depth_times[depth_idx]) - timestamp)):
        depth_idx -= 1

    depth = np.load(f"{depth_folder}/{depth_times[depth_idx]}.npy")
    depth = depth.reshape(480, 640) / 1000
    depth = cv2.resize(np.copy(depth[82:-88, 110:-126]), (640,480))
    depth = cv2.warpPerspective(depth, H[flo_idx],
        (640,480), borderValue=255)[16:-16]

    return depth

def convert_v_over_d_to_flow(v_over_d):
    angles_h = np.tile(np.arange(-320,320).reshape(-1,1),448).T*FOV_X/640
    angles_v = np.tile(np.arange(-224,224).reshape(-1,1),640)*FOV_Y/480

    flo_data = np.copy(v_over_d)
    flo_data[...,0] = flo_data[...,0]*(np.sin(angles_h)*np.cos(angles_h))
    flo_data[...,1] = flo_data[...,1]*(np.sin(angles_v)*np.cos(angles_v))
    flo_data[:,:,320] = (flo_data[:,:,319] + flo_data[:,:,321])/2
    flo_data[:,224] = (flo_data[:,223] + flo_data[:,225])/2
    flo_magnitude = np.sqrt(flo_data[...,0]**2 + flo_data[...,1]**2)
    return flo_magnitude

def compute_rmse_slice(rmses, vel_rmses, angle_rmses, slice_idx):
    # rmse can't be computed for the entire capture at once so the
    # computation is done in slices. slice count can be modified based
    # on memory  availability
    flo_timestamps = get_timestamps(DATA_FOLDER + "/rgb/")[slice_idx::12]
    depth_timestamps = get_depth_timestamps(DATA_FOLDER + "/depth/")
    H = load_homographies(DATA_FOLDER + "/vel", flo_timestamps)

    # mmaped to ensure only one slice is loaded at a time
    v_over_d = np.load(V_OVER_D_ESTIMATE, mmap_mode='r')[slice_idx::12]
    flo_data = convert_v_over_d_to_flow(v_over_d)
    vel_est = np.load(VELOCITY_ESTIMATE)[slice_idx::12]

    angles_h = np.tile(np.arange(-320,320).reshape(-1,1),448).T*FOV_X/640
    angles_v = np.tile(np.arange(-224,224).reshape(-1,1),640)*FOV_Y/480
    angle_mtx = np.sqrt((angles_h*180/np.pi)**2 + (angles_v*180/np.pi)**2)

    for f in range(flo_data.shape[0]):
        flo_center = np.unravel_index(np.argmin(
            flo_data[f,112:336,160:480]), (224,320))
        flo_center = (flo_center[0] + 112, flo_center[1] + 160)

        algn_angles_h = np.tile(np.arange(
            -flo_center[1], 640 - flo_center[1]).reshape(-1,1),448).T*FOV_X/640
        algn_angles_v = np.tile(np.arange(
            -flo_center[0], 448 - flo_center[0]).reshape(-1,1),640)*FOV_Y/480

        denominator = np.sqrt((np.sin(algn_angles_h)*np.cos(algn_angles_h))**2
                            + (np.sin(algn_angles_v)*np.cos(algn_angles_v))**2)
        new_v_over_d_est = flo_data[f,:,:]/denominator

        depth_est = np.abs(vel_est[f])/new_v_over_d_est
        gt_depth = get_nearest_depth(flo_timestamps[f],
            DATA_FOLDER + "/depth/", depth_timestamps, flo_timestamps, H)

        filtered_depth = gt_depth[np.bitwise_and(gt_depth != 0, gt_depth!=255)]
        filtered_est = depth_est[np.bitwise_and(gt_depth != 0, gt_depth!=255)]
        rmse = np.sqrt(np.median((filtered_depth - filtered_est)**2))
        rmses.append(rmse)

        # rmses at different depth cutoffs (within 1m, 3m, 5m, max distance)
        for depth_i, depth_cutoff in enumerate([1, 3, 5, 255]):
            filtered_depth = gt_depth[np.bitwise_and(np.bitwise_and(
                gt_depth != 0, gt_depth != 255), gt_depth < depth_cutoff)]
            filtered_est = depth_est[np.bitwise_and(np.bitwise_and(
                gt_depth != 0, gt_depth != 255), gt_depth < depth_cutoff)]
            rmse = np.sqrt(np.median((filtered_depth - filtered_est)**2))

            for vel_i, vel_cutoff in enumerate(np.linspace(0, .4, 20)):
                if np.abs(vel_est[f]) > vel_cutoff:
                    vel_rmses[depth_i][vel_i].append(rmse)

        # rmses at different angle ranges (e.g., between 10 and 20 degrees)
        # away from center of image
        angle_cutoffs = np.linspace(0,np.max(angle_mtx),11)
        for angle_i in range(10):
            angle_mask = np.bitwise_and(angle_mtx > angle_cutoffs[angle_i],
                angle_mtx < angle_cutoffs[angle_i + 1])

            filtered_depth = gt_depth[np.bitwise_and(
                np.bitwise_and(gt_depth != 0, gt_depth != 255), angle_mask)]
            filtered_est = depth_est[np.bitwise_and(
                np.bitwise_and(gt_depth != 0, gt_depth != 255), angle_mask)]

            rmse = np.sqrt(np.median((filtered_depth - filtered_est)**2))

            angle_rmses[angle_i].append(rmse)

def main():
    rmses = []
    vel_rmses = [[[] for _ in range(20)] for _ in range(4)]
    angle_rmses = [[] for _ in range(10)]

    for slice_idx in range(12):
        compute_rmse_slice(rmses, vel_rmses, angle_rmses, slice_idx)

    rmses = np.array(rmses)
    vel_rmses = np.array(vel_rmses)
    angle_rmses = np.array(angle_rmses)

    # Angle cutoff plot
    angles_h = np.tile(np.arange(-320,320).reshape(-1,1),448).T*FOV_X/640
    angles_v = np.tile(np.arange(-224,224).reshape(-1,1),640)*FOV_Y/480
    angle_mtx = np.sqrt((angles_h*180/np.pi)**2 + (angles_v*180/np.pi)**2)

    result_image = np.zeros((448,640))
    angle_cutoffs = np.linspace(0,np.max(angle_mtx),11)
    for angle_i in range(10):
        angle_mask = np.bitwise_and(angle_mtx > angle_cutoffs[angle_i],
            angle_mtx <= angle_cutoffs[angle_i + 1])
        result_image[angle_mask] = np.nanmedian(angle_rmses[angle_i])
    plt.imshow(result_image, extent=[-FOV_X*90/np.pi, FOV_X*90/np.pi,
        -FOV_Y*90/np.pi, FOV_Y*90/np.pi], aspect='auto')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 30
    cbar.ax.set_ylabel("Root Median Squared Error", rotation=270)
    plt.xlabel("Degrees")
    plt.ylabel("Degrees")
    plt.show()

    # Error distribution plot
    plt.figure(figsize=(4.5,5))
    plt.plot(np.linspace(0,.4,20), [np.median(rmses) for rmses in vel_rmses[3]],
        color="#1b9e77", linewidth=2.5)
    for p_cutoff in np.linspace(.01, 49.99, 10):
        percentiles = np.array([np.percentile(rmses, (p_cutoff, 100-p_cutoff))
                                for rmses in vel_rmses[3]])
        plt.fill_between(np.linspace(0,.4, 20), percentiles[:,0],
            percentiles[:,1], color="#1b9e77", alpha=.1,
            linewidth=0)
        plt.ylim(0, 1.5)
        plt.xlim(0, .4)
    plt.show()

    # Velocity cutoff error plot
    plt.figure(figsize=(4.5,5))
    for i, depth_cutoff in enumerate([1, 3, 5]):
        vel_error = np.array([np.median(rmses) for rmses in vel_rmses[i]])

        plt.plot(np.linspace(0,4,20),
            vel_error, label=f"<{depth_cutoff}m", color=COLORS[i], linewidth=2.5)
    plt.show()

if __name__ == "__main__":
    main()
