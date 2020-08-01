import numpy as np

from code.imu_model import sensor_error_model, misalignment
from code.quaternion import Quaternion

from helpers import gravity_ned
from CONFIG import *

def residual_acc(theta, accs, idxs, least_squares = True):
    '''
    Accelerometer cost function according to equation (10) in the paper
    '''
    M, b = sensor_error_model(theta[0:9])
    C = np.linalg.inv(M)
    
    z = np.zeros(3 * np.sum(idxs))
    standstill_counter = 0;
    for i, standstill in enumerate(idxs):
        if standstill:
            acc = C @ (accs[i].T - b)
            # equations (11) and (12)
            roll = np.arctan2(acc[1], acc[2])
            pitch = np.arctan2(-acc[0], np.sqrt(acc[2]**2 + acc[1]**2))

            # equation (13)
            u = np.linalg.norm(gravity_ned(LATITUDE_RAD, HEIGHT_METERS)) * np.array([-np.sin(pitch),
                                    np.cos(pitch)*np.sin(roll),
                                    np.cos(pitch)*np.cos(roll)])
            residual = u - acc

            z[2*standstill_counter]     = residual[0]
            z[2*standstill_counter + 1] = residual[1]
            z[2*standstill_counter + 2] = residual[2]
            standstill_counter += 1
           
    if least_squares:
        return z.flatten()
    else:
        return z.flatten() @ z.flatten()


def residual_gyr(theta, gyrs, accs, standstill_flags, dt, least_squares = True):
    '''
    Gyroscope cost function according to equation (16) in the paper
    '''
    M, b = sensor_error_model(theta[0:9])
    C = np.linalg.inv(M)
    
    acc_to_gyro = misalignment(theta[-3:])
    Rm = np.linalg.inv(acc_to_gyro)
    
    # find the indexes, when the sensor was rotated
    standstill_changes = np.hstack((0, np.diff(standstill_flags)))
    motion_starts = np.where(standstill_changes == -1)
    motion_ends = np.where(standstill_changes == 1)
    motion_start_end_idxs = []
    for s, e in zip(*motion_starts, *motion_ends):
        motion_start_end_idxs.append([s - 5, e + 5])
    
    z = np.zeros(2 * len(motion_start_end_idxs))
    prev_idx_motion_end = 0
    for i, (idx_motion_start, idx_motion_end) in enumerate(motion_start_end_idxs):
        acc_start = np.mean(accs[prev_idx_motion_end:idx_motion_start,:], axis=0)
        
        roll_start_acc = np.arctan2(acc_start[1], acc_start[2])
        pitch_start_acc = np.arctan2(-acc_start[0], np.sqrt(acc_start[2]**2 + acc_start[1]**2))
        yaw_start = 0.0
        
        # gyroscope measurements during rotation, w_j in eq. (16)
        gyr = gyrs[idx_motion_start:idx_motion_end,:]
        # equivalent to R_{a, j_start} in eq. (16)
        q = Quaternion.from_euler(roll_start_acc, pitch_start_acc, yaw_start)
        for w in gyr:
            w_b = C @ Rm @ w - C @ b
            q_upd = Quaternion.exactFromOmega((w_b) * dt)
            q = q.prod(q_upd)

        roll_end_gyr, pitch_end_gyr,_ = q.to_euler()
        
        if i < len(motion_start_end_idxs) - 1:
            next_idx_motion_start = motion_start_end_idxs[i+1][0]
        else:
            next_idx_motion_start = -1

        acc_end = np.mean(accs[idx_motion_end:next_idx_motion_start], axis=0)
        roll_end_acc = np.arctan2(acc_end[1], acc_end[2])
        pitch_end_acc = np.arctan2(-acc_end[0], np.sqrt(acc_end[2]**2 + acc_end[1]**2))
        
        ori_diff_gyr = np.array([[roll_end_gyr,
                                 pitch_end_gyr]])
        ori_diff_acc = np.array([[roll_end_acc,
                                 pitch_end_acc]])

        # equivalent to (Rg{w} * R_{a, j_start})^T in eq. (16)
        qgyr = Quaternion.from_euler(roll_end_gyr, pitch_end_gyr, 0.0)
        # R_{a, j_end} in eq. (16)
        qacc = Quaternion.from_euler(roll_end_acc, pitch_end_acc, 0.0)

        qdiff = qacc.prod(qgyr.conj())
        euldiff = qdiff.to_euler()
        
        z[2*i]     = euldiff[0]
        z[2*i + 1] = euldiff[1]
        
        prev_idx_motion_end = idx_motion_end
           
    if least_squares:
        return z.flatten()
    else:
        return z.flatten() @ z.flatten()