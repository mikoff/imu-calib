import numpy as np
import scipy.optimize as optimize

from code.imu_model import sensor_error_model, misalignment
from code.quaternion import Quaternion

def find_calib_params_acc(least_squares, residual, theta, accelerations, idxs_standstill):
    '''
    Find calibration parameters according to cost function from eq. (10)
    For details see residual_acc function inside cost_functions.py
    '''
    res = optimize.least_squares(residual, theta, 
                                 args=(accelerations,
                                       idxs_standstill,
                                       least_squares,),
                                 max_nfev = 25,
                                 x_scale = [10., 10., 10., 10., 10., 10., 1., 1., 1.],
                                 method='trf', loss='soft_l1',
                                 bounds = [
                                       (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -1.1, -1.1, -1.1),
                                       (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  1.1,  1.1,  1.1)],
        )
    return res.x

def find_calib_params_gyr(least_squares, residual, theta, ang_velocities, accelerations, standstill_flags, dt):
    '''
    Find calibration parameters according to cost function from eq. (16)
    For details see residual_gyr function inside cost_functions.py
    '''
    res = optimize.least_squares(
            residual, 
            theta, 
            args=(ang_velocities, 
                  accelerations,
                  standstill_flags,
                  dt,
                  least_squares,),
            max_nfev = 50,
            x_scale = [10., 10., 10., 10., 10., 10., 10., 10., 10., 10, 10, 10],
            method='trf', loss='soft_l1',
            bounds = [
                    (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11),
                    (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11)],
        )
    return res.x

def calibrate_accelerometer(measurements, sensor_errors):
    '''
    Calibrate sensor measurements according to the model.
    See eq. (7).
    '''
    M, b = sensor_error_model(sensor_errors[0:9])
    C = np.linalg.inv(M)

    calibrated_measurements = np.copy(measurements)
    for measurement in calibrated_measurements:
        measurement[:] = C @ (measurement - b)

    return calibrated_measurements

def calibrate_gyroscope(measurements, sensor_errors, epsilon):
    '''
    Calibrate sensor measurements according to the model.
    See eq. (8).
    '''
    M, b = sensor_error_model(sensor_errors[0:9])
    C = np.linalg.inv(M)
    Rm = np.linalg.inv(misalignment(epsilon))

    calibrated_measurements = np.copy(measurements)
    for measurement in calibrated_measurements:
        measurement[:] = C @ Rm @ measurement - C @ b

    return calibrated_measurements

def integrate_gyroscope(roll, pitch, yaw, measurements, dt, return_as = 'array'):
    q = Quaternion.from_euler(roll, pitch, yaw)
    orientations = []
    for w in measurements:
        q = q.prod(Quaternion.exactFromOmega((w) * dt))
        if return_as == 'array':
            orientations.append(q.q)
        else:
            orientations.append(q)
    return orientations