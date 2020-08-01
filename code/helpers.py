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

def gravity_ned(latitude_rad, height): 
    R_0 = 6378137         # WGS84 Equatorial radius in meters 
    R_P = 6356752.31425   # WGS84 Polar radius in meters 
    e = 0.0818191908425   # WGS84 eccentricity 
    f = 1 / 298.257223563 # WGS84 flattening 
    mu = 3.986004418E14   # WGS84 Earth gravitational constant (m^3 s^-2) 
     
    sinsqL = np.sin(latitude_rad) ** 2 
    g_0 = 9.7803253359 * (1 + 0.001931853 * sinsqL) / np.sqrt(1 - e**2 * sinsqL) 
     
     
    return np.array([-8.08E-9 * height * np.sin(2 * latitude_rad), 
        0.0, 
        g_0 * (1 - (2 / R_0) * (1 + f * (1 - 2 * sinsqL) + (omega_ie**2 * R_0**2 * R_P / mu)) * height + (3 * height**2 / R_0**2))])