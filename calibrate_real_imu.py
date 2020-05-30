import numpy as np
import argparse
import time

from code.helpers import *
from code.cost_functions import *
from code.utilities import *

np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run calibration on real data from IMU.')
    parser.add_argument('--sampling_frequency', help = 'Sampling frequency for logfile.', 
        required = True, type = int)
    parser.add_argument('--file', help = 'Path to file with data from IMU.',
        required = True, type = str)
    args = parser.parse_args()

    dt = 1 / args.sampling_frequency
    datafile = args.file

    # read file with ax, ay, az, wx, wy, wz measurements from IMU
    imu_data = np.genfromtxt(datafile, delimiter=' ')
    standstill = generate_standstill_flags(imu_data)

    plot_imu_data_and_standstill(imu_data, standstill)

    accs, angs = imu_data[:,0:3], imu_data[:,3:6]

    # find accelerometer calibration parameters and calibrate accel measurements
    theta_found_acc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    time_start = time.time()
    theta_found_acc = find_calib_params_acc(True, residual_acc, theta_found_acc, accs, standstill > 0)
    time_end = time.time()

    print("ACC calibration done in: ", time_end - time_start, "seconds")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z   ]")
    print(theta_found_acc)
    accs_calibrated = calibrate_accelerometer(accs, theta_found_acc)
    plot_accelerations_before_and_after(accs, accs_calibrated)


    # find gyroscope calibration parameters
    theta_found_gyr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta_found_gyr[-6:-3] = np.mean(angs[0:100,:], axis=0)

    residualSum = lambda: np.sum(np.rad2deg(residual_gyr(theta_found_gyr, 
             angs, 
             accs_calibrated,
             standstill, dt))**2)

    print("Gyroscope residuals before calibration: ", residualSum())
    time_start = time.time()
    theta_found_gyr = find_calib_params_gyr(True, residual_gyr, theta_found_gyr, 
        angs, accs_calibrated, standstill, 1.0/args.sampling_frequency)
    time_end = time.time()
    print("GYR calibration done in: ", time_end - time_start, "seconds")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z     E_X     E_Y     E_Z  ]")
    print(theta_found_gyr)
    print("Gyroscope residuals after calibration: ", residualSum())