import numpy as np
import time

from code.simulator import *
from code.cost_functions import residual_acc, residual_gyr
from code.helpers import *
from code.utilities import *

np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})


def monte_carlo_cycle(N_samples, dt, theta_acc, theta_gyr, randomized_rotations, plot):
    fail = 0
    
    simulation_results = {}
    simulation_results['theta_acc_generated'] = theta_acc
    simulation_results['theta_gyr_generated'] = theta_gyr

    # start orientation
    roll, pitch, yaw = np.deg2rad([-150, -75, -150])
    yaw_increment = np.deg2rad(18)

    # generate ideal measurements
    times, ideal_gyroscope, ideal_accelerometer, standstill_flags = \
        generate_ideal_imu_measurements_for_rotation_sequence(
            roll, pitch, yaw, yaw_increment, N_samples, dt, randomized_rotations)

    # restore true oreintations from ideal measurements
    orientations_true = integrate_gyroscope(roll, pitch, yaw, ideal_gyroscope, dt)

    # corrupt imu measurements according to sensor errors
    corrupted_gyroscope_measurements, corrupted_accelerometer_measurements = \
        corrupt_imu_measurements(theta_acc, theta_gyr, ideal_gyroscope, ideal_accelerometer, 0.004, 0.04)
    
    if plot:
        plot_corrupted_accelerometer_and_gyro_measurements(corrupted_accelerometer_measurements,
            corrupted_gyroscope_measurements, standstill_flags)

    # find accelerometer calibration parameters and calibrate accel measurements
    theta_found_acc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    time_start = time.time()
    theta_found_acc = find_calib_params_acc(True, residual_acc, theta_found_acc, 
        corrupted_accelerometer_measurements, standstill_flags > 0)
    time_end = time.time()

    print("ACC calibration done in: ", time_end - time_start, "seconds")
    print("First row - true parameters, second - their estimate, third - their difference")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z   ]")
    print(theta_acc)
    print(theta_found_acc)
    print(np.abs(theta_found_acc - theta_acc))
    accs_calibrated = calibrate_accelerometer(corrupted_accelerometer_measurements, theta_found_acc)

    if plot:
        plot_calibrated_and_uncalibrated_acc_norms(
            corrupted_accelerometer_measurements,
            accs_calibrated)


    # find gyroscope calibration parameters
    theta_found_gyr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    residualSum = lambda: np.sum(np.rad2deg(residual_gyr(theta_found_gyr, 
             corrupted_gyroscope_measurements, 
             accs_calibrated,
             standstill_flags,
             dt) ** 2))

    print("Gyroscope residuals before calibration: ", residualSum())

    theta_found_gyr[-6:-3] = np.mean(corrupted_gyroscope_measurements[0:N_samples,:], axis=0)
    gyr_calibrated_bias_only = calibrate_gyroscope(corrupted_gyroscope_measurements, theta_gyr[0:9], theta_gyr[-3:])
    print("Gyroscope residuals with only bias calibrated: ", residualSum())

    time_start = time.time()
    theta_found_gyr = find_calib_params_gyr(True, residual_gyr, theta_found_gyr, 
        corrupted_gyroscope_measurements, accs_calibrated, standstill_flags, dt)
    time_end = time.time()

    print("GYR calibration done in: ", time_end - time_start, "seconds")
    print("First row - true parameters, second - their estimate, third - their difference")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z     E_X     E_Y     E_Z  ]")
    print(theta_gyr)
    print(theta_found_gyr)
    print(np.abs(theta_found_gyr - theta_gyr))

    print("Gyroscope residuals after calibration: ", residualSum())
    
    if residualSum() > 1.0:
        fail = 1
        print("Calibration failed!")

    gyr_calibrated = calibrate_gyroscope(corrupted_gyroscope_measurements, theta_gyr[0:9], theta_gyr[-3:])
    if plot:
        plot_ideal_corrupted_calibrated_measurements(ideal_gyroscope, corrupted_gyroscope_measurements, gyr_calibrated)
        
    simulation_results['theta_acc_found'] = theta_found_acc
    simulation_results['theta_gyr_found'] = theta_found_gyr
    simulation_results['residual_deg'] = residualSum()
    simulation_results['fail'] = fail
    
    return simulation_results