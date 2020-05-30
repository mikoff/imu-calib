import matplotlib.pyplot as plt
import numpy as np

from code.quaternion import Quaternion
from code.helpers import integrate_gyroscope

def generate_standstill_flags(imu_data):
    standstill = np.zeros(len(imu_data))
    counter_after_motion = 60
    # generate standstill flags
    for i, m in enumerate(imu_data):
        if np.linalg.norm(m[3:6]) < 0.13:
            counter_after_motion += 1
            if counter_after_motion > 60:
                standstill[i] = 1
        else:
            standstill[i] = 0
            standstill[i-60:i] = 0
            counter_after_motion = 0

    return standstill

def plot_corrupted_accelerometer_and_gyro_measurements(corrupted_accelerometer_measurements,
        corrupted_gyroscope_measurements, standstill_flags):

    idxs_standstill = standstill_flags > 0
    idxs_motion = standstill_flags < 1e-8

    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    ax[0].plot(np.linalg.norm(corrupted_accelerometer_measurements, axis=1), label = "Accelerometer norm")
    ax[0].plot(np.arange(0, len(standstill_flags))[idxs_standstill], 
        standstill_flags[idxs_standstill] + 10.0, 
        linestyle='none', marker='.', alpha=0.9, color='green', label = "Standstill")
    ax[0].plot(np.arange(0, len(standstill_flags))[idxs_motion], 
        standstill_flags[idxs_motion] + 10.0, 
        linestyle='none', marker='.', alpha=0.9, color='red', label = "In motion")
    ax[0].set(xlabel = "sample #", ylabel="$m/s^2$")
    ax[0].set_title("Accelerometer measurements norm and standstill during simulated rotations")
    ax[0].legend()

    ax[1].plot(corrupted_gyroscope_measurements)
    ax[1].set_title("Gyroscope measurements during simulated rotations")
    ax[1].set(xlabel = "sample #", ylabel="$rad/s$")
    ax[1].legend(["x", "y", "z"])
    plt.tight_layout()
    plt.show()

def plot_calibrated_and_uncalibrated_acc_norms(uncalibrated, calibrated):
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(np.linalg.norm(uncalibrated, axis=1))
    ax.plot(np.linalg.norm(calibrated, axis=1))
    ax.legend(["Uncalibrated norm", "Calibrated norm"])
    ax.set(xlabel = "sample #", ylabel = "$m/s^2$", title = "Calibration results for accelerometer")
    plt.tight_layout()
    plt.show()

def plot_ideal_corrupted_calibrated_measurements(ideal_gyroscope, corrupted_gyroscope_measurements, gyr_calibrated):
    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    ax[0].plot(ideal_gyroscope[:, 0], label = 'ideal')
    ax[0].plot(corrupted_gyroscope_measurements[:, 0], label = 'corrupted')
    ax[0].plot(gyr_calibrated[:, 0], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    ax[0].legend()
    ax[0].set(xlabel = "sample #", ylabel = "$rad/sec$", title = "Calibration results for gyroscope X axis")

    ax[1].plot(ideal_gyroscope[:, 1], label = 'ideal')
    ax[1].plot(corrupted_gyroscope_measurements[:, 1], label = 'corrupted')
    ax[1].plot(gyr_calibrated[:, 1], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    ax[1].legend()
    ax[1].set(xlabel = "sample #", ylabel = "$rad/sec$", title = "Calibration results for gyroscope Y axis")

    ax[2].plot(ideal_gyroscope[:, 2], label = 'ideal')
    ax[2].plot(corrupted_gyroscope_measurements[:, 2], label = 'corrupted')
    ax[2].plot(gyr_calibrated[:, 2], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    ax[2].legend()
    ax[2].set(xlabel = "sample #", ylabel = "$rad/sec$", title = "Calibration results for gyroscope Z axis")

    plt.tight_layout()
    plt.show()

def plot_imu_data_and_standstill(imu_data, standstill_flags):
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(imu_data[:, 0:3])
    ax[0].legend(["ax", "ay", "az"])

    ax[1].plot(imu_data[:, 3], label = "wx")
    ax[1].plot(imu_data[:, 4], label = "wx")
    ax[1].plot(imu_data[:, 5], label = "wx")
    ax[1].plot(standstill_flags)

    idxs_standstill = standstill_flags > 0
    idxs_motion = standstill_flags < 1e-8
    ax[1].plot(np.arange(0, len(standstill_flags))[idxs_standstill], standstill_flags[idxs_standstill], 
        label = "standstill", linestyle='none', marker='.', alpha=0.9, color='green')
    ax[1].plot(np.arange(0, len(standstill_flags))[idxs_motion], standstill_flags[idxs_motion], 
        label = "in-motion", linestyle='none', marker='.', alpha=0.9, color='red')
    ax[1].legend()

    plt.show()

def plot_accelerations_before_and_after(accs, accs_calibrated):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    times = np.arange(0, len(accs_calibrated)) * 0.01
    ax.plot(times, np.linalg.norm(accs, axis=1), alpha = 0.5)
    ax.plot(times, np.linalg.norm(accs_calibrated, axis=1), alpha = 0.5)
    ax.legend(["Uncalibrated norm", "Calibrated norm"])
    ax.set(xlabel='$time, s$', ylabel='$m/s^2$', ylim = [8.81, 10.81])
    plt.show()
