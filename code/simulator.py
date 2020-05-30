import numpy as np

from code.quaternion import *
from code.imu_model import sensor_error_model, misalignment

def slerp(q_from, q_to, fraction):
    theta = q_from.q.T @ q_to.q
    q = np.sin((1.0 - fraction) * theta) * q_from.q / np.sin(theta) \
      + np.sin(fraction * theta) / np.sin(theta) * q_to.q
    return Quaternion(q).normalized()

def generate_angular_velocity_between_two_orientations(q0, q1, dt):
    return 2.0 * q0.conj().prod(Quaternion(q1.q - q0.q)).v / dt

def generate_orientations(q_from, q_to, n_intermediary_points):
    return [slerp(q_from, q_to, fraction) 
        for fraction in np.linspace(0.0, 1.0, n_intermediary_points)]

def generate_imu_measurements_from_orientation_sequence(orientations, dt = 0.01):
    angular_velocities, accelerations = [], []

    o_prev = orientations[0]
    for o in orientations[0:]:
        angular_velocities.append(
            generate_angular_velocity_between_two_orientations(o_prev, o, dt))
        accelerations.append(
            o.Rm().T @ np.array([[0.0], [0.0], [9.81]]).flatten())
        o_prev = o

    return np.array(angular_velocities), np.array(accelerations)

def generate_imu_measurements_between_orientations(fromOrientation, toOrientation, n_intermediary_points, dt):
    orientations = generate_orientations(
        fromOrientation, toOrientation, n_intermediary_points)

    angular_velocities, accelerations = \
        generate_imu_measurements_from_orientation_sequence(orientations, dt)
    
    return angular_velocities, accelerations

def generate_ideal_imu_measurements_for_rotation_sequence(start_roll, start_pitch, start_yaw, 
    yaw_increment, N_samples = 100, 
    dt=0.01, randomized = False):

    zero_epoch_orientation = Quaternion.from_euler(start_roll, start_pitch, start_yaw)

    # generate first orientation chunk while the sensor was motionless
    ang_velocities, accelerations = generate_imu_measurements_between_orientations(
                zero_epoch_orientation, zero_epoch_orientation, N_samples, dt)
    standstill = np.ones(len(ang_velocities))
    
    
    roll, pitch, yaw = start_roll, start_pitch, start_yaw
    roll_prev, pitch_prev, yaw_prev = start_roll, start_pitch, start_yaw

    times = []
    for pitch in np.linspace(start_pitch + np.pi/4, start_pitch + np.pi, 4):
        for roll in np.linspace(start_roll, start_roll + 2 * np.pi, 6):
            if randomized:
                roll  += np.deg2rad(np.random.randint(60)) * np.random.choice([-1, 0, 1])
                pitch  += np.deg2rad(np.random.randint(45)) * np.random.choice([-1, 0, 1])
            yaw += yaw_increment
            
            # generate data through rotation
            fromOrientation = Quaternion.from_euler(roll_prev, pitch_prev, yaw_prev)
            toOrientation = Quaternion.from_euler(roll, pitch, yaw)
            ang, acc = generate_imu_measurements_between_orientations(
                fromOrientation, toOrientation, N_samples, dt)
            
            ang_velocities = np.concatenate((ang_velocities, ang), axis=0)
            accelerations = np.concatenate((accelerations, acc), axis=0)
            standstill = np.concatenate((standstill, np.zeros(N_samples)), axis=0)

            # generate data for standstill
            ang, acc = generate_imu_measurements_between_orientations(
                toOrientation, toOrientation, N_samples, dt)
            
            ang_velocities = np.concatenate((ang_velocities, ang), axis=0)
            accelerations = np.concatenate((accelerations, acc), axis=0)
            standstill = np.concatenate((standstill, np.ones(N_samples)), axis=0)
            
            roll_prev, pitch_prev, yaw_prev = roll, pitch, yaw
            
            if len(times):
                times.append(times[-1] + dt)
            else:
                times.append(0.0)

    return times, ang_velocities, accelerations, standstill

def corrupt_imu_measurements(theta_acc, theta_gyr, ang_velocities, accelerations,
    gyr_noise_std = 0.001, acc_noise_std = 0.04):
    ang_velocities = np.copy(ang_velocities)
    accelerations = np.copy(accelerations)
    # according to equations (1), (2), (5) and (6) in the paper
    Ma, ba = sensor_error_model(theta_acc[0:9])
    Mg, bg = sensor_error_model(theta_gyr[0:9])

    Rm_gyro_to_acc = misalignment(theta_gyr[-3:])
    for i, _ in enumerate(ang_velocities):
        ang_velocities[i] = Rm_gyro_to_acc @ (Mg @ ang_velocities[i] + bg + np.random.normal(0, gyr_noise_std, 3))
        accelerations[i] = Ma @ (accelerations[i]) + ba + np.random.normal(0, acc_noise_std, 3)
        
    return ang_velocities, accelerations