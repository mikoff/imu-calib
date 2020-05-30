import numpy as np
import argparse

from code.monte_carlo import monte_carlo_cycle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run monte carlo simulations for accelerometer and gyroscope calibration.')
    parser.add_argument('--sampling_frequency', help = 'Sampling frequency for synthesized IMU measurements.', 
        required = False, default = 100, type = float)
    parser.add_argument('--randomize_rotations', help = 'Flag to whether or not randomize the rotations.',
        required = False, default = True, type = bool)
    parser.add_argument('--plot', help = 'Flag to whether or not plot the results.',
        required = False, default = False, type = bool)
    parser.add_argument('--iterations_num', help = 'Number of Monte-carlo runs.',
        required = False, default = 1, type = int)
    args = parser.parse_args()

    N_samples = args.sampling_frequency
    dt = 1 / args.sampling_frequency
    randomize = args.randomize_rotations
    plot = args.plot

    simulation_results = {}

    for i in range(0, args.iterations_num):
        theta_acc = np.random.randint(100, size=9)  / np.array([1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 
                                                                1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 100* np.random.choice([-1,1]), 100* np.random.choice([-1,1]), 100* np.random.choice([-1,1])])
        theta_gyr = np.random.randint(100, size=12) / np.array([1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 
                                                                1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1]), 1000* np.random.choice([-1,1])])
                                                                    
        simulation_results[i] = monte_carlo_cycle(N_samples, dt, theta_acc, theta_gyr, randomize, plot)