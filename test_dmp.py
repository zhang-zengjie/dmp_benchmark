import numpy as np
import libs.parameters as pr
from libs.logx import EpochLogger
from libs.commons import nm, calculate_reward_dmp_with_obstacle
import pandas as pd
import os
from libs.commons import calculate_dmp_basis
import copy


steps_per_epoch = int(pr.TOTAL_TIME/pr.SAMPLING_TIME)

Ts = pr.SAMPLING_TIME
T = pr.TOTAL_TIME
alpha = pr.DMP_ALPHA
beta = pr.DMP_BETA
gamma = pr.RL_GAMMA
omega = pr.DMP_OMEGA
tau = pr.DMP_TAU

dim = pr.ORIGINAL_DMP_DIM
DoF = pr.DOF

options = {'init_pos': np.array(pr.INIT_POS_FIXED, dtype=np.float32),
           'goal_pos': np.array(pr.GOAL_POS_FIXED, dtype=np.float32) + np.array([0.02, -0.01, 0.01], dtype=np.float32),
           'obst_pos': np.array(pr.OBST_POS_FIXED, dtype=np.float32) + np.array([-0.01, -0.02, 0], dtype=np.float32)}

dirs = 'dmp/'

for seed in pr.SEED_LIST:

    logger = EpochLogger(output_dir=pr.TEST_FIXED_GOAL_DIR + dirs + 'seed=' + str(seed))

    exp_data = pd.read_table(os.path.join(pr.TRAINING_DIR + dirs + 'seed=' + str(seed) + '/progress.txt'))
    theta = [exp_data.theta_1.to_numpy()[-1],
             exp_data.theta_2.to_numpy()[-1],
             exp_data.theta_3.to_numpy()[-1],
             exp_data.theta_4.to_numpy()[-1],
             exp_data.theta_5.to_numpy()[-1],
             exp_data.theta_6.to_numpy()[-1],
             exp_data.theta_7.to_numpy()[-1],
             exp_data.theta_8.to_numpy()[-1],
             exp_data.theta_9.to_numpy()[-1],
             exp_data.theta_10.to_numpy()[-1]]

    s = np.zeros([steps_per_epoch + 1, ], dtype=np.float32)
    s[0] = 1
    phi = np.zeros([steps_per_epoch, dim], dtype=np.float32)
    psi = np.zeros([steps_per_epoch, dim], dtype=np.float32)

    for i in range(steps_per_epoch):
        phi[i, :], psi[i, :] = calculate_dmp_basis(s[i])
        s[i + 1] = s[i] - tau * Ts * omega * s[i]

    g = np.zeros([steps_per_epoch, dim], dtype=np.float32)
    x, xdot = copy.copy(options['init_pos']), np.zeros([DoF, ])
    ter, tru, ep_ret, ep_len = 0, 0, 0, 0
    while not (ter or tru or (steps_per_epoch == ep_len)):
        for j in range(dim):
            g[ep_len, j] = psi[ep_len, j] * s[ep_len] * nm(options['goal_pos'] - options['init_pos'])
        acc = tau * alpha * (beta * (options['goal_pos'] - x) - xdot) + g[ep_len, :] @ theta
        x += Ts * xdot
        xdot += Ts * acc
        r, ter, trn = calculate_reward_dmp_with_obstacle(ep_len, x, acc, options['goal_pos'], options['obst_pos'])
        ep_ret += r
        ep_len += 1
        logger.log_tabular('Time', ep_len * pr.SAMPLING_TIME)
        for dix in range(3):
            logger.log_tabular('InitPos_' + str(dix+1), options['init_pos'][dix])
            logger.log_tabular('GoalPos_' + str(dix+1), options['goal_pos'][dix])
            logger.log_tabular('ObstPos_' + str(dix+1), options['obst_pos'][dix])
            logger.log_tabular('Pos_' + str(dix+1), x[dix])
            logger.log_tabular('Vel_' + str(dix+1), xdot[dix])
        logger.log_tabular('TestEpRet', ep_ret)
        logger.dump_tabular()
