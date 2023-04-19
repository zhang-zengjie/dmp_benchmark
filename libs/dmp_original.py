import numpy as np
import math
from libs.commons import calculate_dmp_basis
import libs.parameters as pr
from libs.commons import nm, calculate_reward_dmp_with_obstacle
from libs.logx import EpochLogger

Ts = pr.SAMPLING_TIME
T = pr.TOTAL_TIME
alpha = pr.DMP_ALPHA
beta = pr.DMP_BETA
gamma = pr.RL_GAMMA
omega = pr.DMP_OMEGA
tau = pr.DMP_TAU

dim = pr.ORIGINAL_DMP_DIM
DoF = pr.DOF

init_pos_fixed = pr.INIT_POS_FIXED
goal_pos_fixed = pr.GOAL_POS_FIXED
obst_pos_fixed = pr.OBST_POS_FIXED


def dmp(epochs=100, trials_per_epoch=50, steps_per_epoch=500, theta_update_rate=15, zeta=10, eps_var=0.01, logger_kwargs=dict(), seed=0):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    np.random.seed(seed)

    theta = np.zeros([dim, ], dtype=np.float32)
    s = np.zeros([steps_per_epoch+1, ], dtype=np.float32)
    s[0] = 1
    phi = np.zeros([steps_per_epoch, dim], dtype=np.float32)
    psi = np.zeros([steps_per_epoch, dim], dtype=np.float32)

    for i in range(steps_per_epoch):
        phi[i, :], psi[i, :] = calculate_dmp_basis(s[i])
        s[i+1] = s[i] - tau*Ts*omega*s[i]

    goal_pos_x = np.random.uniform(goal_pos_fixed[0] - 0.05, goal_pos_fixed[0] + 0.05)
    goal_pos_y = np.random.uniform(goal_pos_fixed[1] - 0.05, goal_pos_fixed[1] + 0.05)
    goal_pos_z = np.random.uniform(goal_pos_fixed[2] - 0.01, goal_pos_fixed[2] + 0.01)

    obst_pos_x = np.random.uniform(obst_pos_fixed[0] - 0.05, obst_pos_fixed[0] + 0.05)
    obst_pos_y = np.random.uniform(obst_pos_fixed[1] - 0.05, obst_pos_fixed[1] + 0.05)
    obst_pos_z = np.random.uniform(obst_pos_fixed[2] - 0.01, obst_pos_fixed[2] + 0.01)

    goal_pos = np.array([goal_pos_x, goal_pos_y, goal_pos_z])
    obst_pos = np.array([obst_pos_x, obst_pos_y, obst_pos_z])

    interaction_counter = 0
    for ep in range(epochs):
        S = np.zeros([steps_per_epoch, trials_per_epoch], dtype=np.float32)
        P = np.zeros([steps_per_epoch, trials_per_epoch], dtype=np.float32)
        rho = np.zeros([dim, trials_per_epoch], dtype=np.float32)
        delta = np.zeros([dim, trials_per_epoch], dtype=np.float32)
        for k in range(trials_per_epoch):

            g = np.zeros([steps_per_epoch, dim], dtype=np.float32)
            Mek = np.zeros([steps_per_epoch, dim], dtype=np.float32)
            x = np.zeros([steps_per_epoch+1, DoF], dtype=np.float32)
            xdot = np.zeros([steps_per_epoch+1, DoF], dtype=np.float32)
            r = np.zeros([steps_per_epoch+1, ], dtype=np.float32)
            x[0] = init_pos_fixed

            for i in range(steps_per_epoch):
                for j in range(dim):
                    g[i, j] = psi[i, j]*s[i]*nm(goal_pos - x[0])
                epsilon = np.random.multivariate_normal(mean=np.zeros([dim, ], dtype=np.float32), cov=eps_var * np.identity(dim))
                g_vec = np.array([g[i, :]], dtype=np.float32)
                temp = (np.array([epsilon], dtype=np.float32) @ np.transpose(g_vec) @ g_vec) / (g_vec @ np.transpose(g_vec))
                Mek[i, :] = temp[0]
                x[i+1] = x[i] + Ts * xdot[i]
                acc = alpha * (beta * (goal_pos - x[i]) - xdot[i]) + g[i, :] @ (theta + epsilon)
                xdot[i+1] = xdot[i] + Ts * tau * acc
                r[i], ter, trn = calculate_reward_dmp_with_obstacle(i+1, x[i+1], acc, goal_pos, obst_pos)
                interaction_counter += 1
                if ter | trn:
                    break

            for i in range(steps_per_epoch):
                S[i, k] = math.exp(sum(r[i:steps_per_epoch+1])/(zeta*steps_per_epoch))

            for j in range(dim):
                rho[j, k] = 0
                for i in range(steps_per_epoch):
                    rho[j, k] = rho[j, k] + (steps_per_epoch-i)*phi[i, j]*S[i, k]*Mek[i, j]

        for i in range(steps_per_epoch):
            P[i, :] = S[i, :]/sum(S[i, :])

        for j in range(dim):
            num = 0
            for i in range(steps_per_epoch):
                num = num + (steps_per_epoch-i)*phi[i, j]*sum(S[i, :])
            delta[j, :] = rho[j, :]/num
            theta[j] += theta_update_rate*sum(delta[j, :])

        # Test after update
        g = np.zeros([steps_per_epoch, dim], dtype=np.float32)
        x = np.zeros([steps_per_epoch + 1, DoF], dtype=np.float32)
        xdot = np.zeros([steps_per_epoch + 1, DoF], dtype=np.float32)
        r = np.zeros([steps_per_epoch + 1, ], dtype=np.float32)
        x[0] = init_pos_fixed

        for i in range(steps_per_epoch):
            for j in range(dim):
                g[i, j] = psi[i, j] * s[i] * nm(goal_pos - x[0])
            x[i + 1] = x[i] + Ts * xdot[i]
            acc = alpha * (beta * (goal_pos - x[i]) - xdot[i]) + g[i, :] @ theta
            xdot[i + 1] = xdot[i] + Ts * tau * acc
            r[i], ter, trn = calculate_reward_dmp_with_obstacle(i + 1, x[i + 1], acc, goal_pos, obst_pos)
            if ter | trn:
                break
            logger.store(EpRet=sum(r))

        logger.log_tabular('Epoch', ep)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', interaction_counter)

        for i in range(len(theta)):
            logger.log_tabular('theta_'+str(i+1), theta[i])

        logger.dump_tabular()
