import numpy as np
import os

# Simulation parameters
SAMPLING_TIME = 0.02
TOTAL_TIME = 5     # 12 # 10
DOF = 3

# DMP parameters
DMP_ALPHA = 10      # 8 # 10
DMP_BETA = 1.2      # 0.6 # 2.5
DMP_OMEGA = 6       # 4 # 8 # 20
RL_GAMMA = 0.99
DMP_TAU = 0.25

ORIGINAL_DMP_BASIS_CENTERS = np.array([1, 0.6294, 0.3962, 0.2494, 0.1569,
                                       0.0988, 0.0622, 0.0391, 0.0246, 0.0155], dtype=np.float32)
ORIGINAL_DMP_BASIS_VARIANCE = np.array([41.6667, 16.5359, 6.5359, 2.5840, 1.0235,
                                        0.4054, 0.1606, 0.0636, 0.0252, 0.0252], dtype=np.float32) / 100
ORIGINAL_DMP_DIM = 10

# Collision avoidance APF parameters
COLLISION_TOLERANCE = 0.05
COLLISION_GAIN = 0.001
COLLISION_FIELD_RANGE = 0.08
OBST_RADIUS = 0.035

# Positions
INIT_POS_FIXED = [0, 0, 0.05]
GOAL_POS_FIXED = [0.30, 0.35, 0.08]
OBST_POS_FIXED = [0.14, 0.16, 0.16]

# Critical directories to store results
SAVE_DATA_DIR = os.getcwd() + '/../data/'
BUFFER_DIR = SAVE_DATA_DIR + 'buffer/'
TRAINING_DIR = SAVE_DATA_DIR + 'training/'
TEST_DIR = SAVE_DATA_DIR + 'test/'
TEST_FIXED_GOAL_DIR = SAVE_DATA_DIR + 'test_fixed_goal/'
GOAL_FOR_TEST_DIR = SAVE_DATA_DIR + 'goals/'

# Seed list
SEED_LIST = [0]
