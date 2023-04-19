import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import libs.parameters as pr


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.arange(0, height_z, 0.05)
    theta = np.arange(0, 2*np.pi, 0.05)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


option_dir = 'dmp'

if __name__ == '__main__':

    data_dir = pr.TEST_FIXED_GOAL_DIR + option_dir
    datasets = []
    for root, _, files in os.walk(data_dir):
        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            datasets.append(exp_data)

    InitPos_x = datasets[0].InitPos_1[0]
    InitPos_y = datasets[0].InitPos_2[0]
    InitPos_z = datasets[0].InitPos_3[0]
    GoalPos_x = datasets[0].GoalPos_1[0]
    GoalPos_y = datasets[0].GoalPos_2[0]
    GoalPos_z = datasets[0].GoalPos_3[0]
    ObstPos_x = datasets[0].ObstPos_1[0]
    ObstPos_y = datasets[0].ObstPos_2[0]
    ObstPos_z = datasets[0].ObstPos_3[0]

    fig = plt.figure(figsize=(3, 2.5))
    ax = plt.axes(projection='3d')

    counter = 0
    for item in datasets:
        CurrPos_x = item.Pos_1.to_numpy()
        CurrPos_y = item.Pos_2.to_numpy()
        CurrPos_z = item.Pos_3.to_numpy()
        counter = counter + 1
        Rew = item.TestEpRet.to_numpy()[-1]

        # CurrPos = np.transpose([CurrPos_x, CurrPos_y, CurrPos_z])
        if Rew < -120:
        # if counter in [1, 4, 6, 8, 9]: # For no BC
            ax.plot3D(CurrPos_x, CurrPos_y, CurrPos_z, color=(255 / 255, 51 / 255, 51 / 255), linewidth=1)
        else:
            ax.plot3D(CurrPos_x, CurrPos_y, CurrPos_z, color=(21 / 255, 21 / 255, 81 / 255), linewidth=1)

    ax.plot(InitPos_x, InitPos_y, InitPos_z, alpha=0.8, color=(0.2, 0.5, 1), marker="o", markersize=8)
    ax.plot(GoalPos_x, GoalPos_y, GoalPos_z, alpha=0.8, color=(128/255, 0, 127/255), marker="*", markersize=10)

    Xc, Yc, Zc = data_for_cylinder_along_z(ObstPos_x, ObstPos_y, 0.035, ObstPos_z)
    ax.plot_surface(Xc, Yc, Zc, color=(255/255, 212/255, 128/255))

    plt.subplots_adjust(top=0.98, bottom=0.155, left=0.0, right=0.9)

    # 3d range wrt cam #
    ax.set_xlim3d(-0.1, 0.5)
    ax.set_ylim3d(0, 0.5)
    ax.set_zlim3d(0, 0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # plt.savefig('test' + option_dir + '.eps', dpi=600)

    plt.show()
