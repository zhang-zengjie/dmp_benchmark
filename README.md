# Dynamic Movement Primitive Benchmark

## Change parameters

Go to parameter file `libs/parameters.py` and change the parameters accordingly:
- Change the simulation parameters;
- Change the DMP hyper-parameters;
- Change the artificial potential field parameters;
- Change the initial and goal positions of the point, and the position of the static obstacle;
- Change the seed list to run multiple trainings.

## Train DMP

Run `train_dmp.py` with proper parameters
- `epochs`: the number of epochs;
- `trials_per_epoch`: the number of exploring trajectories per epoch;
- `steps_per_epoch`: the number of steps per trial;
- `theta_update_rate`: the learning rate of policy;
- `zeta`: heuristic parameter;
- `eps_var`: the variance of noise.
The training results are stored in `../data/training/dmp` by default.

## Test DMP

Run `text_dmp.py` to test the trained dmp model
- The test results are stored in `../data/test_with_fixed_goal/dmp` by default;
- Run `draw_test_fixed_goal.py` to visualize the test trajectory.
