import libs.parameters as pr
from libs.dmp_original import dmp


def main():

    ep_len = int(pr.TOTAL_TIME/pr.SAMPLING_TIME)

    for seed in pr.SEED_LIST:
        logger_kwargs = dict(output_dir=pr.TRAINING_DIR+'dmp/seed=' + str(seed), exp_name='dmp')
        dmp(epochs=1, trials_per_epoch=50, steps_per_epoch=ep_len, theta_update_rate=250,
            zeta=1000, eps_var=0.05, logger_kwargs=logger_kwargs, seed=seed)


if __name__ == '__main__':
    main()
