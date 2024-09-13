import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_curve(env, n_logs=5, evaluate_length=1e4, total=1e6):
    min_l = int(total // evaluate_length) + 1
    ret_list = []
    for s in range(n_logs):
        df = pd.read_csv('saves/' + str(env+1) + '/log' + str(s+1) + '.csv')
        ret = df[['return']].to_numpy().transpose(1, 0)[0]
        if len(ret) < min_l:
            min_l = len(ret)
        for i in range(len(ret) - 1):
            ret[i + 1] = ret[i] * 0.9 + ret[i + 1] * 0.1
        ret_list.append(ret)
    data = np.zeros((n_logs, min_l))
    for s in range(n_logs):
        data[s, :] = ret_list[s][:min_l]
    mean = np.mean(data, axis=0)
    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)
    stamps = np.array([i * (evaluate_length / 1e6) for i in range(min_l)])
    plt.plot(stamps, mean, label='PPO', lw=1.0, c='crimson')
    plt.fill_between(stamps, mini, maxi, alpha=0.2, color='crimson')
    plt.title(env_list[env])
    plt.xlabel('number of environment steps (x $\mathregular{10^6}$)')
    plt.ylabel('return')
    plt.xlim(0, total // 1e6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    env_list = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
    mpl.style.use('seaborn-v0_8')
    for env in range(4):
        plot_curve(env, n_logs=5)
