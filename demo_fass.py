import numpy as np
import matplotlib.pyplot as plt
import sys


def demo_fass(feature, sample):
    X_tilda = np.load('X_tilda_feature{}.npy'.format(feature))
    T_tilda = np.load('T_tilda_feature{}.npy'.format(feature))
    prob = np.load('prob_feature{}.npy'.format(feature))
    t_max = np.max(T_tilda[sample])

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(X_tilda[sample], T_tilda[sample], color='navy')
    # axs[0].set_xlabel('X feature: {} sample: {}'.format(feature, sample))
    axs[0].set_ylabel('t')
    # axs[0].hlines(t_true_sum, X_tilda[1, 0], X_tilda[1, -1], linestyles='--')
    axs[1].plot(X_tilda[sample], prob[sample], color='navy')
    axs[1].set_xlabel('X feature: {} sample: {}'.format(feature, sample))
    axs[1].set_ylabel('pdf')

    plt.ion()
    plt.show()
    plt.waitforbuttonpress()

    t_arr = np.linspace(0, t_max, 50)
    to_delete = []
    prob_list = []
    for t_iteration in t_arr:
        plt.pause(0.5)
        for td in to_delete:
            td.remove()
        to_delete = []
        to_delete.append(axs[0].hlines(t_iteration, X_tilda[1, 0], X_tilda[1, -1]))
        fill_index = T_tilda[sample] <= t_iteration
        x_fill = X_tilda[sample][fill_index]
        y_fill = T_tilda[sample][fill_index]
        prob_fill = prob[sample][fill_index]
        to_delete.append(axs[0].fill_between(x_fill, y_fill, color='navy', alpha=0.5))
        to_delete.append(axs[1].fill_between(x_fill, prob_fill, color='navy', alpha=0.5))
    plt.waitforbuttonpress()


if __name__ == '__main__':
    feature, sample = 0, 1
    demo_fass(feature, sample)
