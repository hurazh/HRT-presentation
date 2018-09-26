import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
import sys


def demo_hrt(feature):

    class TrueConditional:
        def __init__(self, gmm):
            self.gmm = gmm

        def prob_quantiles(self, y, q, X=None):
            if isinstance(q, int) or isinstance(q, np.integer):
                return self.gmm.pdf(y)
            result = np.zeros((len(q), 500))
            result[:] = self.gmm.pdf(y)[:, None].T
            return result

        def sampler_prob(self, y, X=None):
            return self.prob_quantiles(y, 0, X=X)

        def sample(self, X=None):
            return self.gmm.sample()

        def __call__(self):
            return self.gmm.sample(), 1

    X = np.load('x_demo.npy')
    y = np.load('y_demo.npy')
    y = np.array([y])
    data_plot = np.concatenate((X, y.T), axis=1)
    columns = ('Feature 0', 'Feature 1', 'Feature 2', 'Feature 3', 'y')
    model = torch.load('model_demo.pt')
    true_conditional = torch.load('conditional_demo_feature{}.pt'.format(feature))

    fig, axs = plt.subplots(2, 1)
    fig.patch.set_visible(False)
    axs[0].axis('off')
    axs[0].axis('tight')
    data_plot = data_plot[:10]
    table_plot = axs[0].table(cellText=data_plot, colLabels=columns, loc='center')
    fig.tight_layout()

    tstat_hrt = lambda X_test: ((model.predict(X_test) - y) ** 2).mean()
    nperms = 200
    conditional = true_conditional
    lower = np.array([50])
    upper = np.array([50])
    quantiles = np.concatenate([lower, upper])
    conditional.quantiles = quantiles
    t_star = tstat_hrt(X)

    x_axis = np.linspace(0, nperms + 1, nperms)
    y_axis = np.repeat(t_star, nperms)
    axs[1].plot(x_axis, y_axis)
    plt.xlabel('Trails')
    plt.ylabel('Square Error')

    plt.ion()
    plt.show()
    plt.waitforbuttonpress()

    X_null = np.copy(X)
    t_true = tstat_hrt(X_null)
    t_null = np.zeros(nperms)
    quants_null = np.zeros((nperms, quantiles.shape[0]))
    t_weights = np.zeros((nperms, len(lower), len(upper)))
    to_delete = []
    for perm in range(nperms):
        plt.pause(0.01)
        table_plot.remove()
        # Sample from the conditional null model
        X_null[:, feature], quants_null[perm] = conditional()
        data_plot[:, feature] = X_null[:10, feature]
        table_plot = axs[0].table(cellText=data_plot, colLabels=columns, loc='center')
        # Get the test-statistic under the null
        t_null[perm] = tstat_hrt(X_null)
        for td in to_delete:
            td.remove()
        to_delete = []
        to_delete.append(axs[1].scatter(perm, t_null[perm], color='navy'))
        axs[1].scatter(perm, t_null[perm], color='navy', alpha=0.5, linewidths=0)
        if t_null[perm] <= t_true:
            # Over-estimate the likelihood
            t_weights[perm] = quants_null[perm, len(lower):][np.newaxis, :]
        else:
            # Under-estimate the likelihood
            t_weights[perm] = quants_null[perm, :len(lower)][:, np.newaxis]
    p_val = np.squeeze(t_weights[t_null <= t_true].sum(axis=0) / t_weights.sum(axis=0))
    matplotlib.rcParams.update({'font.size': 22})
    plt.title('Est. p_value: {}'.format(p_val))

    plt.waitforbuttonpress()


if __name__ == '__main__':
    class TrueConditional:
        def __init__(self, gmm):
            self.gmm = gmm

        def prob_quantiles(self, y, q, X=None):
            if isinstance(q, int) or isinstance(q, np.integer):
                return self.gmm.pdf(y)
            result = np.zeros((len(q), 500))
            result[:] = self.gmm.pdf(y)[:, None].T
            return result

        def sampler_prob(self, y, X=None):
            return self.prob_quantiles(y, 0, X=X)

        def sample(self, X=None):
            return self.gmm.sample()

        def __call__(self):
            return self.gmm.sample(), 1
    feature = 0
    demo_hrt(feature)
