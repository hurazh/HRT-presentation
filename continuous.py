import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from scipy.stats.mstats import gmean
from utils import batches, create_folds, logsumexp

############################################################
'''Continuous conditionals'''
############################################################
class GaussianMixtureModel:
    def __init__(self, pi, mu, sigma, y_mean=0, y_std=1):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.y_mean = y_mean
        self.y_std = y_std

    def sample(self):
        comps = [np.random.choice(self.pi.shape[1], p=p) for p in self.pi]
        return (np.array([np.random.normal(self.mu[i,k], self.sigma[i,k]) for i,k in enumerate(comps)]) * self.y_std) + self.y_mean

    def pdf(self, y):
        y = (y - self.y_mean) / self.y_std
        if len(y.shape) == 2:
            return (self.pi[:,np.newaxis,:] * norm.pdf(y[:,:,np.newaxis], self.mu[:,np.newaxis,:], self.sigma[:,np.newaxis,:])).sum(axis=2)    
        return (self.pi * norm.pdf(y[:,np.newaxis], self.mu, self.sigma)).sum(axis=1)

    def cdf(self, y):
        y = (y - self.y_mean) / self.y_std
        if len(y.shape) == 2:
            return (self.pi[:,np.newaxis,:] * norm.cdf(y[:,:,np.newaxis], self.mu[:,np.newaxis,:], self.sigma[:,np.newaxis,:])).sum(axis=2)    
        return (self.pi * norm.cdf(y[:,np.newaxis], self.mu, self.sigma)).sum(axis=1)

    def prob(self, y):
        return self.pdf(y)

    def ppf(self, c):
        raise Exception('Not implemented yet')

'''Neural conditional density estimator (GMM)'''
class MixtureDensityNetwork(nn.Module):
    def __init__(self, nfeatures, ncomponents, X_means, X_stds, y_mean, y_std):
        super(MixtureDensityNetwork, self).__init__()
        self.ncomponents = ncomponents
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.fc_in = nn.Sequential(
                nn.Linear(nfeatures, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, 3*ncomponents))
        # self.fc_in = nn.Sequential(nn.Linear(nfeatures,3*ncomponents))
        self.sigma_transform = nn.Softplus()
        self.pi_transform = nn.Softmax(dim=1)
    
    def forward(self, x):
        outputs = self.fc_in(x)
        pi = self.pi_transform(outputs[:,:self.ncomponents].clamp(-1e3,1e3))
        mu = outputs[:,self.ncomponents:2*self.ncomponents].clamp(-30,30)
        sigma = self.sigma_transform(outputs[:,2*self.ncomponents:]).clamp(1e-2, 1e2)
        return pi, mu, sigma

    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means[np.newaxis,:]) / self.X_stds[np.newaxis,:]), requires_grad=False)
        pi, mu, sigma = self.forward(tX)
        return GaussianMixtureModel(pi.data.numpy(), mu.data.numpy(), sigma.data.numpy(), y_mean=self.y_mean, y_std=self.y_std)


def clip_gradient(model, clip=5):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp(-clip,clip)

def fit_mdn(X, y, ncomponents=5,
                  nepochs=100, val_pct=0.1,
                  batch_size=None, target_batch_pct=0.01,
                  min_batch_size=10, max_batch_size=100,
                  verbose=False, lr=0.01, weight_decay=0.0, patience=3,
                  init_model=None, splits=None):
    import uuid
    tmp_file = '/Temp/tmp_file_' + str(uuid.uuid4())

    if batch_size is None:
        batch_size = max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features (helps with gradient propagation)
    Xstd = X.std(axis=0)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ystd = y.std()
    if ystd == 0:
        ystd = 1
    tX = autograd.Variable(torch.FloatTensor((X - X.mean(axis=0,keepdims=True)) / Xstd[np.newaxis, :]), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - y.mean()) / ystd), requires_grad=False)

    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    if init_model is not None:
        model = init_model
    else:
        model = MixtureDensityNetwork(X.shape[1], ncomponents, X.mean(axis=0), Xstd, y.mean(), ystd)

    # Save the model to file
    torch.save(model, tmp_file)

    # Setup the SGD method
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/(2**0.25))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())
    
    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            pi, mu, sigma = model(tX[tidx])

            # Calculate the log-probabilities
            components = torch.distributions.Normal(mu, sigma)
            logprobs = components.log_prob(tY[tidx][:,None])
            
            # -log(GMM(y | x)) loss
            loss = -logsumexp(pi.log() + logprobs, dim=1).mean()
            loss += 1e-2 * ((pi / (1-pi)).log()**2).mean()
            loss += 1e-2 * (mu**2).mean()
            loss += 1e-2 * (1/sigma**2).mean()
            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training conditional model.')
                if verbose:
                    print(prev_pi, prev_mu, prev_sigma)
                    print(pi, mu, sigma)
                break

            # Calculate gradients
            loss.backward()

            # Clip the gradients
            clip_gradient(model)

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data
            prev_pi, prev_mu, prev_sigma = pi, mu, sigma

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            pi, mu, sigma = model(tX[tidx])

            # Calculate the log-probabilities
            components = torch.distributions.Normal(mu, sigma)
            logprobs = components.log_prob(tY[tidx][:,None])
            
            # -log(GMM(y | x)) loss
            loss = -logsumexp(pi.log() + logprobs, dim=1).sum()

            # Track the loss
            validate_loss += loss.data

        train_losses[epoch] = train_loss.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.numpy() / float(len(validate_indices))

        # Adjust the learning rate down if the validation performance is bad
        scheduler.step(val_losses[epoch])
        if num_bad_epochs > patience:
            lr *= 0.5
            num_bad_epochs = 0

        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            init_model = None
            os.remove(tmp_file)
            return fit_mdn(X, y, ncomponents=ncomponents,
                                 nepochs=nepochs, val_pct=val_pct,
                                 batch_size=batch_size, target_batch_pct=target_batch_pct,
                                 min_batch_size=min_batch_size, max_batch_size=max_batch_size,
                                 verbose=verbose, lr=lr*0.5, weight_decay=weight_decay,
                                 patience=patience, init_model=init_model,
                                 splits=(train_indices, validate_indices))

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            torch.save(model, tmp_file)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    model = torch.load(tmp_file)
    os.remove(tmp_file)
    return model

'''Bootstrap confidence interval density estimator'''
class BootstrapConditionalModel:
    def __init__(self, X, models=None, fit_fn=fit_mdn, X_train=None, y_train=None, nbootstraps=100, quantiles=None, verbose=False):
        self.N = X.shape[0]
        assert (models is not None) or ((X_train is not None) and (y_train is not None))
        if models is None:
            self.indices = [np.random.choice(np.arange(X_train.shape[0]), replace=True, size=X_train.shape[0]) for _ in range(nbootstraps)]
            self.models = []
            for i,idx in enumerate(self.indices):
                if verbose:
                    print('\tBootstrap {}'.format(i))
                self.models.append(fit_fn(X_train[idx], y_train[idx]))
        else:
            self.models = models
        self.nbootstraps = len(self.models)
        self.quantiles = quantiles if quantiles is not None else np.array([50,50])
        self.dists = [m.predict(X) for m in self.models]

    def prob_quantiles(self, y, q, axis=0, X=None):
        if X is None:
            dists = self.dists
        else:
            dists = [m.predict(X) for m in self.models]
        probs = np.array([d.prob(y) for d in dists])
        return np.percentile(probs, q, axis=axis)

    def cdf_quantiles(self, y, q, axis=0, X=None):
        if X is None:
            dists = self.dists
        else:
            dists = [m.predict(X) for m in self.models]
        cdfs = np.array([d.cdf(y) for d in dists])
        return np.percentile(cdfs, q, axis=axis)

    def sample(self, X=None):
        if X is None:
            return self.dists[0].sample()
        return self.models[0].predict(X).sample()

    def sampler_prob(self, y, X=None):
        if X is None:
            return self.dists[0].prob(y)
        return self.models[0].predict(X).prob(y)

    def __call__(self):
        return sample_holdout_dists(self.dists, self.quantiles)

class CrossValidationBootstrapConditionalModel:
    def __init__(self, X, y, models, folds, quantiles=None):
        self.N = X.shape[0]
        self.y_shape = y.shape
        self.models = models
        self.folds = folds
        self.quantiles = quantiles if quantiles is not None else np.array([50,50])
        self.dists = [[m.predict(X[fold]) for m in model_set.models] for model_set, fold in zip(self.models, self.folds)]
        
    def prob_quantiles(self, y, q, axis=0, X=None):
        N = self.N if X is None else X.shape[0]
        if isinstance(q, int) or isinstance(q, np.integer):
            result = np.zeros(N)
        else:
            result = np.zeros((N, len(q)))
        for m, fold in zip(self.models, self.folds):
            result[fold] = m.prob_quantiles(y, q, axis=axis, X=X)
        return result

    def cdf_quantiles(self, y, q, axis=0, X=None):
        N = self.N if X is None else X.shape[0]
        if isinstance(q, int) or isinstance(q, np.integer):
            result = np.zeros(N)
        else:
            result = np.zeros((N, len(q)))
        for m, fold in zip(self.models, self.folds):
            result[fold] = m.cdf_quantiles(y, q, axis=axis, X=X)
        return result

    def sample(self, X=None):
        y = np.zeros(y.shape)
        for m, fold in zip(self.models, self.folds):
            y[fold] = m.sample(X=X[fold] if X is not None else None)
        return y

    def sampler_prob(self, y, X=None):
        prob = np.zeros(y.shape)
        for m, fold in zip(self.models, self.folds):
            prob[fold] = m.sampler_prob(X=X[fold] if X is not None else None)
        return y

    def __call__(self):
        y = np.zeros(self.N)
        probs = np.zeros(self.N)
        if self.quantiles is not None:
            quants = np.zeros((self.N, len(self.quantiles)))
        for fold, dist in zip(self.folds, self.dists):
            y[fold], q = sample_holdout_dists(dist, self.quantiles)
            if q is not None:
                quants[fold] = q
        return y, quants if self.quantiles is not None else None




def ks_test(ksstat, nsamples, ntrials=1000):
    null_stats = np.zeros(ntrials)
    null_cdf = (np.arange(nsamples)+1)/float(nsamples)
    for trial in range(ntrials):
        null_data = np.random.uniform(size=nsamples)
        null_data = null_data[np.argsort(null_data)]
        null_stats[trial] = np.max(np.abs(null_data - null_cdf))
    return (ksstat >= null_stats).mean()

def sample_holdout_dists(dists, quantiles):
    y = dists[0].sample()
    logprobs = np.log(np.array([d.prob(y) for d in dists]))
    if quantiles is None:
        return y, None
    probs = np.exp(logprobs - logprobs[0:1]) # likelihood ratio
    quants = np.percentile(probs, quantiles, axis=0) # quantile per-sample
    quants = gmean(quants, axis=1) # (geometric) mean quantile
    return y, quants

def calibrate_continuous(X, feature,
                         X_test=None, nquantiles=101, nbootstraps=100,
                         nfolds=5, ks_threshold=0.005, p_threshold=0.,
                         use_cv=False, fit_fn=None, verbose=0):
    '''Calibrates a bootstrap confidence interval conditional model for a given feature.'''
    # Search over a linear quantile grid to search
    quantile_range = np.linspace(0, 100, nquantiles)

    if fit_fn is None:
        from functools import partial
        fit_fn = partial(fit_mdn, verbose=verbose>1)

    jmask = np.ones(X.shape[1], dtype=bool)
    jmask[feature] = False
    if X_test is None and use_cv:
        # Use k-fold cross-validation to generate conditional density estimates for X_j
        if verbose:
            print('Fitting using {} bootstrap resamples and {} folds'.format(nbootstraps, nfolds))
        cdfs = np.zeros((nquantiles, X.shape[0]))
        proposals = []
        folds = create_folds(X, nfolds)
        for fold_idx, fold in enumerate(folds):
            imask = np.ones(X.shape[0], dtype=bool)
            imask[fold] = False
            model = BootstrapConditionalModel(X[fold][:,jmask], X_train=X[imask][:,jmask], y_train=X[imask][:,feature], nbootstraps=nbootstraps, verbose=verbose>1, fit_fn=fit_fn)
            cdfs[:,fold] = model.cdf_quantiles(X[fold][:,feature], quantile_range, axis=0)
            proposals.append(model)
        model = CrossValidationBootstrapConditionalModel(X[:,jmask], y, proposals, folds)
        # sampler = CrossValidationSampler(X[:,jmask], proposals, folds)
    else:
        if X_test is None:
            print('Using training set as testing set.')
            X_test = X
        # Use a held-out test set
        if verbose:
            print('Fitting using {} bootstrap resamples and a {}/{} train/test split'.format(nbootstraps, X.shape[0], X_test.shape[0]))
        model = BootstrapConditionalModel(X_test[:,jmask], X_train=X[:,jmask], y_train=X[:,feature], nbootstraps=nbootstraps, verbose=verbose, fit_fn=fit_fn)
        if verbose:
            print('Calculating all CDFs')
        cdfs = model.cdf_quantiles(X_test[:,feature], quantile_range, axis=0)
        if verbose:
            print(X_test[:,feature])
            print(cdfs)
            print(cdfs.shape)
        # sampler = HoldoutSampler(X_test[:,jmask], model)

    if verbose:
        print('Finding upper and lower quantiles')

    # Look at the bounds of the CDF along a discrete grid of points
    ks_grid = np.linspace(1e-6,1-1e-6,101)

    # Find the lower quantile that forms a sufficient upper bound on the uniform CDF
    for i in range(1,nquantiles//2):
        lower = quantile_range[nquantiles//2 - i]
        qlower = cdfs[nquantiles//2 - i]
        
        # U(0,1) CDF is the (0,1),(0,1) line. So at every point q on the grid of
        # CDF points, we expect a well-calibrated model to have q*N points with
        # CDF value lower than q. Here we are looking for an upper bound, so
        # we measure the KS distance as the maximum amount the U(0,1) CDF is
        # above the predicted CDF.
        ks_lower = 0
        for ks_point in ks_grid:
            ks_lower = max(ks_lower, ks_point - (qlower <= ks_point).mean())

        ks_pvalue = ks_test(ks_lower, cdfs.shape[1])
        # print('Lower: {} KS: {} p: {}'.format(lower, ks_lower, ks_pvalue))

        # Allow some error tolerance due to noise/finite data
        if ks_lower <= ks_threshold or ks_pvalue <= p_threshold:
            break

    # Find the upper quantile
    for i in range(1,nquantiles//2):
        upper = quantile_range[nquantiles//2+i]
        qupper = cdfs[nquantiles//2 + i]

        # U(0,1) CDF is the (0,1),(0,1) line. So at every point q on the grid of
        # CDF points, we expect a well-calibrated model to have q*N points with
        # CDF value lower than q. Here we are looking for a lower bound, so
        # we measure the KS distance as the maximum amount the U(0,1) CDF is
        # below the predicted CDF.
        ks_upper = 0
        for ks_point in ks_grid:
            ks_upper = max(ks_upper, (qupper <= ks_point).mean() - ks_point)

        ks_pvalue = ks_test(ks_upper, cdfs.shape[1])
        # print('Upper: {} KS: {} p: {}'.format(upper, ks_upper, ks_pvalue))

        # Allow some error tolerance due to noise/finite data
        if ks_upper <= ks_threshold or ks_pvalue <= p_threshold:
            break
        

    # Set the sampler to the chosen regions
    # sampler.quantiles = np.array([lower, upper])
    model.quantiles = np.array([lower, upper])

    # Our KS-distance is the worst-case of the two bounds
    ks_stat = np.max([ks_lower, ks_upper])

    # The p-value on the KS test that the bounded distribution is different
    # from the Uniform distribution
    ks_pvalue = ks_test(ks_stat, cdfs.shape[1])

    if verbose:
        print('Selected intervals: [{},{}]'.format(lower, upper))

    return {'model': model,
            'cdfs': cdfs,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'upper': upper,
            'lower': lower,
            'qupper': qupper,
            'qlower': qlower,
            'quantiles': quantile_range
            }

def test_mdn():
    # Generate the ground truth
    N = 1000
    X = np.random.normal(size=(1000,2))
    logits = np.array([np.exp(X[:,0]**2), np.exp(X[:,0]), np.exp(2*X[:,0])]).T
    pi = logits / logits.sum(axis=1, keepdims=True)
    # pi = np.array([np.ones(X.shape[0])*0.3, np.ones(X.shape[0])*0.5, np.ones(X.shape[0])*0.2]).T
    mu = np.array([X[:,0], 5*X[:,1], -2*X[:,1]*X[:,0]]).T + 100 # constant offset testing
    sigma = np.ones((X.shape[0],3))
    true_gmm = GaussianMixtureModel(pi, mu, sigma)

    # Sample some observations
    y = true_gmm.sample()
    truth = true_gmm.cdf(y)

    # import matplotlib.pylab as plt
    # x1, x2 = np.meshgrid(np.linspace(-5,5,100), np.linspace(-5,5,100))
    # im = np.zeros((100,100))
    # for i in range(100):
    #     for j in range(100):
    #         im[i,j] = 0.3*x2[i,j] + 0.5*5*x2[i,j] - 2 * x2[i,j]
    # plt.imshow(im)
    # plt.colorbar()
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.title('Mean(y)')
    # plt.show()


    # Fit the model
    split = int(np.round(X.shape[0]*0.8))
    model = fit_mdn(X[:split], y[:split], verbose=True, ncomponents=3, batch_size=100, nepochs=50)
        
    # Predict the likelihood of observations
    pred_gmm = model.predict(X)
    pred = pred_gmm.cdf(y)

    print(pred_gmm.sample().max(), pred_gmm.sample().min())

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.clf()
    plt.scatter(truth[split:], pred[split:], color='blue')
    plt.plot([0,1],[0,1],color='red')
    # z = np.linspace(y.min(), y.max(), 1000)
    # print(true_gmm.pi[0], true_gmm.mu[0], true_gmm.sigma[0])
    # print(pred_gmm.pi[0], pred_gmm.mu[0], pred_gmm.sigma[0])
    # plt.plot(z, (true_gmm.pi[0:1]*norm.pdf(z[:,np.newaxis], true_gmm.mu[0], true_gmm.sigma[0])).sum(axis=1), color='blue')
    # plt.plot(z, (pred_gmm.pi[0:1]*norm.pdf(z[:,np.newaxis], pred_gmm.mu[0], pred_gmm.sigma[0])).sum(axis=1), color='orange')
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    print('showing')
    # plt.ion()
    plt.show()
    # plt.hist(truth/pred, bins=100)
    # plt.show()

def test_calibration():
    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Generate the ground truth
    N = 200
    X = (np.random.normal(size=(N,1)) + np.random.normal(size=(N,200))) / 2
    logits = np.array([np.exp(X[:,0]**2 + X[:,1]**2), np.exp(X[:,0] + 2*X[:,2]), np.exp(2*X[:,0] + X[:,3])]).T
    pi = logits / logits.sum(axis=1, keepdims=True)
    # pi = np.array([np.ones(X.shape[0])*0.3, np.ones(X.shape[0])*0.5, np.ones(X.shape[0])*0.2]).T
    mu = np.array([X[:,0]-3*X[:,2], 5*X[:,1]-4*X[:,3], -2*X[:,1]*X[:,0]]).T
    sigma = np.ones((X.shape[0],3))
    true_gmm = GaussianMixtureModel(pi, mu, sigma)

    # Sample some observations of a third variable
    y = true_gmm.sample()
    truth = true_gmm.cdf(y)
    Xy = np.concatenate([X,y[:,np.newaxis]], 1)

    # Fit the calibrated model
    results = calibrate_continuous(Xy, Xy.shape[1]-1, nbootstraps=10, verbose=1)
    print(results)

    # look at the bounds of the CDF
    (model, cdfs,
     ks_stat, ks_pvalue,
     upper, lower,
     qupper, qlower,
     quantile_range) = (results['model'],
                        results['cdfs'],
                        results['ks_stat'],
                        results['ks_pvalue'],
                        results['upper'],
                        results['lower'],
                        results['qupper'], 
                        results['qlower'],
                        results['quantiles'])
    print('Quantile chosen: [{},{}] KS={} p={}'.format(lower, upper, ks_stat, ks_pvalue))

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
            
        plt.clf()
        plt.scatter(truth, qupper, color='blue', label='{:.0f}% quantile'.format(upper))
        plt.scatter(truth, qlower, color='orange', label='{:.0f}% quantile'.format(lower))
        for t,l,u in zip(truth, qlower, qupper):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        plt.plot([0,1],[0,1], color='red')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='upper left', prop=legend_props)
        plt.xlabel('Truth', fontsize=18)
        plt.ylabel('Estimated', fontsize=18)
        plt.savefig('plots/quantile-cdfs-scatter.pdf', bbox_inches='tight')
        plt.close()

        # Plot the confidence bands
        ks_grid = np.linspace(1e-4,1-1e-4,101)
        qlower = qlower[np.argsort(qlower)]
        qupper = qupper[np.argsort(qupper)]
        q50 = cdfs[101//2]
        q50 = q50[np.argsort(q50)]
        plt.plot(truth[np.argsort(truth)], np.arange(len(truth)) / float(len(truth)), color='black', lw=3, label='Truth')
        plt.plot(qlower, np.arange(len(qlower)) / float(len(qlower)), color='orange', lw=3, label='{:.0f}% quantile'.format(lower))
        # plt.plot(q50, np.arange(len(q50)) / float(len(q50)), color='green', lw=3, label='50% quantile')
        plt.plot(qupper, np.arange(len(qupper)) / float(len(qupper)), color='blue', lw=3, label='{:.0f}% quantile'.format(upper))
        plt.plot([0,1], [0,1], color='gray', lw=3, ls='--', label='U(0,1)')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='upper left', prop=legend_props)
        plt.xlabel('CDF value of observed X', fontsize=18)
        plt.ylabel('CDF of CDF', fontsize=18)
        plt.savefig('plots/quantile-cdfs-bands.pdf', bbox_inches='tight')
        plt.close()


def demo_calibration():
    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Generate the ground truth
    N = 100
    X = (np.random.normal(size=(N,1)) + np.random.normal(size=(N,10))) / 2
    logits = np.array([np.exp(X[:,0]**2 + X[:,1]**2), np.exp(X[:,0] + 2*X[:,2]), np.exp(2*X[:,0] + X[:,3])]).T
    pi = logits / logits.sum(axis=1, keepdims=True)
    # pi = np.array([np.ones(X.shape[0])*0.3, np.ones(X.shape[0])*0.5, np.ones(X.shape[0])*0.2]).T
    mu = np.array([X[:,0]-3*X[:,2], 5*X[:,1]-4*X[:,3], -2*X[:,1]*X[:,0]]).T
    sigma = np.ones((X.shape[0],3))
    true_gmm = GaussianMixtureModel(pi, mu, sigma)

    # Sample some observations of a third variable
    y = true_gmm.sample()
    truth = true_gmm.cdf(y)
    truth_weights = 1 / true_gmm.pdf(y)
    truth_weights /= truth_weights.sum()
    Xy = np.concatenate([X,y[:,np.newaxis]], 1)

    class NormalDist:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma

        def sample(self):
            return np.random.normal(self.mu, self.sigma)

        def pdf(self, y):
            return norm.pdf(y, self.mu, self.sigma)

        def cdf(self, y):
            return norm.cdf(y, self.mu, self.sigma)

        def prob(self, y):
            return self.pdf(y)

        def ppf(self, c):
            raise Exception('Not implemented yet')

    class LinearPredictor:
        def __init__(self, weights, offset, sigma):
            self.weights = weights
            self.offset = offset
            self.sigma = sigma

        def predict(self, X):
            return NormalDist((self.weights[None,:] * X).sum(axis=1) + self.offset, self.sigma)

    fit_linear = lambda X_, y_: LinearPredictor(np.linalg.pinv(X_.T.dot(X_)).dot(X_.T.dot((y_ - y_.mean())/ y_.std())), y_.mean(), y_.std())


    # Fit the calibrated model
    results = calibrate_continuous(Xy, Xy.shape[1]-1, nbootstraps=100, verbose=1, fit_fn=fit_linear)
    print(results)

    # look at the bounds of the CDF
    (model, cdfs,
     ks_stat, ks_pvalue,
     upper, lower,
     qupper, qlower,
     quantile_range) = (results['model'],
                        results['cdfs'],
                        results['ks_stat'],
                        results['ks_pvalue'],
                        results['upper'],
                        results['lower'],
                        results['qupper'], 
                        results['qlower'],
                        results['quantiles'])
    print('Quantile chosen: [{},{}] KS={} p={}'.format(lower, upper, ks_stat, ks_pvalue))
    pupper, plower = 1 / model.prob_quantiles(y, [upper,lower])
    pupper_linear = 2*pupper / (pupper + plower).sum()
    plower_linear = 2*plower / (pupper + plower).sum()
    upper_linear = upper
    lower_linear = lower

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
            
        plt.clf()
        plt.scatter(truth, qupper, color='blue', label='{:.0f}% quantile'.format(upper))
        plt.scatter(truth, qlower, color='orange', label='{:.0f}% quantile'.format(lower))
        for t,l,u in zip(truth, qlower, qupper):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        plt.plot([0,1],[0,1], color='red')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='lower right', prop=legend_props)
        plt.xlabel('Truth', fontsize=18)
        plt.ylabel('Estimated', fontsize=18)
        plt.savefig('plots/calibration-scatter-linear.pdf', bbox_inches='tight')
        plt.close()

        plt.clf()
        plt.scatter(truth_weights, pupper_linear, color='blue', label='{:.0f}% quantile'.format(upper))
        plt.scatter(truth_weights, plower_linear, color='orange', label='{:.0f}% quantile'.format(lower))
        for t,l,u in zip(truth_weights, plower_linear, pupper_linear):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        legend_props = {'weight': 'bold', 'size': 14}
        plt.plot([0,np.max([pupper_linear, truth_weights])],[0,np.max([pupper_linear, truth_weights])], color='red')
        plt.legend(loc='upper right', prop=legend_props)
        plt.xlabel('True relative sample weight', fontsize=18)
        plt.ylabel('Estimated relative bounds', fontsize=18)
        plt.xlim([0,truth_weights.max() + 0.01])
        plt.ylim([0,np.max([plower_linear, pupper_linear, truth_weights]) + 0.01])
        plt.savefig('plots/calibration-weights-linear.pdf', bbox_inches='tight')
        plt.close()

        # Plot the confidence bands
        ks_grid = np.linspace(1e-4,1-1e-4,101)
        qlower = qlower[np.argsort(qlower)]
        qupper = qupper[np.argsort(qupper)]
        q50 = cdfs[101//2]
        q50 = q50[np.argsort(q50)]
        plt.plot(truth[np.argsort(truth)], np.arange(len(truth)) / float(len(truth)), color='black', lw=3, label='Truth')
        plt.plot(qlower, np.arange(len(qlower)) / float(len(qlower)), color='orange', lw=3, label='{:.0f}% quantile'.format(lower))
        # plt.plot(q50, np.arange(len(q50)) / float(len(q50)), color='green', lw=3, label='50% quantile')
        plt.plot(qupper, np.arange(len(qupper)) / float(len(qupper)), color='blue', lw=3, label='{:.0f}% quantile'.format(upper))
        plt.plot([0,1], [0,1], color='gray', lw=3, ls='--', label='U(0,1)')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='lower right', prop=legend_props)
        plt.xlabel('CDF value of observed X', fontsize=18)
        plt.ylabel('CDF of CDF', fontsize=18)
        plt.savefig('plots/calibration-bands-linear.pdf', bbox_inches='tight')
        plt.close()

    # Fit the calibrated model
    results = calibrate_continuous(Xy, Xy.shape[1]-1, nbootstraps=100, verbose=1)
    print(results)

    # look at the bounds of the CDF
    (model, cdfs,
     ks_stat, ks_pvalue,
     upper, lower,
     qupper, qlower,
     quantile_range) = (results['model'],
                        results['cdfs'],
                        results['ks_stat'],
                        results['ks_pvalue'],
                        results['upper'],
                        results['lower'],
                        results['qupper'], 
                        results['qlower'],
                        results['quantiles'])
    print('Quantile chosen: [{},{}] KS={} p={}'.format(lower, upper, ks_stat, ks_pvalue))
    pupper, plower = 1 / model.prob_quantiles(y, [upper,lower])
    pupper_nonlinear = 2*pupper / (pupper + plower).sum()
    plower_nonlinear = 2*plower / (pupper + plower).sum()

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
            
        plt.clf()
        plt.scatter(truth, qupper, color='blue', label='{:.0f}% quantile'.format(upper))
        plt.scatter(truth, qlower, color='orange', label='{:.0f}% quantile'.format(lower))
        for t,l,u in zip(truth, qlower, qupper):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        plt.plot([0,1],[0,1], color='red')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='lower right', prop=legend_props)
        plt.xlabel('Truth', fontsize=18)
        plt.ylabel('Estimated', fontsize=18)
        plt.savefig('plots/calibration-scatter-nonlinear.pdf', bbox_inches='tight')
        plt.close()

        # Plot the confidence bands
        ks_grid = np.linspace(1e-4,1-1e-4,101)
        qlower = qlower[np.argsort(qlower)]
        qupper = qupper[np.argsort(qupper)]
        q50 = cdfs[101//2]
        q50 = q50[np.argsort(q50)]
        plt.plot(truth[np.argsort(truth)], np.arange(len(truth)) / float(len(truth)), color='black', lw=3, label='Truth')
        plt.plot(qlower, np.arange(len(qlower)) / float(len(qlower)), color='orange', lw=3, label='{:.0f}% quantile'.format(lower))
        # plt.plot(q50, np.arange(len(q50)) / float(len(q50)), color='green', lw=3, label='50% quantile')
        plt.plot(qupper, np.arange(len(qupper)) / float(len(qupper)), color='blue', lw=3, label='{:.0f}% quantile'.format(upper))
        plt.plot([0,1], [0,1], color='gray', lw=3, ls='--', label='U(0,1)')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='lower right', prop=legend_props)
        plt.xlabel('CDF value of observed X', fontsize=18)
        plt.ylabel('CDF of CDF', fontsize=18)
        plt.savefig('plots/calibration-bands-nonlinear.pdf', bbox_inches='tight')
        plt.close()

        # Nonlinear
        plt.clf()
        plt.scatter(truth_weights, pupper_nonlinear, color='blue', label='{:.0f}% quantile'.format(upper))
        plt.scatter(truth_weights, plower_nonlinear, color='orange', label='{:.0f}% quantile'.format(lower))
        for t,l,u in zip(truth_weights, plower_nonlinear, pupper_nonlinear):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        legend_props = {'weight': 'bold', 'size': 14}
        plt.plot([0,np.max([plower_linear, plower_nonlinear, truth_weights])],[0,np.max([plower_linear, plower_nonlinear, truth_weights])], color='red')
        plt.legend(loc='upper right', prop=legend_props)
        plt.xlabel('True relative sample weight', fontsize=18)
        plt.ylabel('Estimated relative bounds', fontsize=18)
        plt.xlim([0,truth_weights.max() + 0.01])
        plt.ylim([0,np.max([plower_linear, plower_nonlinear, truth_weights]) + 0.01])
        plt.savefig('plots/calibration-weights-nonlinear.pdf', bbox_inches='tight')
        plt.close()

        # Re-plot linear
        plt.clf()
        plt.scatter(truth_weights, pupper_linear, color='blue', label='{:.0f}% quantile'.format(upper_linear))
        plt.scatter(truth_weights, plower_linear, color='orange', label='{:.0f}% quantile'.format(lower_linear))
        for t,l,u in zip(truth_weights, plower_linear, pupper_linear):
            plt.plot([t,t],[l,u], color='gray', alpha=0.5)
        legend_props = {'weight': 'bold', 'size': 14}
        plt.plot([0,np.max([plower_linear, plower_nonlinear, truth_weights])],[0,np.max([plower_linear, plower_nonlinear, truth_weights])], color='red')
        plt.legend(loc='upper right', prop=legend_props)
        plt.xlabel('True relative sample weight', fontsize=18)
        plt.ylabel('Estimated relative bounds', fontsize=18)
        plt.xlim([0,truth_weights.max() + 0.01])
        plt.ylim([0,np.max([plower_linear, plower_nonlinear, truth_weights]) + 0.01])
        plt.savefig('plots/calibration-weights-linear.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    demo_calibration()


