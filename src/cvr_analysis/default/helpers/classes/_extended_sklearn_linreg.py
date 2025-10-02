from sklearn import linear_model
from scipy import stats
import numpy as np


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, fit_intercept = False):
        super().__init__(fit_intercept = fit_intercept)

    def fit(self, X, y, n_jobs=1):
        self = super().fit(X, y, n_jobs)
        self.pred = self.predict(X)
        self.dof = (X.shape[0] - X.shape[1])

        sse = np.sum((self.pred - y) ** 2, axis=0) / self.dof
        Q = np.linalg.pinv(np.dot(X.T, X))
        if sse.ndim == 0:
            se = np.sqrt(np.diagonal(sse * Q))
        else:
            se = np.array([
                            np.sqrt(np.diagonal(sse[i] * Q))
                            for i in range(sse.shape[0])
                        ]).T
        self.se = se
        self.t = self.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), self.dof))
        
        return self