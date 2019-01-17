__all__ = ['MarginalizedLikelihood']

import numpy as np
from scipy.linalg import solve_triangular

from .covariance_matrix import DiagonalCovarianceMatrix, GeneralCovarianceMatrix


class MarginalizedLikelihood(object):
    """
    A class for computing continuum parameter-marginalized likelihoods.

    MarginalizedLikelihood stores parameters of a given analysis problem
    and, when called, uses them to compute marginalized likelihoods and
    related quantities.

    Parameters
    ----------
    y_obs : one-dimensional array-like
        The observed spectrum that is being analyzed.

    y_cov : one-dimensional array-like, two-dimensional array-like, or
                object implementing the CovarianceMatrix interface
        The data covariance matrix. Can be a CovarianceMatrix or an array that
        can be converted to a CovarianceMatrix. Both dimensions of y_cov should
        have the same length as y_obs.

    A_m : two-dimensional array-like
        The continuum design matrix.

    A_b : two-dimensional array-like or None (optional)
        The foreground design matrix. If there is no linear foreground term,
        A_b should be None.

    L : an array, LinearOperator, or other object implementing the matrix
            multiplication interface (optional)
        The line spread function. If the line spread function is trivial,
        L should be None.

    LT : same as L (optional)
        The transpose or adjoint of L, in case if this is not available as L.T.

    c_cov : same as y_cov (optional)
        Covariance matrix of the prior on the continuum parameters. To use the
        improper uniform prior, c_cov should be None.

    Methods
    -------
    __call__(d_theta, mu_m, mu_b, keyword arguments)
        Access point for the marginalized likelihood and its gradient as well
        as the conditional distribution of the continuum and foreground
        parameters.

    get_unmarginalized_likelihood(c, d_theta, mu_m, mu_b)
        The unmarginalized likelihood. Requires specifying a set of continuum
        and foreground parameters.

    """
    def __init__(self, y_obs, y_cov, A_m, A_b=None, L=None, LT=None, c_cov=None):
        super(MarginalizedLikelihood, self).__init__()
        #store inputs
        self.y = np.squeeze(np.asarray(y_obs))
        self.A_m = np.asarray(A_m)  #has to be a matrix
        self.A_b = A_b  #could be None

        #store shapes
        self.n_y = self.y.size
        self.n_dth = self.A_m.shape[0]
        self.n_mnlp = self.A_m.shape[1]
        if self.A_b is not None:
            self.A_b = np.asarray(self.A_b)
            self.n_anlp = self.A_b.shape[1]
        else:
            self.n_anlp = 0
        self.n_nlp = self.n_mnlp + self.n_anlp

        #Set data covariance matrix
        if hasattr(y_cov, 'get_logdet') and hasattr(y_cov, 'apply_inverse'):
            self.K = y_cov
        else:
            y_cov = np.atleast_1d(np.asarray(y_cov, dtype=np.double))
            if len(y_cov.shape) == 1:
                self.K = DiagonalCovarianceMatrix(y_cov)
            elif len(y_cov.shape) == 2:
                self.K = GeneralCovarianceMatrix(y_cov)
            else:
                error = ("Shape " + str(y_cov.shape) + " of y_cov is "
                         + "incompatible with the built-in CovarianceMatrix "
                         + "types.")
                raise ValueError(error)

        #Set prior and pre-compute part of normalization constant
        if c_cov is None:
            #assume prior on c is improper and uniform
            self._partial_norm_const = (-0.5 * self.K.get_logdet()
                                        - (0.5 * (self.n_y-self.n_nlp)
                                           * np.log(2.*np.pi)))
            self.Lambda = None
        else:
            #assume prior on c is proper and Gaussian
            if (hasattr(c_cov, 'get_logdet') and hasattr(c_cov, 'apply_inverse')
                and hasattr(c_cov, 'get_inverse')
                and hasattr(c_cov, 'add_inverse')):
                self.Lambda = c_cov
            else:
                c_cov = np.atleast_1d(np.asarray(c_cov, dtype=np.double))
                if len(c_cov.shape) == 1:
                    self.Lambda = DiagonalCovarianceMatrix(c_cov)
                elif len(c_cov.shape) == 2:
                    self.Lambda = GeneralCovarianceMatrix(c_cov)
                else:
                    error = ("Shape " + str(c_cov.shape) + " of c_cov is "
                             + "incompatible with the built-in CovarianceMatrix"
                             + " types.")
                    raise ValueError(error)

            self.Lambda_inv = self.Lambda.get_inverse()
            self._partial_norm_const = (-0.5 * self.K.get_logdet()
                                        - 0.5 * self.Lambda.get_logdet()
                                        - 0.5 * self.n_y * np.log(2.*np.pi))

        self.B_prime = np.zeros([self.n_dth, self.n_nlp])
        self.B_prime[:, :self.n_mnlp] = self.A_m

        self.L = L
        if L is None:
            if self.n_y != self.n_dth:
                error = "L cannot be None if length of y != length of d_theta."
                raise ValueError(error)
            self.LT = self.L
        else:
            if LT is None:
                self.LT = self.L.T
            else:
                self.LT = LT

    def __call__(self,
                 d_theta,
                 mu_m=0,
                 mu_b=0,
                 return_logp=False,
                 return_grad_logp=False,
                 return_cmu=False,
                 return_cmu_cov=False,
                 return_c_draws=False,
                 jac_d_theta=None,
                 jac_mu_m=None,
                 jac_mu_b=None,
                 n_c_draws=1):
        """
        Access point for the marginalized likelihood and its gradient, and
        characteristics and samples from the conditional distribution of the
        continuum and foreground parameters.

        Parameters
        ----------
        d_theta : one-dimensional numpy array
            Transmittances to multiply the continuum by.

        mu_m : one-dimensional numpy array or number (optional)
            The mean of the continuum model. Set to 0 by default.

        mu_b : one-dimensional numpy array or number (optional)
            The mean of the foreground model. Set to 0 by default.

        return_logp : boolean (optional)
            Whether to return the logarithm of the marginalized loglikelihood.

        return_grad : boolean (optional)
            Whether to return the gradient of the logarithm of the marginalized
            loglikelihood.

        return_cmu : boolean (optional)
            Whether to return the conditional mean of the linear continuum and
            foreground parameters.

        return_cmu_cov : boolean (optional)
            Whether to return the conditional covariance of the linear continuum
            and foreground parameters.

        return_c_draws : boolean (optional)
            Whether to return draws from the conditional distribution of the
            linear continuum and foreground parameters.

        jac_d_theta: two-dimensional numpy array, number, or None (optional)
            Derivatives of the transmittance with respect to the parameters
            of interest. If d_theta does not depend on these parameters,
            jac_d_theta should be set to 0. To compute the gradient, values for
            all three Jacobians must be given.

        jac_mu_m : as jac_d_theta (optional)
            As jac_d_theta, but for the mean of the continuum model.

        jac_mu_b : as jac_d_theta (optional)
            As jac_d_theta, but for the mean of the foreground model.

        n_c_draws : integer (optional)
            Number of draws to return from the conditional distribution of the
            linear continuum and foreground parameters.

        Returns
        -------
        return_dict : dictionary
            return_dict contains an entry for each return_(option)=True with key
            (option). If only return_logp=True, return_dict = {'logp':(logp)}.

        """
        if not (return_logp or return_grad_logp or return_cmu or return_cmu_cov
                or return_c_draws):
            return {}

        if return_grad_logp:
            no_jdth = jac_d_theta is None
            no_jmum = jac_mu_m is None
            no_jmub = jac_mu_b is None
            if no_jdth or no_jmum or no_jmub:
                error = ""
                if no_jdth:
                    error += "jac_d_theta, "
                if no_jmum:
                    error += "jac_mu_m, "
                if no_jmub:
                    error += "jac_mu_b, "
                error = error[:-2] + "not given, can't compute grad_logp."
                raise ValueError(error)

        #compute combinations needed by all possible requests
        B = np.empty([self.n_dth, self.n_nlp])
        np.multiply(d_theta[:, None], self.A_m, out=B[:, :self.n_mnlp])
        if self.A_b is not None:
            B[:, self.n_mnlp:] = self.A_b

        if self.L is None:
            L_B = B
            Kinv_L_B = self.K.apply_inverse(L_B)
            r = self.y - (d_theta*mu_m + mu_b)
            Kinv_r = self.K.apply_inverse(r)
            LT_Kinv_r = Kinv_r
        else:
            L_B = self.L @ B
            Kinv_L_B = self.K.apply_inverse(L_B)
            r = self.y - self.L @ (d_theta*mu_m + mu_b)
            Kinv_r = self.K.apply_inverse(r)
            LT_Kinv_r = self.LT @ (Kinv_r)

        C = L_B.T @ Kinv_L_B
        if self.Lambda is not None:
            self.Lambda.add_inverse(C)
        C = covariance_matrix.GeneralCovarianceMatrix(C)
        cmu = C.apply_inverse(B.T @ LT_Kinv_r)

        return_dict = {}

        if return_logp:
            #combine computed pieces into logp
            return_dict['logp'] = self._logp(C, r, Kinv_r, L_B, cmu)

        if return_grad_logp:
            if self.L is not None:
                LT_Kinv_L_B = self.LT @ Kinv_L_B
            else:
                LT_Kinv_L_B = Kinv_L_B
            LT_Kinv_r_m_rmu = LT_Kinv_r - LT_Kinv_L_B@cmu

            #start with scalar grad_logp and use non-inplace addition
            #for generality; this will not be a major issue
            grad_logp = 0
            if jac_d_theta is not 0:
                grad_logp_d_theta = self._grad_logp_wrt_d_theta(LT_Kinv_r_m_rmu,
                                                                cmu, mu_m, C,
                                                                LT_Kinv_L_B)
                grad_logp = jac_d_theta @ grad_logp_d_theta
            if jac_mu_m is not 0:
                grad_logp = grad_logp + jac_mu_m @ (d_theta * LT_Kinv_r_m_rmu)
            if jac_mu_b is not 0:
                grad_logp = grad_logp + jac_mu_b @ (LT_Kinv_r_m_rmu)
            return_dict['grad_logp'] = grad_logp

        if return_cmu:
            return_dict['cmu'] = cmu

        if return_cmu_cov:
            return_dict['cmu_cov'] = C.get_inverse()

        if return_c_draws:
            uncorr_c_draws = np.random.normal(0, 1, [self.n_nlp, n_c_draws])
            c_draws = solve_triangular(C._cho_cov, uncorr_c_draws,
                                       lower=C._cho_lower)
            c_draws = c_draws.T + cmu
            return_dict['c_draws'] = c_draws

        return return_dict

    def _logp(self, C, r, Kinv_r, L_B, cmu):
        rmu = L_B @ cmu
        logp = -0.5 * (np.sum(Kinv_r * (r - rmu)) + C.get_logdet())
        logp += self._partial_norm_const
        return logp

    def _grad_logp_wrt_d_theta(self, LT_Kinv_r_m_rmu, cmu, mu_m, C, LT_Kinv_L_B):
        BT_LT_Kinv_L = LT_Kinv_L_B.T
        Cinv_BpT = C.apply_inverse(self.B_prime.T)
        Cinv_BT_LT_Kinv_L = C.apply_inverse(BT_LT_Kinv_L)
        grad_logp = LT_Kinv_r_m_rmu * (self.B_prime @ cmu + mu_m)
        grad_logp += -0.5 * np.sum(Cinv_BpT * BT_LT_Kinv_L, axis=0)
        grad_logp += -0.5 * np.sum(Cinv_BT_LT_Kinv_L * self.B_prime.T, axis=0)
        return grad_logp

    def get_unmarginalized_likelihood(self, c, d_theta, mu_m=0, mu_b=0):
        """
        The unmarginalized log-likelihood.

        Parameters
        ----------
        c : one-dimensional numpy array
            Linear continuum and foreground parameters.

        d_theta : one-dimensional numpy array
            Transmittances to multiply the continuum by.

        mu_m : one-dimensional numpy array or number (optional)
            The mean of the continuum model. Set to 0 by default.

        mu_b : one-dimensional numpy array or number (optional)
            The mean of the foreground model. Set to 0 by default.

        Returns
        -------
        logp : number
            The unmarginalized log-likelihood given the supplied parameters.
        """
        B = np.empty([self.n_dth, self.n_nlp])
        np.multiply(d_theta[:, None], self.A_m, out=B[:, :self.n_mnlp])
        if self.A_b is not None:
            B[:, self.n_mnlp:] = self.A_b

        y_model = (B @ c + d_theta * mu_m + mu_b)
        if not (self.L is None):
            y_model = self.L @ y_model
        r = self.y - y_model
        logp = -0.5 * np.sum(r * self.K.apply_inverse(r))
        if not (self.Lambda is None):
            logp += -0.5 * np.sum(c * self.Lambda.apply_inverse(c))
        logp += self._partial_norm_const
        logp += 0.5 * self.n_nlp * np.log(2. * np.pi)
        return logp
