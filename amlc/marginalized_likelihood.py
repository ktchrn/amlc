"""
Docstring
"""


import numpy as np
from scipy.linalg import solve_triangular
from .covariance_matrix import DiagonalCovarianceMatrix, GeneralCovarianceMatrix


__all__ = ['MarginalizedLikelihood']


class MarginalizedLikelihood(object):
    """docstring for Marginalizer."""
    def __init__(self, y_obs, y_cov, A_m, A_b, L=None, LT=None, c_cov=None):
        super(MarginalizedLikelihood, self).__init__()
        #store inputs
        self.y = y_obs
        self.A_m = A_m
        self.A_b = A_b

        #store shapes
        self.n_y = self.y.size
        self.n_dth = self.A_m.shape[0]
        self.n_mnlp = self.A_m.shape[1]
        if self.A_b is not None:
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
                         + "types."
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
                             + " types."
                    raise ValueError(error)

            self.Lambda_inv = self.Lambda.get_inverse()
            self._partial_norm_const = (-0.5 * self.K.get_logdet()
                                        - 0.5 * self.Lambda.get_logdet()
                                        - 0.5 * self.n_y * np.log(2.*np.pi))

        self.B_prime = np.zeros([self.n_dth, self.n_nlp])
        self.B_prime[:, :self.n_mnlp] = A_m

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
                 mu_m,
                 mu_b,
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
    Docstring
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

    def get_unmarginalized_likelihood(self, c, d_theta, mu_m, mu_b):
        """
        Docstring
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
