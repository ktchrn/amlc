__all__ = ['DiagonalCovarianceMatrix', 'GeneralCovarianceMatrix']

import numpy as np
from scipy.linalg import cholesky, cho_solve


class CovarianceMatrix(object):
    """
    A template class for covariance matrices.

    CovarianceMatrix defines a unified interface for interacting with
    covariance matrices. Subclasses need to implement left multiplication by
    the inverse of the covariance matrix (the apply_inverse method), at the
    very least.


    Attributes
    ----------
    shape : list
        Shape of the covariance matrix represented by the class.
    """

    def __init__(self, *args):
        super(CovarianceMatrix, self).__init__()
        self.shape = None
        self._logdet = None
        self._inverse = None

    def apply_inverse(self, B):
        pass

    def get_inverse(self):
        if self._inverse is None:
            self._inverse = self._compute_explicit_inverse()
        return self._inverse

    def add_inverse(self, B):
        #default: just add it
        B += self.get_inverse()

    def get_logdet(self):
        if self._logdet is None:
            self._logdet = self._compute_logdet()
        return self._logdet

    def _compute_explicit_inverse(self):
        #default behavior: invert an indentity matrix of appropriate shape
        id = np.identity(self.shape[0])
        return self.apply_inverse(id)


class DiagonalCovarianceMatrix(CovarianceMatrix):
    """
    A covariance matrix with no off-diagonal terms.

    Parameters
    __________
    variance_vector : one-dimensional array-like
        Diagonal terms of a covariance matrix. Should all be strictly positive.
    """
    def __init__(self, variance_vector):
        super(DiagonalCovarianceMatrix, self).__init__()
        self._variances = np.atleast_1d(np.asarray(variance_vector, dtype=np.double))
        if len(self._variances.shape) != 1:
            raise ValueError("variance_vector has more than one non-trivial dimension.")
        elif (self._variances <= 0).any():
            raise ValueError("variance_vector cannot contain zeros.")
        self.shape = self._variances.shape
        self._precisions = 1. / self._variances

    def apply_inverse(self, B):
        B = np.asarray(B, dtype=np.double)
        if len(B.shape) == 1:
            return self._precisions * B
        elif len(B.shape) == 2:
            return self._precisions[:, None] * B
        else:
            raise ValueError("Cannot apply inverse to a value that is not a 1D vector or 2D matrix.")

    def add_inverse(self, B):
        np.fill_diagonal(B, B.diagonal() + self._precisions)

    def _compute_logdet(self):
        return np.sum(np.log(self._variances))

    def _compute_explicit_inverse(self):
        return np.diag(self._precisions)


class GeneralCovarianceMatrix(CovarianceMatrix):
    """
    A covariance matrix without special structure.

    Parameters
    ----------
    covariance_matrix : two-dimensional array-like
        An arbitrary covariance matrix. Should be positive definite.
    """
    def __init__(self, covariance_matrix):
        super(GeneralCovarianceMatrix, self).__init__()
        covariance_matrix = np.asarray(covariance_matrix, dtype=np.double)
        self._cho_lower = False
        self._cho_cov = cholesky(covariance_matrix, self._cho_lower, overwrite_a=False)
        self.shape = self._cho_cov.shape

    def apply_inverse(self, B):
        return cho_solve((self._cho_cov, self._cho_lower), B)

    def _compute_logdet(self):
        return 2. * np.sum(np.log(np.diagonal(self._cho_cov)))
