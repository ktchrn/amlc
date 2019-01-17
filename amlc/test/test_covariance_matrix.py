import pytest
import numpy as np
from ..covariance_matrix import DiagonalCovarianceMatrix, GeneralCovarianceMatrix


@pytest.fixture
def diagonal_covariance_matrix_example():
    size = 5
    variances = np.arange(1, size + 1)
    return DiagonalCovarianceMatrix(variances)


@pytest.fixture
def general_covariance_matrix_example():
    size = 5
    variances = np.arange(1, size + 1)
    covariance_matrix = np.diag(variances)
    return GeneralCovarianceMatrix(covariance_matrix)


@pytest.mark.parametrize('arg', ['diagonal_covariance_matrix_example',
                                 'general_covariance_matrix_example'])
def test_apply_inverse_to_vector(arg, request):
    covariance_matrix_example = request.getfixturevalue(arg)
    size = covariance_matrix_example.shape[0]
    vector = np.arange(1, size + 1)
    correct_answer = 1.
    assert np.allclose(covariance_matrix_example.apply_inverse(vector),
                        correct_answer)


@pytest.mark.parametrize('arg', ['diagonal_covariance_matrix_example',
                                 'general_covariance_matrix_example'])
def test_apply_inverse_to_matrix(arg, request):
    covariance_matrix_example = request.getfixturevalue(arg)
    size = covariance_matrix_example.shape[0]
    matrix = np.ones([size, size + 1]) * np.arange(1, size + 1)[:, None]
    correct_answer = 1.
    assert np.allclose(covariance_matrix_example.apply_inverse(matrix),
                        correct_answer)


@pytest.mark.parametrize('arg', ['diagonal_covariance_matrix_example',
                                 'general_covariance_matrix_example'])
def test_get_inverse(arg, request):
    covariance_matrix_example = request.getfixturevalue(arg)
    size = covariance_matrix_example.shape[0]
    precisions = 1. / np.arange(1, size + 1)
    correct_answer = np.diag(precisions)
    assert np.allclose(covariance_matrix_example.get_inverse(),
                       correct_answer)
    assert np.allclose(covariance_matrix_example._inverse,
                       correct_answer)


@pytest.mark.parametrize('arg', ['diagonal_covariance_matrix_example',
                                 'general_covariance_matrix_example'])
def test_add_inverse(arg, request):
    covariance_matrix_example = request.getfixturevalue(arg)
    size = covariance_matrix_example.shape[0]
    precisions = 1. / np.arange(1, size + 1)
    correct_answer = np.diag(precisions)
    assert np.allclose(covariance_matrix_example.get_inverse(),
                       correct_answer)


@pytest.mark.parametrize('arg', ['diagonal_covariance_matrix_example',
                                 'general_covariance_matrix_example'])
def test_get_logdet(arg, request):
    covariance_matrix_example = request.getfixturevalue(arg)
    size = covariance_matrix_example.shape[0]
    variances = np.arange(1, size + 1)
    correct_answer = np.sum(np.log(variances))
    assert np.allclose(covariance_matrix_example.get_logdet(),
                       correct_answer)
    assert np.allclose(covariance_matrix_example._logdet,
                       correct_answer)
