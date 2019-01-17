import pytest
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import check_grad
from ..marginalized_likelihood import MarginalizedLikelihood

@pytest.fixture
def improper_marginalized_likelihood_example():
    np.random.seed(seed=0)
    size = 150
    continuum_deg = 1
    wave_ax = np.linspace(-1, 1, size)
    transmittance = 1 - np.abs(np.random.normal(0, 0.2, size))
    A_m = np.polynomial.polynomial.polyvander(wave_ax, continuum_deg)
    c_true = np.asarray([1., 0.2])
    y_sd = np.full(size, 0.2)
    y_true = transmittance * (A_m @ c_true)
    y_obs = np.random.normal(y_true, y_sd)

    return MarginalizedLikelihood(y_obs, y_sd**2, A_m)


@pytest.fixture
def proper_marginalized_likelihood_example():
    np.random.seed(seed=0)
    size = 150
    continuum_deg = 1
    wave_ax = np.linspace(-1, 1, size)
    transmittance = 1 - np.abs(np.random.normal(0, 0.2, size))
    A_m = np.polynomial.polynomial.polyvander(wave_ax, continuum_deg)
    c_true = np.asarray([1., 0.2])
    y_sd = np.full(size, 0.2)
    y_true = transmittance * (A_m @ c_true)
    y_obs = np.random.normal(y_true, y_sd)
    c_sd = np.asarray([10., 10.])

    return MarginalizedLikelihood(y_obs, y_sd**2, A_m, c_cov=c_sd**2)


@pytest.mark.parametrize('arg', ['improper_marginalized_likelihood_example',
                                 'proper_marginalized_likelihood_example'])
def test_marginalized_likelihood(arg, request):
    marginalized_likelihood_example = request.getfixturevalue(arg)
    np.random.seed(seed=0)
    size = marginalized_likelihood_example.y.size
    transmittance = 1 - np.abs(np.random.normal(0, 0.2, size))
    def differential_func(c1, c0):
        c = np.asarray([c0, c1])
        return np.exp(marginalized_likelihood_example.get_unmarginalized_likelihood(c, transmittance,
                                                                                    mu_m=1))
    gfun = lambda *args : -np.inf
    hfun = lambda *args : np.inf
    numerical_p, err = dblquad(differential_func, -np.inf, np.inf, gfun, hfun)
    analytic_p = np.exp(marginalized_likelihood_example(transmittance, return_logp=True,
                                                        mu_m=1)['logp'])
    print(numerical_p - analytic_p, err)
    assert np.abs(numerical_p - analytic_p) < err


@pytest.mark.parametrize('arg', ['improper_marginalized_likelihood_example',
                                 'proper_marginalized_likelihood_example'])
def test_grad_logp_wrt_d_theta(arg, request):
    marginalized_likelihood_example = request.getfixturevalue(arg)
    np.random.seed(seed=0)
    size = marginalized_likelihood_example.y.size
    x0 = 1 - np.abs(np.random.normal(0, 0.2, size))
    jac_d_theta = np.identity(size)
    def func(x):
        return marginalized_likelihood_example(x, mu_m=1, return_logp=True)['logp']

    def grad(x):
        return marginalized_likelihood_example(x, mu_m=1, return_grad_logp=True,
                                               jac_d_theta=jac_d_theta,
                                               jac_mu_m=0,
                                               jac_mu_b=0)['grad_logp']
    assert check_grad(func, grad, x0) < 1.e-2


@pytest.mark.parametrize('arg', ['improper_marginalized_likelihood_example',
                                 'proper_marginalized_likelihood_example'])
def test_grad_logp_wrt_d_mu_m(arg, request):
    marginalized_likelihood_example = request.getfixturevalue(arg)
    np.random.seed(seed=0)
    size = marginalized_likelihood_example.y.size
    transmittance = 1 - np.abs(np.random.normal(0, 0.2, size))
    x0 = np.ones(size)
    jac_mu_m = np.identity(size)
    def func(x):
        return marginalized_likelihood_example(transmittance, mu_m=x, return_logp=True)['logp']

    def grad(x):
        return marginalized_likelihood_example(transmittance, mu_m=x, return_grad_logp=True,
                                               jac_d_theta=0,
                                               jac_mu_m=jac_mu_m,
                                               jac_mu_b=0)['grad_logp']
    assert check_grad(func, grad, x0) < 1.e-2


@pytest.mark.parametrize('arg', ['improper_marginalized_likelihood_example',
                                 'proper_marginalized_likelihood_example'])
def test_grad_logp_wrt_d_mu_b(arg, request):
    marginalized_likelihood_example = request.getfixturevalue(arg)
    np.random.seed(seed=0)
    size = marginalized_likelihood_example.y.size
    transmittance = 1 - np.abs(np.random.normal(0, 0.2, size))
    x0 = np.zeros(size)
    jac_mu_b = np.identity(size)
    def func(x):
        return marginalized_likelihood_example(transmittance, mu_m=1, mu_b=x, return_logp=True)['logp']

    def grad(x):
        return marginalized_likelihood_example(transmittance, mu_m=1, mu_b=x, return_grad_logp=True,
                                               jac_d_theta=0,
                                               jac_mu_m=0,
                                               jac_mu_b=jac_mu_b)['grad_logp']
    assert check_grad(func, grad, x0) < 1.e-2
