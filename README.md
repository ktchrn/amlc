[![Build Status](https://travis-ci.org/ktchrn/amlc.svg?branch=master)](https://travis-ci.org/ktchrn/amlc)

`amlc` is a tool for analytically marginalizing over linear continuum parameters in models of absorption line spectra.


Installation instructions
----
`amlc` can be installed with `pip`:

```
pip install amlc --user
```

Dependencies
-----
* Python 3.5+
* `numpy`
* `scipy`


Basic usage
-----------
`amlc` is designed for evaluating the likelihood of a spectrum given the spectrum's uncertainties, a design matrix for the continuum model, and a proposed set of absorption features:
```
from amlc.marginalized_likelihood import MarginalizedLikelihood

marginalized_likelihood_instance = MarginalizedLikelihood(observed_spectrum, variance_of_observed_spectrum, continuum_design_matrix)
marginalized_likelihood_instance(proposed_transmittance_model, return_logp=True)['logp']
```

License
-----
This project is Copyright (c) Kirill Tchernyshyov and is licensed under the MIT
license (see the LICENSE file).
