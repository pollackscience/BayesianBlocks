from __future__ import division, print_function, absolute_import

import warnings

from scipy.special import comb
from scipy.misc.doccer import inherit_docstring_from
from scipy import special
from scipy import optimize
from scipy import integrate
from scipy.special import (gammaln as gamln, gamma as gam, boxcox, boxcox1p)

from numpy import (where, arange, putmask, ravel, sum, shape,
                   log, sqrt, exp, arctanh, tan, sin, arcsin, arctan,
                   tanh, cos, cosh, sinh)

from numpy import polyval, place, extract, any, asarray, nan, inf, pi

import numpy as np
import numpy.random as mtrand
from . import vonmises_cython
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
                                 tukeylambda_kurtosis as _tlkurt)

from ._distn_infrastructure import (
        rv_continuous, valarray, _skew, _kurtosis, _lazywhere,
        _ncx2_log_pdf, _ncx2_pdf, _ncx2_cdf, get_distribution_names,
        )

from ._constants import _XMIN, _EULER, _ZETA3

class pwl_gen(rv_continuous):
    """A power-function continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `powerlaw` is::
        powerlaw.pdf(x, a) = a * x**(a-1)
    for ``0 <= x <= 1``, ``a > 0``.
    `powerlaw` is a special case of `beta` with ``b == 1``.
    %(example)s
    """
    def _pdf(self, x, a):
        return a*x**(a-1.0)

    def _logpdf(self, x, a):
        return log(a) + special.xlogy(a - 1, x)

    def _cdf(self, x, a):
        return x**(a*1.0)

    def _logcdf(self, x, a):
        return a*log(x)

    def _ppf(self, q, a):
        return pow(q, 1.0/a)

    def _stats(self, a):
        return (a / (a + 1.0),
                a / (a + 2.0) / (a + 1.0) ** 2,
                -2.0 * ((a - 1.0) / (a + 3.0)) * sqrt((a + 2.0) / a),
                6 * polyval([1, -1, -6, 2], a) / (a * (a + 3.0) * (a + 4)))

    def _entropy(self, a):
        return 1 - 1.0/a - log(a)
powerlaw = powerlaw_gen(a=0.0, b=1.0, name="powerlaw")

# Collect names of classes and objects in this module.
pairs = list(globals().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_continuous)

__all__ = _distn_names + _distn_gen_names

