"""
Contains a copy of the spline fitting code from scipy (https://github.com/scipy/scipy/blob/v1.13.0/scipy/interpolate/_bsplines.py#L1941-L2126) to also return the fitting parameter lam
"""

import numpy as np
from scipy.linalg import (LinAlgError, cholesky_banded, solve_banded)
from scipy.optimize import minimize_scalar
from scipy.interpolate import BSpline

def _compute_optimal_gcv_parameter(X, wE, y, w):
    """
    Returns an optimal regularization parameter from the GCV criteria [1].

    Parameters
    ----------
    X : array, shape (5, n)
        5 bands of the design matrix ``X`` stored in LAPACK banded storage.
    wE : array, shape (5, n)
        5 bands of the penalty matrix :math:`W^{-1} E` stored in LAPACK banded
        storage.
    y : array, shape (n,)
        Ordinates.
    w : array, shape (n,)
        Vector of weights.

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.

    Notes
    -----
    No checks are performed.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
        for observational data, Philadelphia, Pennsylvania: Society for
        Industrial and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`

    """

    def compute_banded_symmetric_XT_W_Y(X, w, Y):
        """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
        w : array, shape (n,)
            Array of weights
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.

        """
        # compute W Y
        W_Y = np.copy(Y)

        W_Y[2] *= w
        for i in range(2):
            W_Y[i, 2 - i:] *= w[:-2 + i]
            W_Y[3 + i, :-1 - i] *= w[1 + i:]

        n = X.shape[1]
        res = np.zeros((4, n))
        for i in range(n):
            for j in range(min(n-i, 4)):
                res[-j-1, i + j] = sum(X[j:, i] * W_Y[:5-j, i + j])
        return res

    def compute_b_inv(A):
        """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

        def find_b_inv_elem(i, j, U, D, B):
            rng = min(3, n - i - 1)
            rng_sum = 0.
            if j == 0:
                # use 2-nd formula from [1]
                for k in range(1, rng + 1):
                    rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
                rng_sum += D[i]
                B[-1, i] = rng_sum
            else:
                # use 1-st formula from [1]
                for k in range(1, rng + 1):
                    diag = abs(k - j)
                    ind = i + min(k, j)
                    rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
                B[-j - 1, i + j] = rng_sum

        U = cholesky_banded(A)
        for i in range(2, 5):
            U[-i, i-1:] /= U[-1, :-i+1]
        D = 1. / (U[-1])**2
        U[-1] /= U[-1]

        n = U.shape[1]

        B = np.zeros(shape=(4, n))
        for i in range(n - 1, -1, -1):
            for j in range(min(3, n - i - 1), -1, -1):
                find_b_inv_elem(i, j, U, D, B)
        # the first row contains garbage and should be removed
        B[0] = [0.] * n
        return B

    def _gcv(lam, X, XtWX, wE, XtE):
        r"""
        Computes the generalized cross-validation criteria [1].

        Parameters
        ----------
        lam : float, (:math:`\lambda \geq 0`)
            Regularization parameter.
        X : array, shape (5, n)
            Matrix is stored in LAPACK banded storage.
        XtWX : array, shape (4, n)
            Product :math:`X^T W X` stored in LAPACK banded storage.
        wE : array, shape (5, n)
            Matrix :math:`W^{-1} E` stored in LAPACK banded storage.
        XtE : array, shape (4, n)
            Product :math:`X^T E` stored in LAPACK banded storage.

        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            :math:`\lambda`.

        Notes
        -----
        Criteria is computed from the formula (1.3.2) [3]:

        .. math:

        GCV(\lambda) = \dfrac{1}{n} \sum\limits_{k = 1}^{n} \dfrac{ \left(
        y_k - f_{\lambda}(x_k) \right)^2}{\left( 1 - \Tr{A}/n\right)^2}$.
        The criteria is discussed in section 1.3 [3].

        The numerator is computed using (2.2.4) [3] and the denominator is
        computed using an algorithm from [2] (see in the ``compute_b_inv``
        function).

        References
        ----------
        .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
            for observational data, Philadelphia, Pennsylvania: Society for
            Industrial and Applied Mathematics, 1990, pp. 45-65.
            :doi:`10.1137/1.9781611970128`
        .. [2] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`
        .. [3] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
            BSc thesis, 2022. Might be available (in Russian)
            `here <https://www.hse.ru/ba/am/students/diplomas/620910604>`_

        """
        # Compute the numerator from (2.2.4) [3]
        n = X.shape[1]
        c = solve_banded((2, 2), X + lam * wE, y)
        res = np.zeros(n)
        # compute ``W^{-1} E c`` with respect to banded-storage of ``E``
        tmp = wE * c
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += tmp[j, i + 2 - j]
        numer = np.linalg.norm(lam * res)**2 / n

        # compute the denominator
        lhs = XtWX + lam * XtE
        try:
            b_banded = compute_b_inv(lhs)
            # compute the trace of the product b_banded @ XtX
            tr = b_banded * XtWX
            tr[:-1] *= 2
            # find the denominator
            denom = (1 - sum(sum(tr)) / n)**2
        except LinAlgError:
            # cholesky decomposition cannot be performed
            raise ValueError('Seems like the problem is ill-posed')

        res = numer / denom

        return res

    n = X.shape[1]

    XtWX = compute_banded_symmetric_XT_W_Y(X, w, X)
    XtE = compute_banded_symmetric_XT_W_Y(X, w, wE)

    def fun(lam):
        return _gcv(lam, X, XtWX, wE, XtE)

    gcv_est = minimize_scalar(fun, bounds=(0, n), method='Bounded')
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")


def _coeff_of_divided_diff(x):
    """
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array, shape (n,)
        Array which is used for the computation of divided difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.

    Notes
    -----
    Vector ``x`` should have unique elements, otherwise an error division by
    zero might be raised.

    No checks are performed.

    """
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        pp = 1.
        for k in range(n):
            if k != i:
                pp *= (x[i] - x[k])
        res[i] = 1. / pp
    return res


def make_smoothing_spline(x, y, w=None, lam=None):
    r"""
    Compute the (coefficients of) smoothing cubic spline function using
    ``lam`` to control the tradeoff between the amount of smoothness of the
    curve and its proximity to the data. In case ``lam`` is None, using the
    GCV criteria [1] to find it.

    A smoothing spline is found as a solution to the regularized weighted
    linear regression problem:

    .. math::

        \sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2 +
        \lambda\int\limits_{x_1}^{x_n} (f^{(2)}(u))^2 d u

    where :math:`f` is a spline function, :math:`w` is a vector of weights and
    :math:`\lambda` is a regularization parameter.

    If ``lam`` is None, we use the GCV criteria to find an optimal
    regularization parameter, otherwise we solve the regularized weighted
    linear regression problem with given parameter. The parameter controls
    the tradeoff in the following way: the larger the parameter becomes, the
    smoother the function gets.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. `n` must be at least 5.
    y : array_like, shape (n,)
        Ordinates. `n` must be at least 5.
    w : array_like, shape (n,), optional
        Vector of weights. Default is ``np.ones_like(x)``.
    lam : float, (:math:`\lambda \geq 0`), optional
        Regularization parameter. If ``lam`` is None, then it is found from
        the GCV criteria. Default is None.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis
        as a solution of the problem of smoothing splines using
        the GCV criteria [1] in case ``lam`` is None, otherwise using the
        given parameter ``lam``.
    lam : float
        The value of lam used for the fit

    Notes
    -----
    This algorithm is a clean room reimplementation of the algorithm
    introduced by Woltring in FORTRAN [2]. The original version cannot be used
    in SciPy source code because of the license issues. The details of the
    reimplementation are discussed here (available only in Russian) [4].

    If the vector of weights ``w`` is None, we assume that all the points are
    equal in terms of weights, and vector of weights is vector of ones.

    Note that in weighted residual sum of squares, weights are not squared:
    :math:`\sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2` while in
    ``splrep`` the sum is built from the squared weights.

    In cases when the initial problem is ill-posed (for example, the product
    :math:`X^T W X` where :math:`X` is a design matrix is not a positive
    defined matrix) a ValueError is raised.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models for
        observational data, Philadelphia, Pennsylvania: Society for Industrial
        and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`
    .. [2] H. J. Woltring, A Fortran package for generalized, cross-validatory
        spline smoothing and differentiation, Advances in Engineering
        Software, vol. 8, no. 2, pp. 104-113, 1986.
        :doi:`10.1016/0141-1195(86)90098-7`
    .. [3] T. Hastie, J. Friedman, and R. Tisbshirani, "Smoothing Splines" in
        The elements of Statistical Learning: Data Mining, Inference, and
        prediction, New York: Springer, 2017, pp. 241-249.
        :doi:`10.1007/978-0-387-84858-7`
    .. [4] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
        BSc thesis, 2022.
        `<https://www.hse.ru/ba/am/students/diplomas/620910604>`_ (in
        Russian)

    Examples
    --------
    Generate some noisy data

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> n = 200
    >>> def func(x):
    ...    return x**3 + x**2 * np.sin(4 * x)
    >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
    >>> y = func(x) + np.random.normal(scale=1.5, size=n)

    Make a smoothing spline function

    >>> from scipy.interpolate import make_smoothing_spline
    >>> spl = make_smoothing_spline(x, y)

    Plot both

    >>> import matplotlib.pyplot as plt
    >>> grid = np.linspace(x[0], x[-1], 400)
    >>> plt.plot(grid, spl(grid), label='Spline')
    >>> plt.plot(grid, func(grid), label='Original function')
    >>> plt.scatter(x, y, marker='.')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """

    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError('``x`` and ``y`` should be one dimensional and the'
                         ' same size')

    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]

    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')

    # It is known that the solution to the stated minimization problem exists
    # and is a natural cubic spline with vector of knots equal to the unique
    # elements of ``x`` [3], so we will solve the problem in the basis of
    # natural splines.

    # create design matrix in the B-spline basis
    X_bspl = BSpline.design_matrix(x, t, 3)
    # move from B-spline basis to the basis of natural splines using equations
    # (2.1.7) [4]
    # central elements
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = X_bspl[i: i - 4, 3: -3][np.diag_indices(n - 4)]

    # first elements
    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0],
                X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])

    # last elements
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2],
                 (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]

    # create penalty matrix and divide it by vector of weights: W^{-1} E
    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3])\
                   / w[j-2: j+3]

    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6

    if lam is None:
        lam = _compute_optimal_gcv_parameter(X, wE, y, w)
    elif lam < 0.:
        raise ValueError('Regularization parameter should be non-negative')

    # solve the initial problem in the basis of natural splines
    c = solve_banded((2, 2), X + lam * wE, y)
    # move back to B-spline basis using equations (2.2.10) [4]
    c_ = np.r_[c[0] * (t[5] + t[4] - 2 * t[3]) + c[1],
               c[0] * (t[5] - t[3]) + c[1],
               c[1: -1],
               c[-1] * (t[-4] - t[-6]) + c[-2],
               c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]

    return BSpline.construct_fast(t, c_, 3), lam