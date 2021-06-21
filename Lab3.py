import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def Y(x):
    return np.log(x)


def F_empirical(b, n, m):
    f = list()
    count = 0
    for i in range(b):
        count += m[i]
        f.append(count/n)
    return f


def F(yi):
    if yi <= 0:
        return 0
    if yi <= np.log(5):
        return (np.exp(yi)-1)/4
    if yi > np.log(5):
        return 1


def f_theor(yi):
    if yi <= 0 or yi >= np.log(5):
        return 0
    else:
        return np.exp(yi)/4


def M(n):
    if n <= 100:
        return int(n**(1/2))
    else:
        return int(np.random.uniform(2, 4) * np.log(n))


def equiprobable_method(n, y):
    m_count = M(n)
    m = n//m_count
    A = list()
    B = list()
    h = [0] * m_count
    f = [0] * m_count
    A.append(y[0])
    for i in range(1, m_count):
        A.append((y[m*i] + y[m * i + 1]) / 2)
    for i in range(m_count - 1):
        B.append(A[i + 1])
    B.append(y[-1])

    for i in range(m_count):
        h[i] = B[i] - A[i]
        f[i] = m / (n * h[i])
    return (A, B, f, m_count)


def chi_squared_method(a, b, chi_squared):
    chi_x = [((b - a) * np.random.rand() + a) for i in range(chi_squared)]
    chi_y = [Y(x) for x in chi_x]
    y = sorted(chi_y)
    A, B, f, m = equiprobable_method(chi_squared, y)
    if m > chi_squared:
        m = chi_squared
    k = m - 1
    chi = 0
    for i in range(m):
        true_value = F(B[i]) - F(A[i])
        pi_value = 1 / m
        chi += chi_squared * (true_value - pi_value) ** 2 / true_value

    print("Chi-squared = {:.6f}".format(chi))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.90, chi, stats.chi2.ppf(0.90, k), (chi < stats.chi2.ppf(0.90, k))))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.95, chi, stats.chi2.ppf(0.95, k), (chi < stats.chi2.ppf(0.95, k))))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.99, chi, stats.chi2.ppf(0.99, k), (chi < stats.chi2.ppf(0.99, k))))


def kolmogorov_method(a, b, n):
    x = [((b - a) * np.random.rand() + a) for i in range(n)]
    y = [Y(i) for i in x]
    y = sorted(y)

    d = 0
    for i in range(n):
        d = max(d, max(abs((i+1)/n - F(y[i])), abs(F(y[i] - i/n))))
    lambd = (6 * n * d + 1)/(6 * (n * 0.5))

    print("Kolmogorov_lambda = {:.6f}".format(lambd))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.90, lambd, stats.kstwobign.ppf(0.90),
                                                 (lambd < stats.kstwobign.ppf(0.90))))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.95, lambd, stats.kstwobign.ppf(0.95),
                                                 (lambd < stats.kstwobign.ppf(0.95))))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.99, lambd, stats.kstwobign.ppf(0.99),
                                                 (lambd < stats.kstwobign.ppf(0.95))))


def mises_method(a, b, n):
    x = [((b - a) * np.random.rand() + a) for i in range(n)]
    y = [Y(i) for i in x]
    y = sorted(y)

    mises = 1 / (12 * n)
    for i in range(n):
        mises += (F(y[i]) - (i + 0.5) / n)**2
    print("Mises = {:.6f}".format(mises))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.90, mises, 0.347,
                                                 (mises < 0.347)))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.95, mises, 0.461,
                                                 (mises < 0.461)))
    print("{:.2f}: {:.3f} < {:.3f} => {}".format(0.99, mises, 0.744,
                                                 (mises < 0.744)))


def main():
    a = 1
    b = 5

    # хи-квадрат
    print("Chi-squared method:")
    chi_squared = 200
    chi_squared_method(a, b, chi_squared)
    print()

    # Колмогорова
    print("Kolmogorov method:")
    kolmogorov = 30
    kolmogorov_method(a, b, kolmogorov)
    print()

    # Мизеса
    print("Mises method:")
    mises = 50
    mises_method(a, b, mises)


if "__name__" == main():
    main()
