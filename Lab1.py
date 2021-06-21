import numpy as np
import matplotlib.pyplot as plt


def Y(x):
    return np.log(x)


def F(yi):
    if yi <= 0:
        return 0
    if yi <= np.log(5):
        return (np.exp(yi)-1)/4
    if yi > np.log(5):
        return 1


def F_empirical(n):
    return [0] + [(i/n) for i in range(n+1)] + [1]


def main():
    a = 1
    b = 5
    print("Input n:")
    n = int(input())
    x = [a] + [((b-a)*np.random.rand()+a) for i in range(n)]
    y = [Y(a)-1] + [Y(x[i]) for i in range(len(x))] + [Y(b)+1]
    y = sorted(y)
    print("   Yi   |  F*(Yi)")
    f_emp = F_empirical(n)
    for i in range(2, len(y)-1):
        print(" {:.4f}  |  {:.4f}".format(y[i], f_emp[i]))

    x_plot_theor = np.linspace(y[0], y[-1], 10**5)
    y_plot_theor = [F(x) for x in x_plot_theor]
    plt.plot(x_plot_theor, y_plot_theor, color='red', label="Теоретическая функция")
    plt.step(y, f_emp, where='post', color='blue', label="Эмпирическая функция")
    plt.grid(True)
    plt.legend()
    plt.show()


if "__name__" == main():
    main()
