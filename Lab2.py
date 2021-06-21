import numpy as np
import matplotlib.pyplot as plt


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
        return int(np.random.randint(2, 4)*np.log(n))


def equal_interval_method(n, y):
    m_count = M(n)
    h = (y[n-1]-y[0])/m_count
    A = list()
    B = list()
    for i in range(m_count):
        A.append(y[0] + i*h)
    for i in range(m_count-1):
        B.append(A[i+1])
    B.append(y[-1])
    m = [0]*m_count
    f = [0]*m_count
    for i in range(m_count):
        for j in y:
            if B[i] == j:
                if i+1 < m_count:
                    m[i] += 0.5
                    m[i + 1] += 0.5
                else:
                    m[i] += 0.5
            elif A[i] < j < B[i]:
                m[i] += 1
        f[i] = m[i] / (n * h)
    return (A, B, f, m)


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
    return (A, B, f, m)


def main():
    a = 1
    b = 5
    print("Input n:")
    n = int(input())
    x = [((b-a)*np.random.rand()+a) for i in range(n)]
    y = [Y(x[i]) for i in range(len(x))]
    y = sorted(y)

    # Равноитервальный метод
    A_eq, B_eq, f_eq, m_eq = equal_interval_method(n, y)
    print("Равноитервальный метод:")
    print("   A     |   B     |  f_equal_interval")
    for i in range(len(f_eq)):
        print(" {:.4f}  |  {:.4f}  |   {:.4f}".format(A_eq[i], B_eq[i], f_eq[i]))
    plt.subplot(152)
    plt.suptitle("Равноинтервальный метод")
    plt.step(B_eq, f_eq, color='blue')

    # polygon
    x_eq_polygon = list()
    y_eq_polygon = list()
    for i in range(len(B_eq)):
        x_eq_polygon.append((B_eq[i]+A_eq[i])/2)
        y_eq_polygon.append(m_eq[i]/n)

    plt.subplot(153)
    eq_polygon = plt.plot(x_eq_polygon, y_eq_polygon, color='red')

    # empir func
    f_emp_eq = F_empirical(len(B_eq), n, m_eq)
    plt.subplot(154)
    plt.plot(B_eq, f_emp_eq, color='black')
    plt.show()

    # Равновероятностный метод
    A_prob, B_prob, f_prob, m_prob = equiprobable_method(n, y)
    m_prob = [int(m_prob) for i in range(len(B_prob))]
    print("Равновероятностный метод:")
    print("   A     |   B     |  f_equiprobable")
    for i in range(len(f_prob)):
        print(" {:.4f}  |  {:.4f}  |   {:.4f}".format(A_prob[i], B_prob[i], f_prob[i]))

    plt.subplot(152)
    plt.suptitle("Равновероятностный метод")
    plt.step(B_prob, f_prob, color='blue')

    # polygon
    x_prob_polygon = list()
    y_prob_polygon = list()
    for i in range(len(B_prob)):
        x_prob_polygon.append((B_prob[i] + A_prob[i]) / 2)
        y_prob_polygon.append(m_prob[i] / n)

    plt.subplot(153)
    plt.plot(x_prob_polygon, y_prob_polygon, color='red')

    # empir func
    f_emp_prob = F_empirical(len(B_prob), n, m_prob)
    plt.subplot(154)
    plt.plot(B_prob, f_emp_prob, color='black')
    plt.show()

    # функция распределения
    plt.plot([Y(a), *B_eq, Y(b)], [0, *f_emp_eq, 1], label="Равноинтервальный метод", color='blue')
    plt.plot([Y(a), *B_prob, Y(b)], [0, *f_emp_prob, 1], label="Равновероятностный метод", color='red')
    F_x_plot_theor = np.linspace(Y(a) - 0.2, Y(b) + 0.2, 10 ** 5)
    F_y_plot_theor = [F(x) for x in F_x_plot_theor]
    plt.plot(F_x_plot_theor, F_y_plot_theor, color='black', label="Теоретическая плотность")
    plt.grid = True
    plt.legend()
    plt.show()

    # Сравнение теоритичекой плотности и имперической
    plt.plot([Y(a), *B_eq, Y(b)], [0, *f_eq, 0], label="Равноинтервальный метод", color='blue')
    plt.plot([Y(a), *B_prob, Y(b)], [0, *f_prob, 0], label="Равновероятностный метод", color='red')
    x_plot_theor = np.linspace(Y(a)-0.2, Y(b)+0.2, 10 ** 5)
    y_plot_theor = [f_theor(x) for x in x_plot_theor]
    plt.plot(x_plot_theor, y_plot_theor, color='black', label="Теоретическая плотность")
    plt.grid = True
    plt.legend()
    plt.show()


if "__name__" == main():
    main()
