import numpy as np


def square(x: np.ndarray) -> np.ndarray:
    return np.power(x, 2)


def plus_two_square(x: np.ndarray) -> np.ndarray:
    return np.power(np.add(x, 2), 2)


def deriv(func, input_, delta=0.001):
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def multiple_inputs_add(x, y, sigma):
    a = x + y
    return sigma(a)


def multiple_inputs_multiply_backward(x, y, sigma):
    a = x * y
    dsda = deriv(sigma, a)
    dadx, dady = y, x
    return dsda * dadx, dsda * dady


def scalar_product(x, y):
    return np.dot(x, y)


def matmul_backward_first(x, w):
    return np.transpose(w, (1, 0))


def matrix_forward_extra(x, w, sigma):
    # Krok w przód dla funkcji obejmującej mnożenie macieży + dodatkową funkcję
    n = np.dot(x, w)
    return sigma(n)


def matrix_function_backward_x(x, w, sigma):
    # Obliczanie pochodnej funkcji względem pierwszego elementu czyli "X"
    n = np.dot(x, w)
    dsdn = deriv(sigma, n)
    dndx = np.transpose(w, (1, 0))
    return np.dot(dsdn, dndx)


def matrix_function_backward_sum_x(x, w, sigma):
    # Obliczanie pochodnej funkcji mnożenia macieży z sumowaniem, względem pierwszej macierzy x
    n = np.dot(x, w)
    s = sigma(n)
    # l = np.sum(s)

    dLdS = np.ones_like(
        s
    )  # Oznaczenie dLdS oznacza, że liczymy pochodną dla wyjścia L i wejścia S (patrz diagram)
    dSdN = deriv(sigma, n)
    dLdN = dLdS * dSdN

    dNdx = np.transpose(w, (1, 0))
    return np.dot(dLdN, dNdx)


if __name__ == "__main__":
    XX = np.array([1, 2, 3, 4, 5])
    WW = np.array([2, 2, 3, 4, 4])
    print(deriv(square, XX))
    print(deriv(plus_two_square, XX))
    print(multiple_inputs_add(XX, WW, square))
    print(multiple_inputs_multiply_backward(XX, WW, square))

    # Iloczyn skalarny
    WW = np.array([2, 2, 3, 4, 4]).reshape(-1, 1)
    print(matmul_backward_first(XX, WW))

    print(matrix_forward_extra(XX, WW, square))

    print(matrix_function_backward_x(XX, WW, square))

    # TEST
    print(f"N = {square(scalar_product(XX, WW))}")
    XX = np.array([1, 2, 3, 4, 5 + 1])  # Last element 408
    print(
        f"After + 1 change to x6 = {square(scalar_product(XX, WW))}"
    )  # Should be + 408
    # W skrócie -> gradient X względem N określa o ile zmieni się wartość funkcji po zmianie parametru xi.
    # Dla X = [x1, x2, x3] i gradient = [g1, g2, g3] -> wzrost x1 o 0.5 będzie skutkować zmianą wartośći o 0.5 * g1
    #   tzn  N(x1 + 0.5) = N(original) + 0.5 * g1

    # Macierze wielowymiarowe:
    XX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    WW = np.array([[1, 2], [2, 3], [3, 3]])
    print(matrix_function_backward_sum_x(XX, WW, lambda n: n))
