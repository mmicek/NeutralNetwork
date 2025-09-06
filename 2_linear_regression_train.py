import numpy as np
from numpy import ndarray
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def init_weights(n_in: int) -> dict[str, ndarray]:
    """
    Initialize weights on first forward pass of model.
    """

    weights: dict[str, ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)

    weights["W"] = W
    weights["B"] = B

    return weights


def permute_data(X: ndarray, y: ndarray):
    """
    Permute X and y, using the same permutation, along axis=0
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def generate_batch(
    X: ndarray, y: ndarray, start: int = 0, batch_size: int = 10
) -> tuple[ndarray, ndarray]:
    """
    Generate batch from X and y, given a start position
    """
    assert X.ndim == y.ndim == 2, "X and Y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start : start + batch_size], y[start : start + batch_size]

    return X_batch, y_batch


def forward_linear_regression(x, y, weights: dict[str, ndarray]):
    """
    :param x:   macierz (obserwacje x cechy)
    :param y:   odpowiedz
    """
    n = np.dot(x, weights["W"])
    p = n + weights["B"]  # predykcja

    loss = np.mean(np.power(y - p, 2))

    return loss, {"x": x, "N": n, "P": p, "Y": y}


def linear_regression_loss_gradient(
    forward_info: dict[str, ndarray], weights: dict[str, ndarray]
):
    dLdP = -2 * (forward_info["Y"] - forward_info["P"])
    dPdN = np.ones_like(forward_info["N"])
    dPdB = np.ones_like(weights["B"])
    dNdW = np.transpose(forward_info["x"], (1, 0))

    dLdN = dLdP * dPdN
    return {"W": np.dot(dNdW, dLdN), "B": (dLdP * dPdB).sum(axis=0)}


def train(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = 1000,
    learning_rate: float = 0.01,
    batch_size: int = 100,
    return_losses: bool = False,
    return_weights: bool = False,
    seed: int = 1,
) -> None:
    """
    Train model for a certain number of epochs.
    """
    if seed:
        np.random.seed(seed)
    start = 0

    # Initialize weights
    weights = init_weights(x.shape[1])

    # Permute data
    x, y = permute_data(x, y)

    if return_losses:
        losses = []

    for i in range(n_iter):

        # Generate batch
        if start >= x.shape[0]:
            x, y = permute_data(x, y)
            start = 0

        x_batch, y_batch = generate_batch(x, y, start, batch_size)
        start += batch_size

        # Train net using generated batch
        loss, forward_info = forward_linear_regression(x_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = linear_regression_loss_gradient(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights

    return None


if __name__ == "__main__":
    boston = load_boston()
    data = boston.data
    target = boston.target
    features = boston.feature_names

    s = StandardScaler()
    data = s.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=80718
    )
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    train_info = train(
        X_train,
        y_train,
        n_iter=1000,
        learning_rate=0.001,
        batch_size=23,
        return_losses=True,
        return_weights=True,
        seed=180708,
    )
    losses = train_info[0]
    weights = train_info[1]
