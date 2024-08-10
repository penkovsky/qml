import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane.numpy import pi


# From Ito color scheme
C1 = [
    [240 / 255.0, 228 / 255.0, 66 / 255.0],  # Yellow
    [0, 114 / 255.0, 178 / 255.0],
]  # Blue


def two_spirals(_n_points, noise=0.8):
    """
    Returns the two spirals dataset.
    """
    n_points = _n_points // 2
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 10.0
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return np.array(x, requires_grad=False), np.array(
        y, dtype="uint8", requires_grad=False
    )


def colorf(Y, c=C1):
    colors = []
    for y in Y:
        if y == 0:
            colors.append(c[0])
        else:
            colors.append(c[1])
    return np.array(colors)


def rot(phi, theta, omega):
    """Rotation around three axes"""
    return np.array(
        [
            [
                np.exp(-1j * (phi + omega) / 2) * np.cos(theta / 2),
                -np.exp(1j * (phi - omega) / 2) * np.sin(theta / 2),
            ],
            [
                np.exp(-1j * (phi - omega) / 2) * np.sin(theta / 2),
                np.exp(1j * (phi + omega) / 2) * np.cos(theta / 2),
            ],
        ]
    )


def circuit(param, x, tgt_st):
    """This is what a quantum computer does.
    But since in this example we deal only with a single qubit, it becomes
    trivial to simulate this circuit on a classical computer.
    """
    # Ground state |0>
    state = np.zeros(2)
    state[0] = 1

    for p in param:
        state = rot(p[0] * x[0], p[1] * x[1], 0) @ state
        state = rot(p[3], p[4], p[5]) @ state

    # Compute the fidelity of the state with the target state
    tgt_conj = np.conj(tgt_st)
    fidel = np.abs(np.dot(tgt_conj, state)) ** 2
    # The expectation value of the projector onto the target
    # state. That is, the probability of measuring the target state if we were
    # to measure our quantum state (measuring 0 when the target state is |0>
    # and, 1 when the target state is |1>).
    return fidel


def get_meshgrid(X, h=0.1):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # Generate a grid of points with distance h between them
    return np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


def plot_decision_boundary(model, X, Y=None):
    xx, yy = get_meshgrid(X)
    # Predict the function value for the whole grid
    inputs = np.c_[xx.ravel(), yy.ravel()]
    print("inputs", inputs.shape)

    Z = model(inputs)

    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    img = plt.contourf(xx, yy, Z, cmap=plt.cm.plasma)

    # Insert the colorbar for the contourf.
    # Take care about the colorbar size.
    plt.colorbar(img, fraction=0.046, pad=0.04)

    # plt.xlabel('x1', fontsize=18)
    # plt.ylabel('x2', fontsize=18)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])

    if Y is not None:
        Y1 = Y
    else:
        Y1 = model(X)

    visualize(X, Y1, title="")

    return img


def visualize(X, Y, title=None, ax=None, s=40, c=C1):
    colors = colorf(Y, c=c)

    if ax is None:
        plt.title(title)
        plt.scatter(X[:, 0], X[:, 1], c=colors, s=s)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=s)


PARAMS = np.array(
    [
        [0.29827245, 0.08220332, 0.3931218, 1.42578576, 0.50208184, 1.5928177],
        [0.05869113, 0.26620008, 0.35719233, 1.12377337, 0.46328539, 1.70133469],
        [1.18456692, -0.45356308, 0.50775784, 0.74022415, 0.21701071, -0.29728818],
        [-0.73229329, 0.90770549, 0.59536637, 0.36220917, 0.91383889, 0.04663861],
        [-0.54919759, 0.96878071, 0.64874711, 1.39345259, 1.35545392, 0.38011819],
        [0.2557679, -0.32282553, 0.32085454, 0.57328026, 1.60893581, 1.21153585],
        [1.14139141, 1.09135017, 0.05473271, 0.6102359, -0.55232082, -0.47481543],
        [1.36793678, 1.19127662, 0.84387941, 0.67343336, 0.34314725, -0.0578214],
        [1.47484026, 1.01214414, 0.92880796, 0.76562758, -0.2116039, 0.83520595],
        [0.47545624, -0.37081129, 0.62133062, 0.96669664, 0.05487279, 0.24774081],
        [1.48150396, -1.89516311, 0.05383052, -0.15641872, -0.09757299, 0.04472809],
        [1.76554667, -2.2032729, 0.44418341, 0.34249646, 0.05324532, 0.62406002],
        [1.05227216, -0.94414955, 0.77804631, 0.67808506, 0.91553152, 1.03300285],
        [0.60005659, 1.34494046, 0.24860488, -0.31970111, 0.28673467, 1.3734422],
        [1.57574681, 0.40458179, 0.34557959, -1.87592608, -0.88068899, 0.23475442],
    ],
    requires_grad=True,
)


def model(params, x, st):

    fidel_function = lambda y: circuit(params, x, y)
    return fidel_function(st)


def main():
    np.random.seed(42)
    X, y = two_spirals(200)

    states = [np.zeros(2) for i in range(2)]
    for i in range(2):
        states[i][i] = 1

    for i in range(2):

        plt.figure(figsize=(5, 5))
        plt.gca().set_aspect("equal")
        plot_decision_boundary(
            lambda x: np.array([model(PARAMS, _x, states[i]) for _x in x]), X, y
        )
        plt.savefig(
            f"spirals-fidelity-{i}.png",
            dpi=150,
            format="png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    main()
