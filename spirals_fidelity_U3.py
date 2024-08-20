import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane.numpy import pi


# par = """
# 2.326129,-0.510986,1.324918,0.247266,2.773757;-1.950482,0.392850,-0.741124,-0.304963,2.107857;1.828872,-0.735590,-1.191067,0.348093,0.057093;0.886712,-1.489898,0.986148,-0.829879,0.811990;1.670201,-0.729539,0.023443,-0.483618,0.115849;2.951782,-1.530138,-0.058613,-0.405944,-0.014673;1.843292,-2.457700,-0.272808,-0.006721,-0.831912;0.795503,-2.047194,-0.412478,-0.818925,0.780022;-0.466999,-1.358838,1.511162,0.811780,0.813280;-1.784794,-0.724001,-0.000549,-0.100984,1.236930;0.322903,0.134435,0.313488,-0.258160,0.068965;-0.201617,-0.762900,-0.025386,0.064825,0.621889;0.753745,-1.568190,0.165242,-0.303520,1.673183;-1.079880,1.394898,1.820436,-0.510282,0.853383;1.007509,-0.183778,1.165979,0.374044,0.079085
# """

par = """
1.873457,-1.271325,1.574982,0.581540,3.125468;-1.740409,0.464150,-0.787948,-0.407372,2.233190;2.337631,-1.089066,-1.437560,0.668919,-0.025033;0.720979,-1.240496,1.026050,-0.771492,0.580650;1.675318,-1.306447,-0.511799,-0.985686,0.892292;2.227948,-2.079982,0.098623,-0.080788,-0.001149;1.946551,-2.667200,0.109069,0.489111,-0.406843;1.075479,-2.032830,0.239851,-1.051282,0.508573;0.480355,-1.058879,1.691445,1.994997,0.672091;-2.296355,-0.572643,0.015525,0.088538,1.678404;0.423832,0.562777,0.344113,-0.514148,0.085048;-0.226999,-0.372636,0.220690,-0.287838,0.556229;1.199385,-1.548740,0.489988,-0.345803,1.511788;-1.711430,0.500284,1.993603,-0.283929,1.016596;1.928735,-0.470987,0.900611,0.374044,-0.207826
"""

PARAMS = np.array([[float(p) for p in pp.split(",")] for pp in par.split(";")])


def U3(theta, phi, delta):
    """U3 gate"""
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * delta) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + delta)) * np.cos(theta / 2),
            ],
        ]
    )


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


def circuit(param, x, tgt_st):
    """This is what a quantum computer does.
    But since in this example we deal only with a single qubit, it becomes
    trivial to simulate this circuit on a classical computer.
    """
    # Ground state |0>
    state = np.zeros(2)
    state[0] = 1

    for p in param:
        state = U3(p[0] * x[0], p[1] * x[1], 0) @ state
        state = U3(p[2], p[3], p[4]) @ state

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


def model(params, x, st):
    fidel_function = lambda y: circuit(params, x, y)
    return fidel_function(st)


def main():
    np.random.seed(42)
    _ = two_spirals(400)
    X, y = two_spirals(20)

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
            f"figures/spirals-fidelity-U3-{i}.png",
            dpi=150,
            format="png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    main()
