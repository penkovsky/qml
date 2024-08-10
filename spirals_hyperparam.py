"""
Inspired by https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier/
"""

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane.numpy import pi
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pandas as pd


def plot_data(x, y, fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="tab:orange", s=20)
    ax.scatter(x[blues, 0], x[blues, 1], c="tab:blue", s=20)
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    for a in ["x", "y"]:
        plt.tick_params(axis=a, labelsize=15)


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
        state = rot(p[0] * x[0], p[1] * x[1], p[2] * x[2]) @ state
        state = rot(p[3], p[4], p[5]) @ state

    # Compute the fidelity of the state with the target state
    tgt_conj = np.conj(tgt_st)
    fidel = np.abs(np.dot(tgt_conj, state)) ** 2
    # The expectation value of the projector onto the target
    # state. That is, the probability of measuring the target state if we were
    # to measure our quantum state (measuring 0 when the target state is |0>
    # and, 1 when the target state is |1>).
    return fidel


def cost(param, x, y, st):
    loss = 0.0
    for i in range(len(x)):
        f = circuit(param, x[i], st[y[i]])
        loss = loss + (1 - f) ** 2
    return loss / len(x)


def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


def test(params, x, y, states):
    """
    Tests on a given set of data.

    Returns:
        predicted (array([int]): predicted labels for test data
        output_states (array[float]): output quantum states from the circuit
    """
    fidelity_values = []
    predicted = []

    for i in range(len(x)):
        fidelities = [circuit(params, x[i], st) for st in states]
        best_fidel = np.argmax(fidelities)

        predicted.append(best_fidel)
        fidelity_values.append(fidelities)

    return np.array(predicted), np.array(fidelity_values)


def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Returns:
        score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred
    return score.sum() / len(y_true)


def train(dta, num_layers=8, learning_rate=0.01, epochs=10, batch_size=32):
    X_train, y_train, X_val, y_val = dta

    states = [np.zeros(2) for i in range(2)]
    for i in range(2):
        states[i][i] = 1

    # Seed: same number of layers will produce the same initial weights
    np.random.seed(2)
    params = np.random.uniform(size=(num_layers, 6), requires_grad=True)

    # Zero epoch
    predicted_train, fidel_train = test(params, X_train, y_train, states)
    accuracy_train = accuracy_score(y_train, predicted_train)

    predicted_val, fidel_val = test(params, X_val, y_val, states)
    accuracy_val = accuracy_score(y_val, predicted_val)

    # save predictions with random weights for comparison
    initial_predictions = predicted_val

    loss = cost(params, X_val, y_val, states)

    print(
        "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Val Accuracy: {:3f}".format(
            0, loss, accuracy_train, accuracy_val
        )
    )

    opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    for it in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(
            X_train, y_train, batch_size=batch_size
        ):
            params, _, _, _ = opt.step(cost, params, Xbatch, ybatch, states)

        predicted_train, fidel_train = test(params, X_train, y_train, states)
        accuracy_train = accuracy_score(y_train, predicted_train)
        loss = cost(params, X_train, y_train, states)

        predicted_val, fidel_val = test(params, X_val, y_val, states)
        accuracy_val = accuracy_score(y_val, predicted_val)
        res = [it + 1, loss, accuracy_train, accuracy_val]
        print(
            "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Val accuracy: {:3f}".format(
                *res
            )
        )

    return {
        "params": params,
        "accuracy_train": accuracy_train,
        "accuracy_val": accuracy_val,
        "initial_predictions": initial_predictions,
        "predicted_val": predicted_val,
    }


def main():
    # Set a random seed: better reproducibility
    np.random.seed(42)

    num_training = 200
    num_val = 400

    X_data, y_train = two_spirals(num_training)
    X_train = np.hstack((X_data, np.zeros((X_data.shape[0], 1), requires_grad=False)))

    X_val, y_val = two_spirals(num_val)
    X_val = np.hstack((X_val, np.zeros((X_val.shape[0], 1), requires_grad=False)))

    # Fixed number of epochs
    epochs = 10

    # Number of experiments for hyperparameters search
    n_experiments = 30

    experiments = []
    # Hyperparameters optimization by random search
    # Generate all hyperparameters for the experiments.
    # Note, that for a more reliable result, one should
    # average the accuracies over several runs with the same
    # hyperparameters but different initial parameters (angles).
    for i in range(n_experiments):
        e = {
            "num_layers": np.random.randint(6, 16 + 1),
            "learning_rate": round(10 ** np.random.uniform(-3, -1), 4),
            "batch_size": np.random.choice([16, 32, 64]),
        }
        experiments.append(e)

    # Save the experiments plan to a CSV file.
    # This might be helpful e.g. to resume the experiments
    # if the script is interrupted.
    df = pd.DataFrame(experiments)
    df.to_csv("experiments_plan_spirals.csv")

    for i, e in enumerate(experiments):
        num_layers = e["num_layers"]
        learning_rate = e["learning_rate"]
        batch_size = e["batch_size"]

        print(
            f"Training {i}, num_layers={num_layers}, learning_rate={learning_rate}, "
            f"batch_size={batch_size}"
        )

        r = train(
            (X_train, y_train, X_val, y_val),
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Store the configuration and results to a CSV file
        res = {
            "accuracy_train": r["accuracy_train"],
            "accuracy_val": r["accuracy_val"],
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            # Convert parameters to a string with :.6f formatting
            "params": ";".join(
                [",".join([f"{p:.6f}" for p in pp]) for pp in r["params"]]
            ),
        }
        print()

        # Append the results to the CSV file
        res = {k: [v] for k, v in res.items()}
        df = pd.DataFrame(res)
        df.to_csv(
            "results_spirals.csv", mode="a", header=(i == 0), sep=" ", index=False
        )


if __name__ == "__main__":
    main()
