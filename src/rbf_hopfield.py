"""Radial basis function and Hopfield neural networks."""
from __future__ import annotations

import numpy as np


class RadialBasisFunctionNetwork:
    """Gaussian RBF network trained with k-means centers and ridge regression.

    ``fit`` accepts one- or multi-output targets. ``predict`` returns the
    corresponding continuous output scores; use ``predict_classes`` for
    thresholded binary predictions or argmax multiclass predictions.
    """

    def __init__(
        self,
        n_centers: int = 10,
        gamma: float | None = None,
        ridge: float = 1e-6,
        random_state: int | None = 42,
    ) -> None:
        if n_centers < 1:
            raise ValueError("n_centers must be at least 1.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be greater than 0.")
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")

        self.n_centers = n_centers
        self.gamma = gamma
        self.ridge = ridge
        self.random_state = random_state
        self.centers_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.gamma_: float | None = None
        self._target_was_1d = False

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "RadialBasisFunctionNetwork":
        """Fit the network to feature rows ``X`` and continuous targets ``y``."""
        X = self._features(X, "X")
        y = np.asarray(y, dtype=np.float64)
        self._validate_finite(y, "y")
        if y.ndim == 1:
            self._target_was_1d = True
            y = y.reshape(-1, 1)
        elif y.ndim == 2:
            self._target_was_1d = False
        else:
            raise ValueError("y must be a 1D or 2D array.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if y.shape[1] == 0:
            raise ValueError("y must contain at least one target column.")

        from sklearn.cluster import KMeans

        n_centers = min(self.n_centers, X.shape[0])
        self.centers_ = KMeans(
            n_clusters=n_centers,
            random_state=self.random_state,
            n_init=10,
        ).fit(X).cluster_centers_

        if self.gamma is None:
            center_distances = self._squared_distances(
                self.centers_, self.centers_
            )
            np.fill_diagonal(center_distances, np.inf)
            nearest_squared = center_distances.min(axis=1)
            positive_distances = nearest_squared[np.isfinite(nearest_squared)]
            positive_distances = positive_distances[positive_distances > 0]
            scale = (
                float(positive_distances.mean())
                if positive_distances.size
                else 1.0
            )
            self.gamma_ = 1.0 / (2.0 * scale)
        else:
            self.gamma_ = float(self.gamma)

        activations = self._design_matrix(X)
        gram = activations.T @ activations
        penalty = np.eye(gram.shape[0]) * self.ridge
        penalty[-1, -1] = 0.0
        self.weights_ = np.linalg.solve(
            gram + penalty, activations.T @ y
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return Gaussian activations for ``X`` and the fitted centers."""
        self._require_fitted()
        X = self._features(X, "X")
        if X.shape[1] != self.centers_.shape[1]:
            raise ValueError("X has a different number of features than training data.")
        return np.exp(-self.gamma_ * self._squared_distances(X, self.centers_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous output scores for each row of ``X``."""
        self._require_fitted()
        activations = self.transform(X)
        design = np.column_stack((activations, np.ones(activations.shape[0])))
        predictions = design @ self.weights_
        return predictions[:, 0] if self._target_was_1d else predictions

    def predict_classes(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Convert scores to binary labels, or choose the highest multiclass score."""
        predictions = np.asarray(self.predict(X))
        if predictions.ndim == 1:
            return (predictions >= threshold).astype(int)
        if predictions.shape[1] == 1:
            return (predictions[:, 0] >= threshold).astype(int)
        return np.argmax(predictions, axis=1)

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        activations = np.exp(
            -self.gamma_ * self._squared_distances(X, self.centers_)
        )
        return np.column_stack((activations, np.ones(activations.shape[0])))

    @staticmethod
    def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distances = (
            np.sum(left * left, axis=1)[:, None]
            + np.sum(right * right, axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        return np.maximum(distances, 0.0)

    @staticmethod
    def _features(values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D array.")
        RadialBasisFunctionNetwork._validate_finite(values, name)
        return values

    @staticmethod
    def _validate_finite(values: np.ndarray, name: str) -> None:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")

    def _require_fitted(self) -> None:
        if self.centers_ is None or self.weights_ is None or self.gamma_ is None:
            raise RuntimeError("Call fit before transform or predict.")


class HopfieldNetwork:
    """Hebbian Hopfield associative memory for bipolar (-1/+1) patterns."""

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None

    def fit(self, patterns: np.ndarray) -> "HopfieldNetwork":
        """Store patterns and compute symmetric Hebbian connection weights."""
        patterns = np.asarray(patterns, dtype=np.float64)
        if (
            patterns.ndim != 2
            or patterns.shape[0] == 0
            or patterns.shape[1] == 0
        ):
            raise ValueError("patterns must be a non-empty 2D array.")
        self._validate_bipolar(patterns, "patterns")

        n_neurons = patterns.shape[1]
        self.weights_ = patterns.T @ patterns / n_neurons
        np.fill_diagonal(self.weights_, 0.0)
        return self

    def recall(
        self,
        pattern: np.ndarray,
        max_iter: int = 100,
        update: str = "asynchronous",
        random_state: int | None = None,
    ) -> np.ndarray:
        """Recall a stored pattern using synchronous or asynchronous updates.

        Asynchronous updates process neurons in order, or in a randomized order
        when ``random_state`` is provided. Neurons with zero net input keep
        their current state.
        """
        self._require_fitted()
        state = np.asarray(pattern, dtype=np.float64)
        if state.ndim != 1 or state.size != self.weights_.shape[0]:
            raise ValueError("pattern must be a 1D vector matching the network size.")
        self._validate_bipolar(state, "pattern")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if update not in {"asynchronous", "synchronous"}:
            raise ValueError("update must be 'asynchronous' or 'synchronous'.")

        state = state.copy()
        rng = np.random.default_rng(random_state)
        for _ in range(max_iter):
            previous = state.copy()
            if update == "synchronous":
                state = self._activate(self.weights_ @ state, state)
            else:
                order = rng.permutation(state.size) if random_state is not None else range(state.size)
                for neuron in order:
                    state[neuron] = self._activate(
                        np.array([self.weights_[neuron] @ state]),
                        np.array([state[neuron]]),
                    )[0]
            if np.array_equal(state, previous):
                break
        return state.astype(int)

    def energy(self, pattern: np.ndarray) -> float:
        """Return the Hopfield energy of a bipolar state."""
        self._require_fitted()
        state = np.asarray(pattern, dtype=np.float64)
        if state.ndim != 1 or state.size != self.weights_.shape[0]:
            raise ValueError("pattern must be a 1D vector matching the network size.")
        self._validate_bipolar(state, "pattern")
        return float(-0.5 * state @ self.weights_ @ state)

    @staticmethod
    def _activate(inputs: np.ndarray, previous: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1.0, np.where(inputs < 0, -1.0, previous))

    @staticmethod
    def _validate_bipolar(values: np.ndarray, name: str) -> None:
        if not np.all(np.isfinite(values)) or not np.all(
            (values == -1) | (values == 1)
        ):
            raise ValueError(f"{name} must contain only -1 and +1 values.")

    def _require_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit before recall or energy.")
