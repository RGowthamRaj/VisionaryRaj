"""
Feature Engineering for Hand Gesture Recognition
================================================
Provides GestureFeatureExtractor to compute:
1. Raw coordinates (63 features).
2. Translation- and Scale-Invariant coordinates (63 features):
   - Translation Invariance: Subtract wrist (landmark 0) from all points.
   - Scale Invariance: Divide all coordinates by Euclidean distance
     from wrist (0) to middle-finger MCP (9).
"""

import numpy as np


class GestureFeatureExtractor:
    """
    Extracts raw and invariant features from MediaPipe hand landmarks.
    Works with:
    - MediaPipe landmark object lists (len 21)
    - 1D flat sequences/arrays of 63 coordinates
    - 2D arrays of shape (21, 3)
    - Batches of shape (N, 63)
    """

    @staticmethod
    def _parse_landmarks(landmarks):
        """
        Normalizes various input formats into a (21, 3) numpy array.
        """
        if isinstance(landmarks, np.ndarray):
            if landmarks.ndim == 1 and len(landmarks) == 63:
                return landmarks.reshape(21, 3).astype(np.float32)
            elif landmarks.ndim == 2 and landmarks.shape == (21, 3):
                return landmarks.astype(np.float32)
            elif landmarks.ndim == 1 and len(landmarks) == 21:
                # Array of objects with x, y, z
                coords = [[lm.x, lm.y, lm.z] for lm in landmarks]
                return np.array(coords, dtype=np.float32)

        elif isinstance(landmarks, (list, tuple)):
            if len(landmarks) == 63 and isinstance(landmarks[0], (int, float, np.number)):
                return np.array(landmarks, dtype=np.float32).reshape(21, 3)
            elif len(landmarks) == 21:
                coords = []
                for lm in landmarks:
                    if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                        coords.append([lm.x, lm.y, lm.z])
                    elif isinstance(lm, (list, tuple)) and len(lm) == 3:
                        coords.append(lm)
                    else:
                        raise ValueError(f"Unrecognized landmark element format: {type(lm)}")
                return np.array(coords, dtype=np.float32)

        raise ValueError(
            f"Invalid landmarks format. Expected 21 landmarks or 63 coordinates, got: {type(landmarks)}"
        )

    def extract_raw(self, landmarks):
        """
        Returns the 63 raw coordinates [x0, y0, z0, ..., x20, y20, z20].
        """
        pts = self._parse_landmarks(landmarks)
        return pts.flatten()

    def extract_invariant(self, landmarks):
        """
        Extracts translation- and scale-invariant coordinates:
        1. Translation Invariance: Subtract wrist (landmark 0) from all points.
        2. Scale Invariance: Divide by distance between wrist (0) and middle MCP (9).
        """
        pts = self._parse_landmarks(landmarks)  # shape (21, 3)

        # 1. Translation Invariance: Subtract wrist (index 0)
        wrist = pts[0]
        translated = pts - wrist

        # 2. Scale Invariance: Euclidean distance from wrist (0) to middle MCP (9)
        scale_dist = np.linalg.norm(translated[9])

        # Prevent division by zero
        if scale_dist < 1e-6:
            scale_dist = 1.0

        invariant = translated / scale_dist
        return invariant.flatten()

    def extract_raw_batch(self, X):
        """
        Vectorized raw feature extraction for an (N, 63) array or DataFrame.
        """
        return np.asarray(X, dtype=np.float32)

    def extract_invariant_batch(self, X):
        """
        Vectorized invariant feature extraction for an (N, 63) matrix:
        1. Subtract landmark 0 (wrist) for each sample.
        2. Divide by Euclidean distance from landmark 0 to landmark 9 (middle MCP).
        """
        arr = np.asarray(X, dtype=np.float32)
        N = arr.shape[0]
        pts = arr.reshape(N, 21, 3)

        # 1. Translation Invariance: Subtract wrist (0) from all 21 points
        wrist = pts[:, 0:1, :]  # shape (N, 1, 3)
        translated = pts - wrist  # shape (N, 21, 3)

        # 2. Scale Invariance: Distance between wrist (0) and middle-finger MCP (9)
        scale_dist = np.linalg.norm(translated[:, 9, :], axis=1, keepdims=True)  # shape (N, 1)
        scale_dist = np.where(scale_dist < 1e-6, 1.0, scale_dist)
        scale_dist = scale_dist[:, :, np.newaxis]  # shape (N, 1, 1)

        invariant = translated / scale_dist
        return invariant.reshape(N, 63)
