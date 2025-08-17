"""
Late Fusion: объединение предсказаний
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


class LateFusion:
    """Объединение предсказаний от разных моделей"""

    def __init__(self, config: dict):
        self.config = config
        self.weights = {'text': 0.5, 'metadata': 0.5}
        self.threshold = 0.5

    def optimize(self, text_probs: np.ndarray, meta_probs: np.ndarray,
                 y_true: np.ndarray):
        """Оптимизация весов и порога"""

        # Оптимизация весов
        def objective(w):
            combined = w[0] * text_probs + (1 - w[0]) * meta_probs
            preds = (combined > 0.5).astype(int)
            return -f1_score(y_true, preds)

        result = minimize(objective, x0=[0.5], bounds=[(0, 1)], method='SLSQP')
        self.weights['text'] = result.x[0]
        self.weights['metadata'] = 1 - result.x[0]

        # Оптимизация порога
        combined = self.weights['text'] * text_probs + self.weights['metadata'] * meta_probs

        best_threshold = 0.5
        best_f1 = 0

        for thr in np.arange(0.3, 0.7, 0.01):
            preds = (combined > thr).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        self.threshold = best_threshold
        print(f"Optimized weights: text={self.weights['text']:.3f}, "
              f"metadata={self.weights['metadata']:.3f}, threshold={self.threshold:.3f}")

    def predict(self, text_probs: np.ndarray, meta_probs: np.ndarray,
                text_available: np.ndarray = None,
                meta_available: np.ndarray = None) -> np.ndarray:
        """
        Адаптивное объединение с учетом доступности данных

        Args:
            text_probs: Вероятности от text модели
            meta_probs: Вероятности от metadata модели
            text_available: Маска доступности текстовых данных
            meta_available: Маска доступности метаданных
        """

        if text_available is None:
            text_available = np.ones(len(text_probs), dtype=bool)
        if meta_available is None:
            meta_available = np.ones(len(meta_probs), dtype=bool)

        # Адаптивные веса
        predictions = np.zeros(len(text_probs))

        # Случай 1: Есть оба типа данных
        both_available = text_available & meta_available
        if both_available.any():
            combined = (self.weights['text'] * text_probs[both_available] +
                        self.weights['metadata'] * meta_probs[both_available])
            predictions[both_available] = (combined > self.threshold).astype(int)

        # Случай 2: Только текст
        text_only = text_available & ~meta_available
        if text_only.any():
            predictions[text_only] = (text_probs[text_only] > self.threshold).astype(int)

        # Случай 3: Только метаданные
        meta_only = ~text_available & meta_available
        if meta_only.any():
            predictions[meta_only] = (meta_probs[meta_only] > self.threshold).astype(int)

        # Случай 4: Нет данных - предсказываем класс 0
        no_data = ~text_available & ~meta_available
        if no_data.any():
            predictions[no_data] = 0

        return predictions.astype(int)