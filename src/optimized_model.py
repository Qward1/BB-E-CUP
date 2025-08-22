import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')


class OptimizedCounterfeitDetector:
    """
    Оптимизированная модель с фокусом на стабильность train/test
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.threshold = 0.5
        self.important_features = None
        self.feature_stats = {}

    def analyze_data_quality(self, train_df, test_df):
        """
        Анализ качества данных и различий train/test
        """
        print("=" * 70)
        print("АНАЛИЗ ДАННЫХ")
        print("=" * 70)

        print(f"\nРазмеры:")
        print(f"Train: {train_df.shape}")
        print(f"Test: {test_df.shape}")

        # Проверяем пропуски
        train_nulls = train_df.isnull().sum()
        test_nulls = test_df.isnull().sum()

        print(f"\nПропуски в train (топ-10):")
        print(train_nulls[train_nulls > 0].sort_values(ascending=False).head(10))

        print(f"\nПропуски в test (топ-10):")
        print(test_nulls[test_nulls > 0].sort_values(ascending=False).head(10))

        # Проверяем уникальные значения в категориальных признаках
        categorical_cols = ['brand_name', 'CommercialTypeName4', 'SellerID']

        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_unique = set(train_df[col].dropna().unique())
                test_unique = set(test_df[col].dropna().unique())

                only_in_test = test_unique - train_unique
                only_in_train = train_unique - test_unique

                print(f"\n{col}:")
                print(f"  Уникальных в train: {len(train_unique)}")
                print(f"  Уникальных в test: {len(test_unique)}")
                print(f"  Только в test: {len(only_in_test)} ({len(only_in_test) / len(test_unique) * 100:.1f}%)")
                print(f"  Только в train: {len(only_in_train)} ({len(only_in_train) / len(train_unique) * 100:.1f}%)")

        return train_nulls, test_nulls

    def create_stable_features(self, df, is_train=True):
        """
        Создание стабильных признаков, которые хорошо работают на test
        """
        print("\nСоздание стабильных признаков...")

        df = df.copy()

        # 1. Базовые соотношения (не зависят от абсолютных значений)
        # Рейтинги
        if 'total_ratings' in df.columns:
            df['has_ratings'] = (df['total_ratings'] > 0).astype(int)
            df['low_ratings_flag'] = (df['total_ratings'] < 10).astype(int)
            df['high_ratings_flag'] = (df['total_ratings'] > 100).astype(int)

        # Соотношения рейтингов (устойчивы к масштабу)
        if 'rating_5_ratio' in df.columns and 'rating_1_ratio' in df.columns:
            df['positive_negative_ratio'] = df['rating_5_ratio'] / (df['rating_1_ratio'] + 0.001)
            df['extreme_ratings'] = df['rating_1_ratio'] + df['rating_5_ratio']

        # 2. Аномалии в возвратах (относительные метрики)
        if 'return_rate_30d' in df.columns:
            df['high_return_flag'] = (df['return_rate_30d'] > 0.3).astype(int)
            df['suspicious_returns'] = (df['return_rate_30d'] > 0.5).astype(int)

        # 3. Паттерны продавца (категориальные)
        if 'seller_age_months' in df.columns:
            df['new_seller'] = (df['seller_age_months'] < 6).astype(int)
            df['established_seller'] = (df['seller_age_months'] > 24).astype(int)

        # 4. Ценовые аномалии (квантили)
        if 'PriceDiscounted' in df.columns:
            # Используем квантили для устойчивости
            price_q1 = df['PriceDiscounted'].quantile(0.25)
            price_q3 = df['PriceDiscounted'].quantile(0.75)
            df['price_outlier'] = ((df['PriceDiscounted'] < price_q1 * 0.5) |
                                   (df['PriceDiscounted'] > price_q3 * 2)).astype(int)

        # 5. Комплексные индикаторы риска
        risk_score = 0

        if 'fake_return_rate_30d' in df.columns:
            risk_score += (df['fake_return_rate_30d'] > 0).astype(int) * 2

        if 'return_anomaly_score' in df.columns:
            risk_score += (df['return_anomaly_score'] > df['return_anomaly_score'].quantile(0.9)).astype(int)

        if 'rating_anomaly_score' in df.columns:
            risk_score += (df['rating_anomaly_score'] > df['rating_anomaly_score'].quantile(0.9)).astype(int)

        df['risk_level'] = risk_score

        # 6. Убираем нестабильные признаки
        unstable_features = [
            'item_id', 'SellerID',  # ID могут быть разные в test
            'GmvTotal7', 'GmvTotal30', 'GmvTotal90',  # Абсолютные значения
            'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30',  # Могут сильно отличаться
        ]

        for feature in unstable_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])

        return df

    def select_best_features(self, X_train, y_train, max_features=80):
        """
        Выбор лучших признаков с учетом стабильности
        """
        print(f"\nВыбор топ-{max_features} признаков...")

        # Исключаем эмбеддинги из основного набора
        base_features = [col for col in X_train.columns
                         if not col.startswith(('embed_', 'namerus_embed_'))]

        # Берем только несколько главных компонент эмбеддингов
        embed_features = [f'embed_{i}' for i in range(5) if f'embed_{i}' in X_train.columns]
        namerus_features = [f'namerus_embed_{i}' for i in range(5) if f'namerus_embed_{i}' in X_train.columns]

        selected_features = base_features + embed_features + namerus_features

        # Обучаем легкую модель для оценки важности
        temp_model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=self.random_state,
            verbosity=0
        )

        temp_model.fit(X_train[selected_features].fillna(0), y_train)

        # Получаем важность
        importance = pd.DataFrame({
            'feature': selected_features,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Выбираем топ признаков
        self.important_features = importance.head(max_features)['feature'].tolist()

        print(f"Выбрано {len(self.important_features)} признаков")
        print("\nТоп-15 важных признаков:")
        print(importance.head(15)[['feature', 'importance']])

        # Сохраняем статистики для нормализации
        for feature in self.important_features:
            if feature in X_train.columns:
                self.feature_stats[feature] = {
                    'mean': X_train[feature].mean(),
                    'std': X_train[feature].std() + 1e-6
                }

        return self.important_features

    def normalize_features(self, X, is_train=False):
        """
        Нормализация признаков для стабильности
        """
        X_normalized = X.copy()

        for feature in self.important_features:
            if feature in X_normalized.columns and feature in self.feature_stats:
                stats = self.feature_stats[feature]
                X_normalized[feature] = (X_normalized[feature] - stats['mean']) / stats['std']
                # Обрезаем выбросы
                X_normalized[feature] = X_normalized[feature].clip(-3, 3)

        return X_normalized

    def train_conservative_model(self, X_train, y_train, X_val, y_val):
        """
        Обучение консервативной модели с защитой от переобучения
        """
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ КОНСЕРВАТИВНОЙ МОДЕЛИ")
        print("=" * 70)

        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Параметры для минимизации переобучения
        params = {
            'n_estimators': 150,
            'max_depth': 5,  # Очень мелкие деревья
            'learning_rate': 0.03,  # Медленное обучение
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'min_child_weight': 10,  # Большое значение для регуляризации
            'gamma': 0.3,  # Сильная регуляризация
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight * 0.8,  # Немного уменьшаем для баланса
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': self.random_state,
            'verbosity': 0,
            'n_jobs': -1
        }

        self.model = XGBClassifier(**params)

        # Обучаем с early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Оценка
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)

        print(f"\nРезультаты обучения:")
        print(f"Train F1: {train_f1:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print(f"Overfitting gap: {train_f1 - val_f1:.4f}")

        if train_f1 - val_f1 > 0.15:
            print("⚠️ Высокое переобучение! Модель может плохо работать на test")

        return train_f1, val_f1

    def optimize_threshold_conservative(self, X_val, y_val):
        """
        Консервативная оптимизация порога
        """
        print("\n" + "=" * 70)
        print("ОПТИМИЗАЦИЯ ПОРОГА")
        print("=" * 70)

        y_proba = self.model.predict_proba(X_val)[:, 1]

        # Тестируем пороги
        thresholds = np.linspace(0.3, 0.7, 21)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })

        results_df = pd.DataFrame(results)

        # Выбираем порог с лучшим F1, но не слишком экстремальный
        best_f1_idx = results_df['f1'].idxmax()
        best_threshold = results_df.loc[best_f1_idx, 'threshold']

        # Если порог слишком низкий или высокий, корректируем
        if best_threshold < 0.35:
            best_threshold = 0.35
        elif best_threshold > 0.65:
            best_threshold = 0.65

        self.threshold = best_threshold

        print(f"\nОптимальный порог: {self.threshold:.3f}")
        print(f"F1 при этом пороге: {results_df.loc[best_f1_idx, 'f1']:.4f}")
        print(f"Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
        print(f"Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")

        return self.threshold

    def fit(self, train_df, test_df, target_col='resolution'):
        """
        Полный пайплайн обучения
        """
        print("\n" + "=" * 70)
        print("ПОЛНЫЙ ПАЙПЛАЙН")
        print("=" * 70)

        # 1. Анализ данных
        self.analyze_data_quality(train_df, test_df)

        # 2. Создание стабильных признаков
        train_df = self.create_stable_features(train_df, is_train=True)
        test_df = self.create_stable_features(test_df, is_train=False)

        # 3. Разделение на train/val
        from sklearn.model_selection import train_test_split

        X = train_df.drop(columns=[target_col] if target_col in train_df.columns else [])
        y = train_df[target_col] if target_col in train_df.columns else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # 4. Выбор признаков
        self.select_best_features(X_train, y_train)

        # 5. Применяем отбор признаков
        X_train = X_train[self.important_features].fillna(0)
        X_val = X_val[self.important_features].fillna(0)

        # 6. Нормализация
        X_train = self.normalize_features(X_train, is_train=True)
        X_val = self.normalize_features(X_val, is_train=False)

        # 7. Обучение модели
        train_f1, val_f1 = self.train_conservative_model(X_train, y_train, X_val, y_val)

        # 8. Оптимизация порога
        self.optimize_threshold_conservative(X_val, y_val)

        # 9. Финальная проверка
        y_val_pred = self.predict(X_val)
        final_f1 = f1_score(y_val, y_val_pred)

        print("\n" + "=" * 70)
        print(f"📊 ФИНАЛЬНЫЙ F1 на валидации: {final_f1:.4f}")
        print("=" * 70)

        return self

    def predict(self, X):
        """
        Предсказание с оптимальным порогом
        """
        # Применяем те же преобразования
        X = self.create_stable_features(X, is_train=False)

        # Выбираем нужные признаки
        missing_features = set(self.important_features) - set(X.columns)
        if missing_features:
            print(f"⚠️ Отсутствуют признаки: {missing_features}")
            for feature in missing_features:
                X[feature] = 0

        X = X[self.important_features].fillna(0)
        X = self.normalize_features(X, is_train=False)

        # Предсказываем вероятности
        y_proba = self.model.predict_proba(X)[:, 1]

        # Применяем порог
        predictions = (y_proba >= self.threshold).astype(int)

        return predictions

    def predict_proba(self, X):
        """
        Вероятности предсказаний
        """
        X = self.create_stable_features(X, is_train=False)

        missing_features = set(self.important_features) - set(X.columns)
        for feature in missing_features:
            X[feature] = 0

        X = X[self.important_features].fillna(0)
        X = self.normalize_features(X, is_train=False)

        return self.model.predict_proba(X)[:, 1]


# ========================================
# КРОСС-ВАЛИДАЦИЯ ДЛЯ ОЦЕНКИ СТАБИЛЬНОСТИ
# ========================================



# ========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ========================================




# ========================================
# ИСПОЛЬЗОВАНИЕ
# ========================================

