"""
Metadata Pipeline: feature engineering + модель
"""

import pandas as pd
import numpy as np
from typing import Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MetadataPipeline:
    """Полный pipeline для метаданных"""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}

    def create_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Feature engineering для метаданных"""
        features = pd.DataFrame(index=df.index)

        # Ценовые признаки
        if 'price' in df.columns:
            if fit:
                self.feature_stats['price_mean'] = df['price'].mean()
                self.feature_stats['price_std'] = df['price'].std()

            features['price_zscore'] = (df['price'] - self.feature_stats.get('price_mean', df['price'].mean())) / \
                                       self.feature_stats.get('price_std', df['price'].std() + 1e-6)
            features['price_log'] = np.log1p(df['price'])

        # Категориальные признаки
        if 'brand' in df.columns:
            if fit:
                self.label_encoders['brand'] = LabelEncoder()
                # Добавляем 'unknown' для обработки новых брендов
                all_brands = list(df['brand'].fillna('unknown').unique()) + ['unknown']
                self.label_encoders['brand'].fit(all_brands)

            # Безопасное кодирование с обработкой неизвестных значений
            brand_encoded = df['brand'].fillna('unknown').apply(
                lambda x: self._safe_transform(self.label_encoders['brand'], x)
            )
            features['brand_encoded'] = brand_encoded

            # Популярность бренда
            if fit:
                self.feature_stats['brand_counts'] = df['brand'].value_counts().to_dict()

            features['brand_popularity'] = df['brand'].map(
                self.feature_stats.get('brand_counts', {})
            ).fillna(0)

        # Признаки продавца
        if 'seller_rating' in df.columns:
            features['seller_rating'] = df['seller_rating'].fillna(df['seller_rating'].median())
            features['low_seller_rating'] = (features['seller_rating'] < 4.0).astype(int)

        # Категория
        if 'category' in df.columns:
            if fit:
                self.label_encoders['category'] = LabelEncoder()
                all_categories = list(df['category'].fillna('unknown').unique()) + ['unknown']
                self.label_encoders['category'].fit(all_categories)

            features['category_encoded'] = df['category'].fillna('unknown').apply(
                lambda x: self._safe_transform(self.label_encoders['category'], x)
            )

        # Cross-features
        if 'price' in df.columns and 'category' in df.columns:
            if fit:
                self.feature_stats['category_price_median'] = df.groupby('category')['price'].median().to_dict()

            features['price_to_category_median'] = df.apply(
                lambda row: row['price'] / self.feature_stats.get('category_price_median', {}).get(row['category'],
                                                                                                   row['price']),
                axis=1
            )

        return features

    def _safe_transform(self, encoder: LabelEncoder, value):
        """Безопасное преобразование с обработкой неизвестных значений"""
        try:
            return encoder.transform([value])[0]
        except:
            # Если значение не встречалось при обучении
            return encoder.transform(['unknown'])[0]

    def fit(self, metadata: pd.DataFrame, labels: pd.Series):
        """Обучение модели"""
        # Feature engineering
        features = self.create_features(metadata, fit=True)

        # Заполнение пропусков
        features = features.fillna(0)

        # Масштабирование
        features_scaled = self.scaler.fit_transform(features)

        # Обучение XGBoost
        params = self.config['metadata_model']['params']
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(features_scaled, labels)

        # Сохраняем названия признаков
        self.feature_names = features.columns.tolist()

    def predict_proba(self, metadata: pd.DataFrame) -> np.ndarray:
        """Предсказание вероятностей"""
        # Feature engineering
        features = self.create_features(metadata, fit=False)

        # Заполнение пропусков
        features = features.fillna(0)

        # Убеждаемся, что есть все нужные колонки
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        features = features[self.feature_names]

        # Масштабирование
        features_scaled = self.scaler.transform(features)

        # Предсказание
        if self.model is None:
            # Если модель не обучена, возвращаем нейтральные вероятности
            return np.ones(len(metadata)) * 0.5

        return self.model.predict_proba(features_scaled)[:, 1]