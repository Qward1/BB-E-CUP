"""
Единый модуль для работы с данными
Объединяет: загрузку, оптимизацию памяти, обработку пропусков
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Generator, Tuple, Optional


class DataProcessor:
    """Единый класс для всех операций с данными"""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config['data']['batch_size']

    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация типов данных для экономии памяти"""
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != 'object':
                c_min, c_max = df[col].min(), df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

        return df

    def load_train_data(self, path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Загрузка обучающих данных"""
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
            df = self.optimize_memory(df)

        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=self.config['project']['seed'])

        return df

    def load_test_batches(self, path: str) -> Generator[pd.DataFrame, None, None]:
        """Генератор батчей для inference"""
        if path.endswith('.parquet'):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                yield self.optimize_memory(batch.to_pandas())
        else:
            for chunk in pd.read_csv(path, chunksize=self.batch_size):
                yield self.optimize_memory(chunk)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Разделение на текст, метаданные и метки"""
        # Текст
        text_cols = self.config['data']['text_columns']
        available_text = [col for col in text_cols if col in df.columns]

        if available_text:
            # Объединяем все текстовые поля
            texts = df[available_text].fillna('').agg(' '.join, axis=1)
        else:
            texts = pd.Series([''] * len(df))

        # Метаданные
        meta_cols = self.config['data']['metadata_columns']
        available_meta = [col for col in meta_cols if col in df.columns]

        if available_meta:
            metadata = df[available_meta].copy()
        else:
            metadata = pd.DataFrame(index=df.index)

        # Метки (если есть)
        target_col = self.config['data']['target_column']
        labels = df[target_col] if target_col in df.columns else None

        return texts, metadata, labels

    def handle_missing(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропусков в метаданных"""
        # Числовые - медиана
        numeric_cols = metadata.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            metadata[col].fillna(metadata[col].median(), inplace=True)

        # Категориальные - мода или 'unknown'
        categorical_cols = metadata.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if metadata[col].mode().empty:
                metadata[col].fillna('unknown', inplace=True)
            else:
                metadata[col].fillna(metadata[col].mode()[0], inplace=True)

        return metadata