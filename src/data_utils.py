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

    def extract_counterfeit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция для извлечения признаков для определения контрафакта

        Args:
            df: DataFrame с колонками:
                - id, brand_name, name_rus, CommercialTypeName4
                - rating_1_count, rating_2_count, rating_3_count, rating_4_count, rating_5_count
                - comments_published_count, photos_published_count, videos_published_count
                - PriceDiscounted, item_time_alive
                - item_count_fake_returns7/30/90, item_count_sales7/30/90, item_count_returns7/30/90
                - GmvTotal7/30/90, ExemplarAcceptedCountTotal7/30/90
                - OrderAcceptedCountTotal7/30/90, ExemplarReturnedCountTotal7/30/90
                - ExemplarReturnedValueTotal7/30/90
                - ItemVarietyCount, ItemAvailableCount, seller_time_alive
                - ItemID, SellerID

        Returns:
            DataFrame с исходными и новыми признаками
        """

        # Создаем копию датафрейма
        result_df = df.copy()

        # ========================================
        # 1. ПРИЗНАКИ ИЗ РЕЙТИНГОВ И ОТЗЫВОВ
        # ========================================

        # Заполняем пропуски нулями для рейтингов
        rating_cols = ['rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count']
        for col in rating_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Общее количество рейтингов
        result_df['total_ratings'] = result_df[rating_cols].sum(axis=1)

        # Процентное распределение рейтингов
        for i, col in enumerate(rating_cols, 1):
            result_df[f'rating_{i}_ratio'] = result_df[col] / (result_df['total_ratings'] + 1)

        # Средний взвешенный рейтинг
        result_df['avg_rating'] = (
                                          result_df['rating_1_count'] * 1 +
                                          result_df['rating_2_count'] * 2 +
                                          result_df['rating_3_count'] * 3 +
                                          result_df['rating_4_count'] * 4 +
                                          result_df['rating_5_count'] * 5
                                  ) / (result_df['total_ratings'] + 1)

        # Дисперсия и стандартное отклонение рейтингов
        def calculate_rating_stats(row):
            vals = []
            for i, col in enumerate(rating_cols, 1):
                vals.extend([i] * int(row[col]))
            if vals:
                return pd.Series([np.std(vals), np.var(vals)])
            return pd.Series([0, 0])

        result_df[['rating_std', 'rating_variance']] = result_df.apply(calculate_rating_stats, axis=1)

        # Энтропия распределения рейтингов
        def calculate_entropy(row):
            total = row['total_ratings']
            if total == 0:
                return 0
            entropy = 0
            for col in rating_cols:
                if row[col] > 0:
                    p = row[col] / total
                    entropy -= p * np.log(p + 1e-10)
            return entropy

        result_df['rating_entropy'] = result_df.apply(calculate_entropy, axis=1)

        # Коэффициент поляризации (много крайних оценок)
        result_df['rating_polarization'] = (
                                                   result_df['rating_1_count'] + result_df['rating_5_count']
                                           ) / (result_df['total_ratings'] + 1)

        # U-образное распределение (подозрительный паттерн)
        result_df['rating_u_shape'] = (
                                              (result_df['rating_1_count'] + result_df['rating_5_count']) -
                                              (result_df['rating_2_count'] + result_df['rating_3_count'] + result_df[
                                                  'rating_4_count'])
                                      ) / (result_df['total_ratings'] + 1)

        # Перекос в сторону положительных оценок
        result_df['positive_skew'] = (
                                             result_df['rating_4_count'] + result_df['rating_5_count']
                                     ) / (result_df['total_ratings'] + 1)

        # Перекос в сторону отрицательных оценок
        result_df['negative_skew'] = (
                                             result_df['rating_1_count'] + result_df['rating_2_count']
                                     ) / (result_df['total_ratings'] + 1)

        # Доля максимальных оценок (5 звезд)
        result_df['perfect_rating_ratio'] = result_df['rating_5_count'] / (result_df['total_ratings'] + 1)

        # Медиана рейтинга (приблизительная)
        def calculate_median_rating(row):
            total = row['total_ratings']
            if total == 0:
                return 0
            cumsum = 0
            median_pos = total / 2
            for i, col in enumerate(rating_cols, 1):
                cumsum += row[col]
                if cumsum >= median_pos:
                    return i
            return 5

        result_df['median_rating'] = result_df.apply(calculate_median_rating, axis=1)

        # ========================================
        # 2. ПРИЗНАКИ ИЗ КОММЕНТАРИЕВ И МЕДИА
        # ========================================

        # Заполняем пропуски
        media_cols = ['comments_published_count', 'photos_published_count', 'videos_published_count']
        for col in media_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Общее количество медиа контента
        result_df['total_media'] = (
                result_df['photos_published_count'] +
                result_df['videos_published_count']
        )

        # Соотношение комментариев к рейтингам
        result_df['comments_per_rating'] = (
                result_df['comments_published_count'] /
                (result_df['total_ratings'] + 1)
        )

        # Соотношение медиа к рейтингам
        result_df['media_per_rating'] = (
                result_df['total_media'] /
                (result_df['total_ratings'] + 1)
        )

        # Доля отзывов с фото
        result_df['photo_review_ratio'] = (
                result_df['photos_published_count'] /
                (result_df['total_ratings'] + 1)
        )

        # Доля отзывов с видео
        result_df['video_review_ratio'] = (
                result_df['videos_published_count'] /
                (result_df['total_ratings'] + 1)
        )

        # Флаг активности в комментариях
        result_df['high_comment_activity'] = (result_df['comments_per_rating'] > 0.5).astype(int)
        result_df['no_comments_flag'] = (result_df['comments_published_count'] == 0).astype(int)

        # ========================================
        # 3. ПРИЗНАКИ ИЗ ПРОДАЖ И ВОЗВРАТОВ
        # ========================================

        # Продажи
        sales_cols = ['item_count_sales7', 'item_count_sales30', 'item_count_sales90']
        for col in sales_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Динамика продаж
        result_df['sales_velocity_7d'] = result_df['item_count_sales7'] / 7
        result_df['sales_velocity_30d'] = result_df['item_count_sales30'] / 30
        result_df['sales_velocity_90d'] = result_df['item_count_sales90'] / 90

        # Ускорение продаж
        result_df['sales_acceleration'] = (
                result_df['sales_velocity_7d'] - result_df['sales_velocity_30d']
        )

        # Стабильность продаж
        result_df['sales_stability'] = 1 - abs(
            result_df['sales_velocity_7d'] - result_df['sales_velocity_30d']
        ) / (result_df['sales_velocity_30d'] + 1)

        # Возвраты
        returns_cols = ['item_count_returns7', 'item_count_returns30', 'item_count_returns90']
        for col in returns_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Процент возвратов
        result_df['return_rate_7d'] = (
                result_df['item_count_returns7'] /
                (result_df['item_count_sales7'] + 1)
        )
        result_df['return_rate_30d'] = (
                result_df['item_count_returns30'] /
                (result_df['item_count_sales30'] + 1)
        )
        result_df['return_rate_90d'] = (
                result_df['item_count_returns90'] /
                (result_df['item_count_sales90'] + 1)
        )

        # Динамика возвратов
        result_df['return_acceleration'] = (
                result_df['return_rate_7d'] - result_df['return_rate_30d']
        )

        # Фейковые возвраты (очень подозрительно!)
        fake_returns_cols = ['item_count_fake_returns7', 'item_count_fake_returns30', 'item_count_fake_returns90']
        for col in fake_returns_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Доля фейковых возвратов
        result_df['fake_return_rate_7d'] = (
                result_df['item_count_fake_returns7'] /
                (result_df['item_count_returns7'] + 1)
        )
        result_df['fake_return_rate_30d'] = (
                result_df['item_count_fake_returns30'] /
                (result_df['item_count_returns30'] + 1)
        )

        # Флаг наличия фейковых возвратов
        result_df['has_fake_returns'] = (
                (result_df['item_count_fake_returns7'] > 0) |
                (result_df['item_count_fake_returns30'] > 0) |
                (result_df['item_count_fake_returns90'] > 0)
        ).astype(int)

        # ========================================
        # 4. ФИНАНСОВЫЕ МЕТРИКИ
        # ========================================

        # GMV (Gross Merchandise Value)
        gmv_cols = ['GmvTotal7', 'GmvTotal30', 'GmvTotal90']
        for col in gmv_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Средний чек
        result_df['avg_order_value_7d'] = (
                result_df['GmvTotal7'] /
                (result_df['item_count_sales7'] + 1)
        )
        result_df['avg_order_value_30d'] = (
                result_df['GmvTotal30'] /
                (result_df['item_count_sales30'] + 1)
        )
        result_df['avg_order_value_90d'] = (
                result_df['GmvTotal90'] /
                (result_df['item_count_sales90'] + 1)
        )

        # Изменение среднего чека
        result_df['avg_order_value_change'] = (
                                                      result_df['avg_order_value_7d'] - result_df['avg_order_value_30d']
                                              ) / (result_df['avg_order_value_30d'] + 1)

        # Соотношение цены и среднего чека
        if 'PriceDiscounted' in result_df.columns:
            result_df['PriceDiscounted'] = result_df['PriceDiscounted'].fillna(0)
            result_df['price_to_avg_order_ratio'] = (
                    result_df['PriceDiscounted'] /
                    (result_df['avg_order_value_30d'] + 1)
            )

            # Ценовой сегмент (логарифмическая шкала)
            result_df['price_segment'] = np.log1p(result_df['PriceDiscounted'])

            # Флаги ценовых аномалий
            price_median = result_df['PriceDiscounted'].median()
            result_df['low_price_flag'] = (result_df['PriceDiscounted'] < price_median * 0.3).astype(int)
            result_df['high_price_flag'] = (result_df['PriceDiscounted'] > price_median * 3).astype(int)

        # Стоимость возвратов
        return_value_cols = ['ExemplarReturnedValueTotal7', 'ExemplarReturnedValueTotal30',
                             'ExemplarReturnedValueTotal90']
        for col in return_value_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Доля возвратов от GMV
        result_df['return_value_ratio_7d'] = (
                result_df['ExemplarReturnedValueTotal7'] /
                (result_df['GmvTotal7'] + 1)
        )
        result_df['return_value_ratio_30d'] = (
                result_df['ExemplarReturnedValueTotal30'] /
                (result_df['GmvTotal30'] + 1)
        )

        # ========================================
        # 5. ПРИЗНАКИ АКТИВНОСТИ И ВРЕМЕНИ
        # ========================================

        # Возраст товара
        if 'item_time_alive' in result_df.columns:
            result_df['item_time_alive'] = result_df['item_time_alive'].fillna(0)
            result_df['item_age_months'] = result_df['item_time_alive'] / 30
            result_df['item_age_log'] = np.log1p(result_df['item_time_alive'])

            # Категории по возрасту товара
            result_df['is_new_item'] = (result_df['item_time_alive'] < 30).astype(int)
            result_df['is_old_item'] = (result_df['item_time_alive'] > 365).astype(int)

            # Продажи на день жизни товара
            result_df['sales_per_day_alive'] = (
                    result_df['item_count_sales90'] /
                    (result_df['item_time_alive'] + 1)
            )

            # Рейтинги на день жизни
            result_df['ratings_per_day_alive'] = (
                    result_df['total_ratings'] /
                    (result_df['item_time_alive'] + 1)
            )

        # Возраст продавца
        if 'seller_time_alive' in result_df.columns:
            result_df['seller_time_alive'] = result_df['seller_time_alive'].fillna(0)
            result_df['seller_age_months'] = result_df['seller_time_alive'] / 30
            result_df['seller_age_years'] = result_df['seller_time_alive'] / 365
            result_df['seller_age_log'] = np.log1p(result_df['seller_time_alive'])

            # Категории продавцов
            result_df['is_new_seller'] = (result_df['seller_time_alive'] < 30).astype(int)
            result_df['is_young_seller'] = (result_df['seller_time_alive'] < 90).astype(int)
            result_df['is_mature_seller'] = (result_df['seller_time_alive'] > 365).astype(int)

            # Соотношение возраста товара и продавца
            result_df['item_seller_age_ratio'] = (
                    result_df['item_time_alive'] /
                    (result_df['seller_time_alive'] + 1)
            )

        # ========================================
        # 6. ПРИЗНАКИ ЗАКАЗОВ И ПРИНЯТИЯ
        # ========================================

        # Принятые экземпляры
        accepted_cols = ['ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30', 'ExemplarAcceptedCountTotal90']
        for col in accepted_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Принятые заказы
        order_cols = ['OrderAcceptedCountTotal7', 'OrderAcceptedCountTotal30', 'OrderAcceptedCountTotal90']
        for col in order_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)

        # Среднее количество товаров в заказе
        result_df['items_per_order_7d'] = (
                result_df['ExemplarAcceptedCountTotal7'] /
                (result_df['OrderAcceptedCountTotal7'] + 1)
        )
        result_df['items_per_order_30d'] = (
                result_df['ExemplarAcceptedCountTotal30'] /
                (result_df['OrderAcceptedCountTotal30'] + 1)
        )

        # Процент принятия (acceptance rate)
        result_df['acceptance_rate_7d'] = (
                result_df['ExemplarAcceptedCountTotal7'] /
                (result_df['item_count_sales7'] + result_df['ExemplarAcceptedCountTotal7'] + 1)
        )

        # ========================================
        # 7. ПРИЗНАКИ ИЗ ТЕКСТОВЫХ ПОЛЕЙ
        # ========================================

        # Обработка brand_name
        if 'brand_name' in result_df.columns:
            result_df['brand_clean'] = result_df['brand_name'].fillna('').astype(str).str.strip()
            result_df['brand_length'] = result_df['brand_clean'].str.len()
            result_df['brand_word_count'] = result_df['brand_clean'].str.split().str.len()
            result_df['brand_has_numbers'] = result_df['brand_clean'].str.contains(r'\d', regex=True).astype(int)
            result_df['brand_all_caps'] = (
                    (result_df['brand_clean'].str.isupper()) &
                    (result_df['brand_length'] > 2)
            ).astype(int)
            result_df['has_brand'] = (result_df['brand_length'] > 0).astype(int)

            # Подозрительные паттерны в бренде (включая ACTRUM из примера)
            suspicious_patterns = ['ACTRUM', 'COPY', 'REPLICA', 'TYPE', 'STYLE', 'LIKE',
                                   'АНАЛОГ', 'КОПИЯ', 'ТИП', 'ПОДОБНЫЙ']
            pattern = '|'.join(suspicious_patterns)
            result_df['suspicious_brand_pattern'] = (
                result_df['brand_clean'].str.upper().str.contains(pattern, regex=True, na=False)
            ).astype(int)

        # Обработка name_rus
        if 'name_rus' in result_df.columns:
            result_df['name_clean'] = result_df['name_rus'].fillna('').astype(str).str.strip()
            result_df['name_length'] = result_df['name_clean'].str.len()
            result_df['name_word_count'] = result_df['name_clean'].str.split().str.len()
            result_df['name_has_dots'] = result_df['name_clean'].str.contains('\.\.\.', regex=True).astype(int)
            result_df['has_name'] = (result_df['name_length'] > 0).astype(int)

            # Обрезанное название (подозрительно)
            result_df['name_truncated'] = result_df['name_clean'].str.endswith('...').astype(int)

        # Обработка категории
        if 'CommercialTypeName4' in result_df.columns:
            result_df['category_clean'] = result_df['CommercialTypeName4'].fillna('').astype(str).str.strip()
            result_df['category_length'] = result_df['category_clean'].str.len()
            result_df['has_category'] = (result_df['category_length'] > 0).astype(int)

        # Соответствие бренда и названия
        if 'brand_name' in result_df.columns and 'name_rus' in result_df.columns:
            def check_brand_in_name(row):
                brand = str(row.get('brand_clean', '')).lower()
                name = str(row.get('name_clean', '')).lower()
                if brand and name and len(brand) > 2:
                    return 1 if brand in name else 0
                return 0

            result_df['brand_name_match'] = result_df.apply(check_brand_in_name, axis=1)

        # ========================================
        # 8. ПРИЗНАКИ ИНВЕНТАРЯ
        # ========================================

        # Заполняем пропуски
        inventory_cols = ['ItemVarietyCount', 'ItemAvailableCount']
        for col in inventory_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(1)

        # Соотношение доступности к разнообразию
        result_df['availability_ratio'] = (
                result_df['ItemAvailableCount'] /
                (result_df['ItemVarietyCount'] + 1)
        )

        # Флаги подозрительных паттернов
        result_df['single_item_seller'] = (result_df['ItemVarietyCount'] == 1).astype(int)
        result_df['low_variety_flag'] = (result_df['ItemVarietyCount'] < 5).astype(int)

        # ========================================
        # 9. ПЕРЕКРЕСТНЫЕ ПРИЗНАКИ
        # ========================================

        # Соотношение рейтингов к продажам
        result_df['ratings_per_sale'] = (
                result_df['total_ratings'] /
                (result_df['item_count_sales90'] + 1)
        )

        # Соотношение комментариев к продажам
        result_df['comments_per_sale'] = (
                result_df['comments_published_count'] /
                (result_df['item_count_sales90'] + 1)
        )

        # Соотношение возвратов к рейтингам
        result_df['returns_per_rating'] = (
                result_df['item_count_returns30'] /
                (result_df['total_ratings'] + 1)
        )

        # Несоответствие рейтингов и возвратов
        def calculate_return_rating_anomaly(row):
            if row['avg_rating'] > 0 and row['total_ratings'] > 0:
                # Эмпирическая формула ожидаемых возвратов
                expected_returns = max(0, 5 - row['avg_rating']) * row['total_ratings'] * 0.05
                actual_returns = row['item_count_returns30']
                return abs(actual_returns - expected_returns) / (expected_returns + 1)
            return 0

        result_df['return_rating_anomaly'] = result_df.apply(calculate_return_rating_anomaly, axis=1)

        # Новый продавец с большим ассортиментом
        result_df['new_seller_high_variety'] = (
                result_df['is_new_seller'] *
                np.minimum(result_df['ItemVarietyCount'] / 10, 1)
        )

        # Новый товар со многими отзывами (подозрительно)
        result_df['new_item_many_reviews'] = (
                result_df['is_new_item'] *
                np.minimum(result_df['total_ratings'] / 50, 1)
        )

        # Соотношение фейковых возвратов к общим возвратам
        result_df['fake_to_total_returns'] = (
                result_df['item_count_fake_returns30'] /
                (result_df['item_count_returns30'] + 1)
        )

        # ========================================
        # 10. КОМПЛЕКСНЫЕ ПОКАЗАТЕЛИ РИСКА
        # ========================================

        # Базовый risk score
        result_df['risk_score'] = (
                result_df['has_fake_returns'] * 5 +  # Фейковые возвраты - самый сильный сигнал
                result_df.get('suspicious_brand_pattern', 0) * 3 +
                result_df['is_new_seller'] * 1.5 +
                result_df['rating_u_shape'] * 2 +
                result_df['single_item_seller'] * 1 +
                (result_df['return_rate_30d'] > 0.3).astype(int) * 2 +
                result_df['new_item_many_reviews'] * 1.5 +
                (result_df['fake_to_total_returns'] > 0.1).astype(int) * 3
        )

        # Нормализованный risk score (0-1)
        max_risk = result_df['risk_score'].max()
        if max_risk > 0:
            result_df['risk_score_normalized'] = result_df['risk_score'] / max_risk
        else:
            result_df['risk_score_normalized'] = 0

        # Аномальность по возвратам
        result_df['return_anomaly_score'] = (
                result_df['return_rate_30d'] * 5 +
                result_df['fake_to_total_returns'] * 10 +
                result_df['return_acceleration'] * 2
        )

        # Аномальность по рейтингам
        result_df['rating_anomaly_score'] = (
                result_df['rating_u_shape'] * 3 +
                result_df['rating_polarization'] * 2 +
                abs(result_df['avg_rating'] - result_df['median_rating']) +
                (1 - result_df['rating_entropy'] / (result_df['rating_entropy'].max() + 1))
        )

        # Аномальность продавца
        result_df['seller_anomaly_score'] = (
                result_df['is_new_seller'] * 2 +
                result_df['new_seller_high_variety'] * 3 +
                result_df['single_item_seller'] * 1.5 +
                (result_df['item_seller_age_ratio'] > 2).astype(int) * 1  # Товар старше продавца
        )

        # Аномальность продаж
        result_df['sales_anomaly_score'] = (
                abs(result_df['sales_acceleration']) * 2 +
                (result_df['ratings_per_sale'] > 2).astype(int) * 3 +  # Слишком много отзывов на продажу
                (result_df['avg_order_value_change'] > 0.5).astype(int) * 2
        )

        # Общий показатель аномальности
        result_df['total_anomaly_score'] = (
                result_df['return_anomaly_score'] * 0.3 +
                result_df['rating_anomaly_score'] * 0.25 +
                result_df['seller_anomaly_score'] * 0.25 +
                result_df['sales_anomaly_score'] * 0.2
        )

        # Специальный индикатор для контрафакта
        result_df['counterfeit_indicator'] = (
                result_df['has_fake_returns'] +
                result_df.get('suspicious_brand_pattern', 0) +
                (result_df['return_rate_30d'] > 0.2).astype(int) +
                (result_df['rating_u_shape'] > 0.3).astype(int) +
                (result_df['fake_to_total_returns'] > 0.05).astype(int) +
                result_df['new_seller_high_variety'] +
                result_df['new_item_many_reviews']
        )

        # ========================================
        # 11. СТАТИСТИЧЕСКИЕ АГРЕГАТЫ
        # ========================================

        # Процент заполненности данных
        text_fields = ['brand_name', 'name_rus', 'CommercialTypeName4']
        filled_count = 0
        for field in text_fields:
            if field in result_df.columns:
                filled_count += (result_df[field].notna() & (result_df[field] != '')).astype(int)

        result_df['data_completeness'] = filled_count / len(text_fields)

        # Активность товара (композитный показатель)
        result_df['item_activity_score'] = (
                result_df['sales_velocity_30d'] * 0.4 +
                result_df['ratings_per_day_alive'] * 100 * 0.3 +
                result_df['comments_per_sale'] * 10 * 0.3
        )

        # Качество товара (композитный показатель)
        result_df['quality_score'] = (
                result_df['avg_rating'] / 5 * 0.4 +
                (1 - result_df['return_rate_30d']) * 0.4 +
                (1 - result_df['fake_to_total_returns']) * 0.2
        )

        # ========================================
        # ФИНАЛЬНАЯ ОЧИСТКА
        # ========================================

        # Удаляем временные колонки
        temp_cols = ['brand_clean', 'name_clean', 'category_clean']
        for col in temp_cols:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])

        # Заменяем inf на NaN, затем на 0
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # Заполняем оставшиеся NaN нулями для числовых колонок
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_cols] = result_df[numeric_cols].fillna(0)

        # Подсчет новых признаков
        original_cols = df.columns.tolist()
        new_cols = [col for col in result_df.columns if col not in original_cols]
        # Вывод топ важных признаков
        print("\n📊 Key features for counterfeit detection:")
        important_features = [
            'counterfeit_indicator', 'risk_score', 'total_anomaly_score',
            'has_fake_returns', 'fake_to_total_returns', 'return_rating_anomaly',
            'suspicious_brand_pattern', 'rating_u_shape', 'return_rate_30d',
            'new_seller_high_variety', 'new_item_many_reviews', 'quality_score'
        ]

        for feat in important_features:
            if feat in result_df.columns:
                print(f"  ✓ {feat}")

        return result_df