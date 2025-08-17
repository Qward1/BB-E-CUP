"""
Production inference - БЕЗ обучения
"""

import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def predict(test_data_path: str, output_path: str, models_dir: str = "models/"):
    """
    Главная функция для inference
    БЕЗ обучения - только использование обученных моделей
    """

    print("Loading models...")
    # Загрузка всех компонентов
    text_pipeline = joblib.load(f"{models_dir}/text_pipeline.pkl")
    meta_pipeline = joblib.load(f"{models_dir}/meta_pipeline.pkl")
    fusion = joblib.load(f"{models_dir}/fusion.pkl")
    data_processor = joblib.load(f"{models_dir}/data_processor.pkl")
    config = joblib.load(f"{models_dir}/config.pkl")

    print("Processing test data...")
    all_predictions = []
    all_ids = []

    # Обработка батчами
    for batch_idx, batch_df in enumerate(data_processor.load_test_batches(test_data_path)):
        print(f"Processing batch {batch_idx + 1}...")

        # IDs
        id_col = config['data']['id_column']
        if id_col in batch_df.columns:
            all_ids.extend(batch_df[id_col].values)
        else:
            all_ids.extend(range(len(batch_df)))

        # Подготовка данных
        texts, metadata, _ = data_processor.prepare_features(batch_df)
        metadata = data_processor.handle_missing(metadata)

        # Проверка доступности данных
        text_available = texts.str.len() > 0
        meta_available = ~metadata.isna().all(axis=1)

        # Предсказания
        text_probs = text_pipeline.predict_proba(texts)
        meta_probs = meta_pipeline.predict_proba(metadata)

        # Fusion
        batch_preds = fusion.predict(
            text_probs, meta_probs,
            text_available.values, meta_available.values
        )

        all_predictions.extend(batch_preds)

    # Сохранение результатов
    submission = pd.DataFrame({
        config['data']['id_column']: all_ids,
        config['data']['target_column']: all_predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"✅ Saved {len(submission)} predictions to {output_path}")

    return submission


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python predict.py <test_data_path> <output_path>")
        sys.exit(1)

    predict(sys.argv[1], sys.argv[2])