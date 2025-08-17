"""
Скрипт обучения всех моделей
"""

import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import DataProcessor
from text_model import TextPipeline
from metadata_model import MetadataPipeline
from fusion import LateFusion


def train_pipeline(config_path: str = 'config.yaml'):
    """Полный цикл обучения"""

    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading data...")
    data_processor = DataProcessor(config)
    df = data_processor.load_train_data(config['paths']['train_data'])

    # Подготовка данных
    texts, metadata, labels = data_processor.prepare_features(df)

    # Обработка пропусков в метаданных
    metadata = data_processor.handle_missing(metadata)

    # Разделение на train/val
    X_text_train, X_text_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        texts, metadata, labels,
        test_size=0.2,
        stratify=labels,
        random_state=config['project']['seed']
    )

    # 1. Text Model
    print("Training text model...")
    text_pipeline = TextPipeline(config, use_transformer=False)  # Для скорости используем TF-IDF
    text_pipeline.fit(X_text_train, y_train)
    text_val_probs = text_pipeline.predict_proba(X_text_val)

    # 2. Metadata Model
    print("Training metadata model...")
    meta_pipeline = MetadataPipeline(config)
    meta_pipeline.fit(X_meta_train, y_train)
    meta_val_probs = meta_pipeline.predict_proba(X_meta_val)

    # 3. Fusion
    print("Optimizing fusion...")
    fusion = LateFusion(config)
    fusion.optimize(text_val_probs, meta_val_probs, y_val)

    # Финальная оценка
    final_preds = fusion.predict(text_val_probs, meta_val_probs)
    from sklearn.metrics import classification_report
    print("\nFinal results on validation:")
    print(classification_report(y_val, final_preds))

    # Сохранение моделей
    print("Saving models...")
    joblib.dump(text_pipeline, f"{config['paths']['models_dir']}/text_pipeline.pkl")
    joblib.dump(meta_pipeline, f"{config['paths']['models_dir']}/meta_pipeline.pkl")
    joblib.dump(fusion, f"{config['paths']['models_dir']}/fusion.pkl")
    joblib.dump(data_processor, f"{config['paths']['models_dir']}/data_processor.pkl")

    # Сохранение конфигурации
    joblib.dump(config, f"{config['paths']['models_dir']}/config.pkl")

    print("Training complete!")


if __name__ == "__main__":
    train_pipeline()