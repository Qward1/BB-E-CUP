"""
Text Pipeline: препроцессинг + модель
"""

import re
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextPipeline:
    """Полный pipeline для текстовых данных"""

    def __init__(self, config: dict, use_transformer: bool = True):
        self.config = config
        self.use_transformer = use_transformer

        if use_transformer:
            model_name = config['text_model']['name']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        else:
            # Fallback на TF-IDF для быстрого baseline
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.model = LogisticRegression(class_weight='balanced')

    def preprocess(self, texts: pd.Series) -> pd.Series:
        """Предобработка текстов"""
        # Заполнение пропусков
        texts = texts.fillna('')

        # Очистка
        texts = texts.apply(self._clean_text)

        return texts

    def _clean_text(self, text: str) -> str:
        """Очистка одного текста"""
        if not text:
            return ''

        # Удаление HTML тегов
        text = re.sub(r'<[^>]+>', ' ', text)

        # Lowercase
        text = text.lower()

        # Удаление лишних пробелов
        text = ' '.join(text.split())

        return text

    def extract_features(self, texts: pd.Series) -> dict:
        """Извлечение дополнительных признаков из текста"""
        features = {
            'text_length': texts.str.len(),
            'word_count': texts.str.split().str.len(),
            'has_brand_typos': texts.apply(self._check_brand_typos),
            'excessive_caps': texts.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        }
        return pd.DataFrame(features)

    def _check_brand_typos(self, text: str) -> int:
        """Проверка на опечатки в брендах"""
        suspicious_patterns = [
            r'n[i1]ke', r'ad[i1]das', r'pu[mn]a',  # Опечатки в брендах
            r'0riginal', r'kachestvo',  # Подозрительные слова
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text.lower()):
                return 1
        return 0

    def fit(self, texts: pd.Series, labels: pd.Series):
        """Обучение модели"""
        texts = self.preprocess(texts)

        if self.use_transformer:
            # Fine-tuning трансформера (упрощенная версия)
            print("Fine-tuning transformer model...")
            # Здесь должен быть код fine-tuning
            pass
        else:
            # TF-IDF + LogisticRegression
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Предсказание вероятностей"""
        texts = self.preprocess(texts)

        if self.use_transformer:
            return self._predict_transformer(texts)
        else:
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)[:, 1]

    def _predict_transformer(self, texts: pd.Series) -> np.ndarray:
        """Предсказание с помощью трансформера"""
        self.model.eval()
        all_probs = []
        batch_size = self.config['text_model']['batch_size']

        for i in range(0, len(texts), batch_size):
            batch = texts.iloc[i:i + batch_size].tolist()

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config['text_model']['max_length'],
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs[:, 1].cpu().numpy())

        return np.concatenate(all_probs)