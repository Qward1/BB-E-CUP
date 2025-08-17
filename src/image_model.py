# src/image_model.py
"""
Image Pipeline: обработка изображений товаров
"""

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Optional, Union
import cv2
import pytesseract
from pathlib import Path


class ImagePipeline:
    """Pipeline для обработки изображений товаров"""

    def __init__(self, config: dict, use_pretrained: bool = True):
        self.config = config
        self.use_pretrained = use_pretrained
        self.device = torch.device('cpu')  # Принудительно CPU для теста

        # Препроцессинг для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if use_pretrained:
            # Используем легкую модель для CPU
            self.feature_extractor = models.mobilenet_v2(pretrained=True)
            self.feature_extractor.eval()
            # Убираем последний слой для получения эмбеддингов
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        # Простой классификатор
        self.classifier = None
        self.image_features_dim = 1280  # Для MobileNetV2

    def extract_visual_features(self, image_path: str) -> dict:
        """Извлечение визуальных признаков из изображения"""
        features = {}

        try:
            # Загрузка изображения
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            # 1. Базовые характеристики
            features['img_width'] = img.width
            features['img_height'] = img.height
            features['aspect_ratio'] = img.width / max(img.height, 1)

            # 2. Качество изображения
            features['image_quality'] = self._assess_image_quality(img_array)

            # 3. Цветовые характеристики
            color_features = self._extract_color_features(img_array)
            features.update(color_features)

            # 4. Детекция текста и логотипов
            features['has_text'] = self._detect_text(img_array)
            features['text_area_ratio'] = self._calculate_text_area_ratio(img_array)

            # 5. Детекция водяных знаков
            features['has_watermark'] = self._detect_watermark(img_array)

            # 6. Блюр и резкость
            features['blurriness'] = self._calculate_blurriness(img_array)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Возвращаем дефолтные значения
            features = {
                'img_width': 0, 'img_height': 0, 'aspect_ratio': 1,
                'image_quality': 0, 'has_text': 0, 'text_area_ratio': 0,
                'has_watermark': 0, 'blurriness': 0
            }

        return features

    def _assess_image_quality(self, img_array: np.ndarray) -> float:
        """Оценка качества изображения"""
        # Используем дисперсию лапласиана как меру резкости
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Нормализуем значение
        quality_score = min(laplacian_var / 1000, 1.0)
        return quality_score

    def _extract_color_features(self, img_array: np.ndarray) -> dict:
        """Извлечение цветовых характеристик"""
        features = {}

        # Средние значения по каналам
        features['mean_red'] = img_array[:, :, 0].mean() / 255
        features['mean_green'] = img_array[:, :, 1].mean() / 255
        features['mean_blue'] = img_array[:, :, 2].mean() / 255

        # Стандартное отклонение (разнообразие цветов)
        features['color_std'] = img_array.std() / 255

        # Доминирующий цвет
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        features['dominant_hue'] = hsv[:, :, 0].mean() / 180
        features['mean_saturation'] = hsv[:, :, 1].mean() / 255
        features['mean_brightness'] = hsv[:, :, 2].mean() / 255

        return features

    def _detect_text(self, img_array: np.ndarray) -> int:
        """Детекция текста на изображении"""
        try:
            # Простая детекция через OCR
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(gray, timeout=2)
            return 1 if len(text.strip()) > 10 else 0
        except:
            return 0

    def _calculate_text_area_ratio(self, img_array: np.ndarray) -> float:
        """Расчет доли площади с текстом"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Детекция краев (текст обычно имеет четкие края)
            edges = cv2.Canny(gray, 50, 150)

            # Морфологические операции для выделения текстовых областей
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Подсчет доли пикселей с текстом
            text_ratio = np.sum(dilated > 0) / dilated.size
            return min(text_ratio, 1.0)
        except:
            return 0.0

    def _detect_watermark(self, img_array: np.ndarray) -> int:
        """Детекция водяных знаков"""
        try:
            # Проверяем наличие полупрозрачных областей
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Применяем пороговую фильтрацию
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            # Если много очень светлых пикселей - возможно водяной знак
            white_ratio = np.sum(thresh == 255) / thresh.size

            return 1 if white_ratio > 0.3 else 0
        except:
            return 0

    def _calculate_blurriness(self, img_array: np.ndarray) -> float:
        """Расчет степени размытости"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Чем меньше дисперсия, тем более размыто изображение
        # Нормализуем обратно
        blurriness = 1.0 / (1.0 + variance / 100)
        return blurriness

    def extract_cnn_features(self, image_path: str) -> np.ndarray:
        """Извлечение CNN эмбеддингов"""
        if not self.use_pretrained:
            return np.zeros(self.image_features_dim)

        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                features = features.squeeze().numpy()

            return features
        except:
            return np.zeros(self.image_features_dim)

    def process_batch(self, image_paths: List[str]) -> pd.DataFrame:
        """Обработка батча изображений"""
        all_features = []

        for path in image_paths:
            # Визуальные признаки
            visual_features = self.extract_visual_features(path)

            # CNN признаки (опционально, так как медленно на CPU)
            if self.use_pretrained and len(image_paths) < 100:  # Ограничение для CPU
                cnn_features = self.extract_cnn_features(path)
                visual_features['cnn_mean'] = cnn_features.mean()
                visual_features['cnn_std'] = cnn_features.std()

            all_features.append(visual_features)

        return pd.DataFrame(all_features)

    def detect_counterfeit_patterns(self, image_path: str) -> dict:
        """Детекция паттернов контрафакта"""
        patterns = {}

        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            # 1. Низкое разрешение (часто у контрафакта)
            patterns['low_resolution'] = 1 if img.width < 500 or img.height < 500 else 0

            # 2. Искаженные пропорции логотипа
            patterns['distorted_aspect'] = 1 if abs(img.width / img.height - 1) > 2 else 0

            # 3. Избыточная компрессия JPEG
            patterns['high_compression'] = self._detect_jpeg_artifacts(img_array)

            # 4. Несоответствие цветов бренда
            patterns['color_deviation'] = self._check_brand_colors(img_array)

        except:
            patterns = {
                'low_resolution': 0,
                'distorted_aspect': 0,
                'high_compression': 0,
                'color_deviation': 0
            }

        return patterns

    def _detect_jpeg_artifacts(self, img_array: np.ndarray) -> int:
        """Детекция артефактов сжатия"""
        # Проверяем блочность изображения (8x8 блоки JPEG)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Вычисляем градиенты
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Если много резких переходов на границах блоков - артефакты сжатия
        edge_density = (np.abs(sobelx) + np.abs(sobely)).mean()

        return 1 if edge_density > 50 else 0

    def _check_brand_colors(self, img_array: np.ndarray) -> int:
        """Проверка соответствия цветов бренда"""
        # Здесь должна быть проверка на соответствие фирменным цветам
        # Для примера - проверяем наличие очень ярких неестественных цветов
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Высокая насыщенность + высокая яркость = подозрительно
        high_saturation = hsv[:, :, 1] > 200
        high_brightness = hsv[:, :, 2] > 200

        suspicious_pixels = np.sum(high_saturation & high_brightness)
        suspicious_ratio = suspicious_pixels / hsv[:, :, 0].size

        return 1 if suspicious_ratio > 0.1 else 0