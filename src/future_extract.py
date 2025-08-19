"""
Специализированный экстрактор признаков для e-commerce описаний
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from collections import Counter
import Levenshtein


class EcommerceFeatureExtractor:
    """
    Извлечение специфичных для e-commerce признаков
    """

    def __init__(self):
        # Паттерны для поиска

        self.price_pattern = re.compile(r'\d+\s*(?:руб|рубл|₽|р\.|dollars?|\$|евро|€)')

        # Ключевые слова
        self.copy_keywords = [
            'копия', 'копию', 'аналог', 'заменитель', 'совместим',
            'подходит', 'альтернатива', 'replica', 'copy', 'compatible',
            'replacement', 'substitute', 'как оригинал', 'точная копия'
        ]



    def extract_features(self, text: str, brand_name: Optional[str] = None,
                         category: Optional[str] = None) -> Dict[str, float]:
        """
        Извлечение всех признаков из описания

        Args:
            text: Описание товара
            brand_name: Бренд из метаданных
            category: Категория товара

        Returns:
            Словарь признаков
        """
        features = {}

        # Базовая обработка
        text_clean = self._clean_html(text)
        text_lower = text_clean.lower()

        # 1. HTML анализ
        features.update(self._extract_html_features(text))

        # 2. Модели и совместимость


        # 3. Размеры и спецификации


        # 4. Ключевые слова и паттерны
        features.update(self._extract_keyword_features(text_lower))

        # 5. Стилистические особенности
        features.update(self._extract_style_features(text_clean))

        # 6. Профессиональная терминология


        # 7. Повторения и аномалии
        features.update(self._extract_repetition_features(text_clean))

        return features

    def _clean_html(self, text: str) -> str:
        """Удаление HTML тегов"""
        return re.sub(r'<[^>]+>', ' ', str(text))

    def _extract_html_features(self, text: str) -> Dict[str, float]:
        """Анализ HTML структуры - расширенная версия для выявления контрафакта"""
        features = {}

        # Базовые структурные теги
        br_tags = len(re.findall(r'<br\s*/?>', str(text), re.IGNORECASE))
        p_tags = len(re.findall(r'<p[^>]*>', str(text), re.IGNORECASE))
        div_tags = len(re.findall(r'<div[^>]*>', str(text), re.IGNORECASE))
        span_tags = len(re.findall(r'<span[^>]*>', str(text), re.IGNORECASE))

        # Теги форматирования (контрафакт часто злоупотребляет)
        strong_tags = len(re.findall(r'<(strong|b)\b[^>]*>', str(text), re.IGNORECASE))
        italic_tags = len(re.findall(r'<(em|i)\b[^>]*>', str(text), re.IGNORECASE))
        underline_tags = len(re.findall(r'<u\b[^>]*>', str(text), re.IGNORECASE))

        # Цветовые и стилевые теги (часто в контрафакте для привлечения внимания)
        font_tags = len(re.findall(r'<font[^>]*>', str(text), re.IGNORECASE))
        color_attrs = len(re.findall(r'(?:color|style)\s*=\s*["\'][^"\']*(?:red|green|blue|yellow|#[0-9a-f]{6})', str(text),
                                     re.IGNORECASE))
        style_attrs = len(re.findall(r'style\s*=\s*["\'][^"\']*["\']', str(text), re.IGNORECASE))

        # Заголовки (H1-H6)
        h_tags = len(re.findall(r'<h[1-6]\b[^>]*>', str(text), re.IGNORECASE))
        h1_tags = len(re.findall(r'<h1\b[^>]*>', str(text), re.IGNORECASE))  # H1 особенно подозрительно в описании





        # Специальные символы HTML entities
        html_entities = len(re.findall(r'&(?:#\d+|#x[0-9a-f]+|\w+);', str(text), re.IGNORECASE))
        nbsp_count = str(text).count('&nbsp;')

        # Скрипты и стили (очень подозрительно)
        script_tags = len(re.findall(r'<script\b[^>]*>', str(text), re.IGNORECASE))
        style_tags = len(re.findall(r'<style\b[^>]*>', str(text), re.IGNORECASE))

        # Невалидные/сломанные теги (признак копипаста)
        broken_tags = len(re.findall(r'<[^>]*<|>[^<]*>', str(text)))  # Незакрытые или двойные
        mismatched_tags = self._count_mismatched_tags(str(text))

        # Формирование признаков
        features['has_html_tags'] = 1 if '<' in str(text) else 0
        features['html_total_tags'] = sum([
            br_tags, p_tags, div_tags, span_tags, strong_tags, italic_tags,
            underline_tags, font_tags, h_tags,

        ])

        # Структурные теги
        features['html_br_count'] = br_tags
        features['html_p_count'] = p_tags
        features['html_div_count'] = div_tags
        features['html_span_count'] = span_tags

        # Форматирование (контрафакт часто переусердствует)
        features['html_formatting_tags'] = strong_tags + italic_tags + underline_tags
        features['html_strong_tags'] = strong_tags
        features['html_italic_tags'] = italic_tags
        features['html_underline_tags'] = underline_tags
        features['excessive_formatting'] = 1 if features['html_formatting_tags'] > 5 else 0

        # Стилизация (подозрительно)
        features['html_font_tags'] = font_tags  # Устаревший тег
        features['html_color_usage'] = color_attrs
        features['html_style_attrs'] = style_attrs
        features['has_colored_text'] = 1 if color_attrs > 0 else 0

        # Заголовки
        features['html_header_tags'] = h_tags
        features['has_h1_tag'] = 1 if h1_tags > 0 else 0  # H1 в описании подозрительно

        # HTML entities
        features['html_entities_count'] = html_entities
        features['html_nbsp_count'] = nbsp_count
        features['excessive_nbsp'] = 1 if nbsp_count > 10 else 0

        # Качество HTML
        features['html_broken_tags'] = broken_tags
        features['html_mismatched_tags'] = mismatched_tags
        features['poor_html_quality'] = 1 if (broken_tags + mismatched_tags) > 0 else 0

        # Соотношения
        text_clean = self._clean_html(text)
        features['html_to_text_ratio'] = 1 - (len(text_clean) / max(len(str(text)), 1))
        features['formatting_density'] = features['html_formatting_tags'] / max(len(str(text).split()), 1)

        # Паттерны контрафакта
        features['html_complexity_score'] = self._calculate_html_complexity(features)
        features['html_suspicious_pattern'] = self._detect_suspicious_html_pattern(text)

        return features

    def _count_mismatched_tags(self, text: str) -> int:
        """Подсчет несоответствующих открывающих и закрывающих тегов"""
        # Простая проверка на парность основных тегов
        mismatched = 0
        tags_to_check = ['p', 'div', 'span', 'strong', 'b', 'em', 'i', 'u', 'ul', 'ol', 'li']

        for tag in tags_to_check:
            opening = len(re.findall(f'<{tag}\\b[^>]*>', str(text), re.IGNORECASE))
            closing = len(re.findall(f'</{tag}>', str(text), re.IGNORECASE))
            if opening != closing:
                mismatched += abs(opening - closing)

        return mismatched

    def _calculate_html_complexity(self, features: Dict[str, float]) -> float:
        """
        Расчет сложности HTML структуры
        Слишком простая или слишком сложная структура может указывать на контрафакт
        """
        complexity = 0

        # Факторы сложности
        complexity += features.get('html_total_tags', 0) * 0.1
        complexity += features.get('html_table_complexity', 0) * 0.5
        complexity += features.get('html_list_tags', 0) * 0.3
        complexity += features.get('html_style_attrs', 0) * 0.2

        # Нормализация от 0 до 10
        return min(complexity, 10)

    def _detect_suspicious_html_pattern(self, text: str) -> int:
        """
        Обнаружение подозрительных HTML паттернов
        """
        suspicious_score = 0

        # Множественные пустые <br/> подряд (часто в контрафакте)
        if re.search(r'(?:<br\s*/?>[\s]*){3,}', str(text), re.IGNORECASE):
            suspicious_score += 1

        # Чрезмерное использование bold/strong
        if re.search(r'(?:<(?:strong|b)>.*?</(?:strong|b)>[\s]*){3,}', str(text), re.IGNORECASE):
            suspicious_score += 1

        # Inline стили с яркими цветами (red, yellow, green)
        if re.search(r'style\s*=\s*["\'][^"\']*(?:red|yellow|#ff0000|#ffff00|#00ff00)', str(text), re.IGNORECASE):
            suspicious_score += 1

        # Множественные восклицательные знаки в HTML
        if re.search(r'<[^>]*>.*?!!!+.*?</[^>]*>', str(text)):
            suspicious_score += 1

        # Использование устаревших тегов
        deprecated_tags = ['font', 'center', 'marquee', 'blink']
        for tag in deprecated_tags:
            if re.search(f'<{tag}\\b', str(text), re.IGNORECASE):
                suspicious_score += 1

        # Скопированные стили из Word/других редакторов
        if 'mso-' in str(text) or 'MsoNormal' in str(text):
            suspicious_score += 2  # Явный признак копипаста

        # Пустые теги
        if re.search(r'<(?:p|div|span|strong|b|em|i)(?:\s[^>]*)?>[\s]*</(?:p|div|span|strong|b|em|i)>', str(text),
                     re.IGNORECASE):
            suspicious_score += 1

        return suspicious_score



    def _extract_keyword_features(self, text: str) -> Dict[str, float]:
        """Анализ ключевых слов"""
        features = {}

        # Слова-индикаторы копий
        copy_score = sum(1 for keyword in self.copy_keywords if keyword in str(text))
        features['copy_words_count'] = copy_score
        features['has_copy_words'] = 1 if copy_score > 0 else 0

        # Упоминание цены
        prices = self.price_pattern.findall(str(text))
        features['price_mentioned'] = 1 if prices else 0
        features['price_mentions_count'] = len(prices)

        # Призывы к действию
        cta_phrases = ['купи', 'закаж', 'спеши', 'скидк', 'акци', 'buy', 'order', 'hurry', 'discount']
        features['call_to_action_count'] = sum(1 for phrase in cta_phrases if phrase in str(text))

        return features

    def _extract_style_features(self, text: str) -> Dict[str, float]:
        """Стилистический анализ
        НУЖНО ДОРАБОТАТЬ АНАЛИЗ АБЗАЦЕВ"""
        features = {}

        # Анализ регистра
        if text:
            upper_chars = sum(1 for c in str(text) if c.isupper())
            lower_chars = sum(1 for c in str(text) if c.islower())
            total_alpha = upper_chars + lower_chars

            features['uppercase_ratio'] = upper_chars / max(total_alpha, 1)
            features['all_uppercase'] = 1 if upper_chars > 0 and lower_chars == 0 else 0
            features['all_lowercase'] = 1 if lower_chars > 0 and upper_chars == 0 else 0

            # Аномальные паттерны капитализации
            words = str(text).split()
            capitalized_words = sum(1 for w in words if w and w[0].isupper())
            features['capitalized_words_ratio'] = capitalized_words / max(len(words), 1)

        # Пунктуация
        punctuation = '!?.,:;-—'
        punct_count = sum(1 for c in str(text) if c in punctuation)
        features['punctuation_density'] = punct_count / max(len(str(text)), 1)
        features['exclamation_marks'] = str(text).count('!')
        features['question_marks'] = str(text).count('?')
        features['ellipsis_count'] = str(text).count('...')

        # Эмодзи и спецсимволы
        emoji_pattern = re.compile(r'[😀-🙏🌀-🗿🚀-🛿🏀-🏿]')
        features['has_emoji'] = 1 if emoji_pattern.search(str(text)) else 0

        # Форматирование абзацев
        paragraphs = str(text).split('\n\n')
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = np.mean([len(p) for p in paragraphs]) if paragraphs else 0

        return features




    def _extract_repetition_features(self, text: str) -> Dict[str, float]:
        """Анализ повторений и паттернов"""
        features = {}

        # Разбиваем на слова
        words = str(text).lower().split()

        if words:
            # Уникальность слов
            unique_words = set(words)
            features['word_uniqueness'] = len(unique_words) / len(words)

            # Повторяющиеся слова
            word_counts = Counter(words)
            repeated_words = [w for w, c in word_counts.items() if c > 2]
            features['repeated_words_count'] = len(repeated_words)

            # Повторяющиеся биграммы
            bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            repeated_bigrams = [b for b, c in bigram_counts.items() if c > 1]
            features['repeated_bigrams_count'] = len(repeated_bigrams)

            # Максимальное количество повторений одного слова
            max_repetition = max(word_counts.values()) if word_counts else 0
            features['max_word_repetition'] = max_repetition

            # Энтропия текста (мера разнообразия)
            total_words = len(words)
            entropy = -sum((count / total_words) * np.log2(count / total_words)
                           for count in word_counts.values())
            features['text_entropy'] = entropy

        # Повторяющиеся предложения
        sentences = re.split(r'[.!?]', str(text))
        sentence_counts = Counter(s.strip() for s in sentences if s.strip())
        repeated_sentences = sum(1 for c in sentence_counts.values() if c > 1)
        features['repeated_sentences'] = repeated_sentences

        return features