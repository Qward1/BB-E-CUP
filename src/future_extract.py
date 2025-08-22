"""
Оптимизированный экстрактор признаков для e-commerce описаний
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Optional, Set, List
from collections import Counter
from functools import lru_cache
import unicodedata
import string


class OptimizedEcommerceExtractor:
    """
    Оптимизированное извлечение признаков для e-commerce
    """

    def __init__(self):
        # Предкомпилированные паттерны (компилируем один раз)
        self._compile_patterns()

        # Загружаем внешние данные
        self._load_external_data()

        # Кэш для часто используемых операций
        self._cache = {}

    def _compile_patterns(self):
        """Предкомпиляция всех regex паттернов для производительности"""
        # HTML паттерны
        self.patterns = {
            # Структурные теги
            'br_tags': re.compile(r'<br\s*/?>', re.IGNORECASE),
            'p_tags': re.compile(r'<p[^>]*>', re.IGNORECASE),
            'div_tags': re.compile(r'<div[^>]*>', re.IGNORECASE),
            'span_tags': re.compile(r'<span[^>]*>', re.IGNORECASE),

            # Форматирование
            'strong_tags': re.compile(r'<(?:strong|b)\b[^>]*>', re.IGNORECASE),
            'italic_tags': re.compile(r'<(?:em|i)\b[^>]*>', re.IGNORECASE),
            'underline_tags': re.compile(r'<u\b[^>]*>', re.IGNORECASE),

            # Стили
            'font_tags': re.compile(r'<font[^>]*>', re.IGNORECASE),
            'color_attrs': re.compile(r'(?:color|style)\s*=\s*["\'][^"\']*(?:red|green|blue|yellow|#[0-9a-f]{6})', re.IGNORECASE),
            'style_attrs': re.compile(r'style\s*=\s*["\'][^"\']*["\']', re.IGNORECASE),

            # Заголовки
            'h_tags': re.compile(r'<h[1-6]\b[^>]*>', re.IGNORECASE),
            'h1_tags': re.compile(r'<h1\b[^>]*>', re.IGNORECASE),

            # Entities и прочее
            'html_entities': re.compile(r'&(?:#\d+|#x[0-9a-f]+|\w+);', re.IGNORECASE),
            'script_tags': re.compile(r'<script\b[^>]*>', re.IGNORECASE),
            'style_tags': re.compile(r'<style\b[^>]*>', re.IGNORECASE),
            'broken_tags': re.compile(r'<[^>]*<|>[^<]*>'),

            # Подозрительные паттерны
            'multiple_br': re.compile(r'(?:<br\s*/?>[\s]*){3,}', re.IGNORECASE),
            'multiple_bold': re.compile(r'(?:<(?:strong|b)>.*?</(?:strong|b)>[\s]*){3,}', re.IGNORECASE),
            'bright_colors': re.compile(r'style\s*=\s*["\'][^"\']*(?:red|yellow|#ff0000|#ffff00|#00ff00)', re.IGNORECASE),
            'multiple_exclamation': re.compile(r'<[^>]*>.*?!!!+.*?</[^>]*>'),
            'empty_tags': re.compile(r'<(?:p|div|span|strong|b|em|i)(?:\s[^>]*)?>[\s]*</(?:p|div|span|strong|b|em|i)>', re.IGNORECASE),

            # Общие паттерны
            'price': re.compile(r'\d+\s*(?:руб|рубл|₽|р\.|dollars?|\$|евро|€)'),
            'clean_html': re.compile(r'<[^>]+>'),
            'sentences': re.compile(r'[.!?]+'),
        }

        # Устаревшие теги (множество для быстрой проверки)
        self.deprecated_tags = frozenset(['font', 'center', 'marquee', 'blink'])

    def _load_external_data(self):
        """Загрузка внешних данных и словарей"""
        # Ключевые слова копий
        self.copy_keywords = frozenset([
            'копия', 'копию', 'аналог', 'заменитель', 'совместим',
            'подходит', 'альтернатива', 'replica', 'copy', 'compatible',
            'replacement', 'substitute', 'как оригинал', 'точная копия'
        ])

        # Призывы к действию
        self.cta_phrases = frozenset([
            'купи', 'закаж', 'спеши', 'скидк', 'акци',
            'buy', 'order', 'hurry', 'discount'
        ])

        # Загрузка эмодзи из Unicode категорий
        self._load_emoji_ranges()

    def _load_emoji_ranges(self):
        """Загрузка диапазонов эмодзи из Unicode стандарта"""
        # Используем стандартные Unicode категории для эмодзи
        # Это более надежно, чем хардкодить диапазоны
        self.emoji_categories = {
            'Emoticons': (0x1F600, 0x1F64F),
            'Miscellaneous Symbols': (0x1F300, 0x1F5FF),
            'Transport and Map': (0x1F680, 0x1F6FF),
            'Supplemental Symbols': (0x1F900, 0x1F9FF),
        }

    @lru_cache(maxsize=1024)
    def _has_emoji(self, text: str) -> bool:
        """Проверка наличия эмодзи (с кэшированием)"""
        for char in text:
            # Проверяем Unicode категорию символа
            if unicodedata.category(char) in ('So', 'Sk'):  # Symbol, Other / Symbol, Modifier
                code = ord(char)
                for start, end in self.emoji_categories.values():
                    if start <= code <= end:
                        return True
        return False

    @lru_cache(maxsize=512)
    def _clean_html_cached(self, text: str) -> str:
        """Кэшированная очистка HTML"""
        return self.patterns['clean_html'].sub(' ', text)

    def extract_features(self, text: str, brand_name: Optional[str] = None,
                         category: Optional[str] = None) -> Dict[str, float]:
        """
        Оптимизированное извлечение признаков
        """
        # Конвертируем в строку один раз
        text_str = str(text) if pd.notna(text) else ''

        if not text_str:
            return self._get_empty_features()

        # Базовая обработка (кэшируем)
        text_clean = self._clean_html_cached(text_str)
        text_lower = text_clean.lower()

        # Параллельное извлечение признаков
        features = {}

        # Используем numpy для векторизованных операций где возможно
        features.update(self._extract_html_features_optimized(text_str))
        features.update(self._extract_keyword_features_optimized(text_lower))
        features.update(self._extract_style_features_optimized(text_clean))
        features.update(self._extract_repetition_features_optimized(text_clean))

        return features

    def _get_empty_features(self) -> Dict[str, float]:
        """Возвращает словарь с нулевыми значениями для пустого текста"""
        return {
            # HTML features
            'has_html_tags': 0, 'html_total_tags': 0, 'html_br_count': 0,
            'html_formatting_tags': 0, 'excessive_formatting': 0,
            'html_suspicious_pattern': 0, 'poor_html_quality': 0,

            # Keyword features
            'copy_words_count': 0, 'has_copy_words': 0,
            'price_mentioned': 0, 'call_to_action_count': 0,

            # Style features
            'uppercase_ratio': 0, 'punctuation_density': 0,
            'exclamation_marks': 0, 'has_emoji': 0,

            # Repetition features
            'word_uniqueness': 0, 'repeated_words_count': 0,
            'text_entropy': 0, 'repeated_sentences': 0
        }

    def _extract_html_features_optimized(self, text: str) -> Dict[str, float]:
        """Оптимизированное извлечение HTML признаков"""
        features = {}

        # Используем предкомпилированные паттерны
        # Считаем все за один проход
        tag_counts = {}
        for name, pattern in self.patterns.items():
            if name.endswith('_tags') or name.endswith('_attrs'):
                tag_counts[name] = len(pattern.findall(text))

        # Базовые признаки
        features['has_html_tags'] = 1 if '<' in text else 0
        features['html_br_count'] = tag_counts.get('br_tags', 0)
        features['html_p_count'] = tag_counts.get('p_tags', 0)
        features['html_div_count'] = tag_counts.get('div_tags', 0)
        features['html_span_count'] = tag_counts.get('span_tags', 0)

        # Форматирование
        formatting = (tag_counts.get('strong_tags', 0) +
                     tag_counts.get('italic_tags', 0) +
                     tag_counts.get('underline_tags', 0))
        features['html_formatting_tags'] = formatting
        features['excessive_formatting'] = 1 if formatting > 5 else 0

        # Стили
        features['html_font_tags'] = tag_counts.get('font_tags', 0)
        features['html_color_usage'] = tag_counts.get('color_attrs', 0)
        features['has_colored_text'] = 1 if tag_counts.get('color_attrs', 0) > 0 else 0

        # Заголовки
        features['html_header_tags'] = tag_counts.get('h_tags', 0)
        features['has_h1_tag'] = 1 if tag_counts.get('h1_tags', 0) > 0 else 0

        # Entities
        features['html_entities_count'] = tag_counts.get('html_entities', 0)
        features['html_nbsp_count'] = text.count('&nbsp;')
        features['excessive_nbsp'] = 1 if features['html_nbsp_count'] > 10 else 0

        # Качество
        features['html_broken_tags'] = tag_counts.get('broken_tags', 0)
        features['poor_html_quality'] = 1 if features['html_broken_tags'] > 0 else 0

        # Подозрительные паттерны (вычисляем один раз)
        suspicious = 0
        suspicious += 1 if self.patterns['multiple_br'].search(text) else 0
        suspicious += 1 if self.patterns['multiple_bold'].search(text) else 0
        suspicious += 1 if self.patterns['bright_colors'].search(text) else 0
        suspicious += 2 if ('mso-' in text or 'MsoNormal' in text) else 0

        # Проверка устаревших тегов
        for tag in self.deprecated_tags:
            if f'<{tag}' in text.lower():
                suspicious += 1

        features['html_suspicious_pattern'] = suspicious

        # Общее количество тегов
        features['html_total_tags'] = sum(v for k, v in tag_counts.items()
                                         if k.endswith('_tags'))

        # Соотношения
        text_clean = self._clean_html_cached(text)
        features['html_to_text_ratio'] = 1 - (len(text_clean) / max(len(text), 1))
        features['formatting_density'] = formatting / max(len(text.split()), 1)

        return features



    def _extract_keyword_features_optimized(self, text_lower: str) -> Dict[str, float]:
        """Оптимизированное извлечение ключевых слов"""
        features = {}

        # Используем frozenset для быстрой проверки вхождения
        copy_score = sum(1 for keyword in self.copy_keywords if keyword in text_lower)
        features['copy_words_count'] = copy_score
        features['has_copy_words'] = 1 if copy_score > 0 else 0

        # Цены
        prices = self.patterns['price'].findall(text_lower)
        features['price_mentioned'] = 1 if prices else 0
        features['price_mentions_count'] = len(prices)

        # CTA фразы
        features['call_to_action_count'] = sum(1 for phrase in self.cta_phrases
                                              if phrase in text_lower)

        return features

    def _extract_style_features_optimized(self, text: str) -> Dict[str, float]:
        """Оптимизированное извлечение стилистических признаков"""
        features = {}

        if not text:
            return features

        # Векторизованный подсчет символов
        text_array = np.array(list(text))

        # Используем numpy для быстрого подсчета
        upper_mask = np.array([c.isupper() for c in text])
        lower_mask = np.array([c.islower() for c in text])

        upper_chars = np.sum(upper_mask)
        lower_chars = np.sum(lower_mask)
        total_alpha = upper_chars + lower_chars

        features['uppercase_ratio'] = upper_chars / max(total_alpha, 1)
        features['all_uppercase'] = 1 if upper_chars > 0 and lower_chars == 0 else 0
        features['all_lowercase'] = 1 if lower_chars > 0 and upper_chars == 0 else 0

        # Капитализация слов
        words = text.split()
        if words:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            features['capitalized_words_ratio'] = capitalized / len(words)

        # Пунктуация (используем set для быстрой проверки)
        punct_set = set('!?.,:;-—')
        punct_count = sum(1 for c in text if c in punct_set)
        features['punctuation_density'] = punct_count / max(len(text), 1)
        features['exclamation_marks'] = text.count('!')
        features['question_marks'] = text.count('?')
        features['ellipsis_count'] = text.count('...')

        # Эмодзи (используем оптимизированную функцию)
        features['has_emoji'] = 1 if self._has_emoji(text[:100]) else 0  # Проверяем только начало

        # Абзацы
        paragraphs = text.split('\n\n')
        features['paragraph_count'] = len(paragraphs)
        if paragraphs:
            lengths = [len(p) for p in paragraphs]
            features['avg_paragraph_length'] = np.mean(lengths)
        else:
            features['avg_paragraph_length'] = 0

        return features

    def _extract_repetition_features_optimized(self, text: str) -> Dict[str, float]:
        """Оптимизированное извлечение признаков повторений"""
        features = {}

        words = text.lower().split()

        if not words:
            return {
                'word_uniqueness': 0,
                'repeated_words_count': 0,
                'repeated_bigrams_count': 0,
                'max_word_repetition': 0,
                'text_entropy': 0,
                'repeated_sentences': 0
            }

        # Используем Counter для эффективного подсчета
        word_counts = Counter(words)

        # Уникальность
        unique_words = len(word_counts)
        total_words = len(words)
        features['word_uniqueness'] = unique_words / total_words

        # Повторения
        repeated = [w for w, c in word_counts.items() if c > 2]
        features['repeated_words_count'] = len(repeated)

        # Максимальное повторение
        features['max_word_repetition'] = max(word_counts.values())

        # Биграммы (оптимизированно)
        if len(words) > 1:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
            features['repeated_bigrams_count'] = repeated_bigrams
        else:
            features['repeated_bigrams_count'] = 0

        # Энтропия (векторизованный расчет)
        probs = np.array(list(word_counts.values())) / total_words
        features['text_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))

        # Повторяющиеся предложения
        sentences = self.patterns['sentences'].split(text)
        sentence_counts = Counter(s.strip() for s in sentences if s.strip())
        features['repeated_sentences'] = sum(1 for c in sentence_counts.values() if c > 1)

        return features

    def extract_features_batch(self, texts: pd.Series,
                              brand_names: Optional[pd.Series] = None,
                              categories: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Батчевое извлечение признаков для ускорения
        """
        features_list = []

        for i, text in enumerate(texts):
            brand = brand_names.iloc[i] if brand_names is not None else None
            category = categories.iloc[i] if categories is not None else None

            features = self.extract_features(text, brand, category)
            features_list.append(features)

        return pd.DataFrame(features_list)