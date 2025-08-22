import re
from unicodedata import category
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from collections import Counter
import Levenshtein
import matplotlib.pyplot as plt
from src import future_extract as fe
from src import text_model as tm
import matplotlib as mpl
import seaborn as sns
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
mpl.style.use('ggplot')
sns.set_style('white')



def load_and_fill_data(path_train, path_test):

    data_raw_full = pd.read_csv(path_train)
    data_val_full = pd.read_csv(path_test)

    data_raw = data_raw_full[['id', 'description', 'brand_name', 'CommercialTypeName4', 'resolution']]
    data_val = data_val_full[['id', 'description', 'brand_name', 'CommercialTypeName4']]

    data_name_rus_raw = data_raw_full['name_rus']
    data_name_rus_val = data_val_full['name_rus']

    data_val['description'] = data_val['description'].fillna('')
    data_val['brand_name'] = data_val['brand_name'].fillna('Unknown')

    data_raw['description'] = data_raw['description'].fillna('')
    data_raw['brand_name'] = data_raw['brand_name'].fillna('Unknown')

    data_val_full['description'] = data_val_full['description'].fillna('')
    data_val_full['brand_name'] = data_val_full['brand_name'].fillna('Unknown')

    data_raw_full['description'] = data_raw_full['description'].fillna('')
    data_raw_full['brand_name'] = data_raw_full['brand_name'].fillna('Unknown')

    return data_raw, data_val, data_raw_full, data_val_full, data_name_rus_raw, data_name_rus_val

def extract_features(data_r, data_v, data_name_rus_raw, data_name_rus_val):

    extractor = fe.OptimizedEcommerceExtractor()

    columns_for_del = ['has_html_tags', 'html_br_count', 'html_p_count', 'html_div_count', 'html_span_count',
                       'html_formatting_tags', 'excessive_formatting', 'html_font_tags', 'html_color_usage',
                       'has_colored_text', 'html_header_tags', 'has_h1_tag', 'html_entities_count', 'html_nbsp_count',
                       'excessive_nbsp', 'html_broken_tags', 'poor_html_quality', 'html_suspicious_pattern',
                       'html_total_tags', 'html_to_text_ratio', 'formatting_density', 'question_marks', 'has_emoji',
                       'paragraph_count']

    features_df_raw = extractor.extract_features_batch(data_r['description'])
    data_raw_with_features = pd.concat([data_r, features_df_raw.add_prefix('desc_')], axis=1)

    features_df_val = extractor.extract_features_batch(data_v['description'])
    data_val_with_features = pd.concat([data_v, features_df_val.add_prefix('desc_')], axis=1)

    features_df_name_rus_raw = extractor.extract_features_batch(data_name_rus_raw)
    features_df_name_rus_raw = features_df_name_rus_raw.drop(columns_for_del, axis=1)
    name_rus_raw_with_features = pd.concat([data_name_rus_raw, features_df_name_rus_raw.add_prefix('desc_')], axis=1)

    features_df_name_rus_val = extractor.extract_features_batch(data_name_rus_val)
    features_df_name_rus_val = features_df_name_rus_val.drop(columns_for_del, axis=1)
    name_rus_val_with_features = pd.concat([data_name_rus_val, features_df_name_rus_val.add_prefix('desc_')], axis=1)

    return data_raw_with_features, data_val_with_features, name_rus_raw_with_features, name_rus_val_with_features


def vectoriz_text(data_raw_with_features, data_val_with_features, name_rus_raw_with_features, name_rus_val_with_features):

    vectorizer = tm.CPUOptimizedVectorizer(strategy='fast')


    data_raw_with_features_and_embed = vectorizer.process_dataframe(
             data_raw_with_features,
             text_column='description',
             batch_size=100,
             add_to_original=True  # Добавить к исходным данным
         )

    data_val_with_features_and_embed = vectorizer.process_dataframe(
        data_val_with_features,
        text_column='description',
        batch_size=100,
        add_to_original=True  # Добавить к исходным данным
    )

    name_rus_raw_with_features_and_embed = vectorizer.process_dataframe(
        name_rus_raw_with_features,
        text_column='name_rus',
        batch_size=1000,
        add_to_original=True  # Добавить к исходным данным
    )

    name_rus_val_with_features_and_embed = vectorizer.process_dataframe(
        name_rus_val_with_features,
        text_column='name_rus',
        batch_size=1000,
        add_to_original=True  # Добавить к исходным данным
    )


    name_rus_raw_with_features_and_embed = name_rus_raw_with_features_and_embed.drop(['name_rus'], axis=1)
    name_rus_val_with_features_and_embed = name_rus_val_with_features_and_embed.drop(['name_rus'], axis=1)

    data_raw_with_features_and_embed = data_raw_with_features_and_embed.drop(['description', 'id'], axis=1)
    data_val_with_features_and_embed = data_val_with_features_and_embed.drop(['description', 'id'], axis=1)

    return data_raw_with_features_and_embed, data_val_with_features_and_embed, name_rus_raw_with_features_and_embed, name_rus_val_with_features_and_embed


def clean_small_brands(data_raw_with_features_and_embed, data_val_with_features_and_embed):

    brand_counts = data_raw_with_features_and_embed["brand_name"].value_counts()
    CommercialTypeName4_counts = data_raw_with_features_and_embed["CommercialTypeName4"].value_counts()

    data_raw_with_features_and_embed["brand_name"] = data_raw_with_features_and_embed["brand_name"].where(
        data_raw_with_features_and_embed["brand_name"].map(brand_counts) >= 20, "Another")

    data_raw_with_features_and_embed["CommercialTypeName4"] = data_raw_with_features_and_embed[
        "CommercialTypeName4"].where(
        data_raw_with_features_and_embed["CommercialTypeName4"].map(CommercialTypeName4_counts) >= 20, "Another")


    brand_counts_val = data_val_with_features_and_embed["brand_name"].value_counts()
    CommercialTypeName4_counts_val = data_val_with_features_and_embed["CommercialTypeName4"].value_counts()


    data_val_with_features_and_embed["brand_name"] = data_val_with_features_and_embed["brand_name"].where(
        data_val_with_features_and_embed["brand_name"].map(brand_counts_val) >= 20, "Another")

    data_val_with_features_and_embed["CommercialTypeName4"] = data_val_with_features_and_embed[
        "CommercialTypeName4"].where(
        data_val_with_features_and_embed["CommercialTypeName4"].map(CommercialTypeName4_counts_val) >= 20, "Another")


    return data_raw_with_features_and_embed, data_val_with_features_and_embed

def contact_name_rus_with_data(data_raw_with_features_and_embed, data_val_with_features_and_embed, name_rus_raw_with_features_and_embed, name_rus_val_with_features_and_embed):

    data_raw_with_features_and_embed = pd.concat([data_raw_with_features_and_embed, name_rus_raw_with_features_and_embed.add_prefix('namerus_')], axis=1)
    data_val_with_features_and_embed = pd.concat([data_val_with_features_and_embed, name_rus_val_with_features_and_embed.add_prefix('namerus_')], axis=1)
    return data_raw_with_features_and_embed, data_val_with_features_and_embed

def poluchenie_dummy(data_raw_with_features_and_embed, data_val_with_features_and_embed, data_raw_with_features):

    all_data = pd.concat([data_raw_with_features_and_embed, data_val_with_features_and_embed], axis=0)

    all_dummies = pd.get_dummies(all_data, columns=["brand_name", "CommercialTypeName4"], dummy_na=True, drop_first=False)

    data_raw_with_features_and_embed_OHE = all_dummies.iloc[:len(data_raw_with_features_and_embed)]
    data_val_with_features_and_embed_OHE = all_dummies.iloc[len(data_raw_with_features_and_embed):]

    # из за OHE создаётся два resolution
    data_raw_with_features_and_embed_OHE = data_raw_with_features_and_embed_OHE.drop('resolution', axis=1)
    data_raw_with_features_and_embed_OHE['resolution'] = data_raw_with_features['resolution']

    return data_raw_with_features_and_embed_OHE, data_val_with_features_and_embed_OHE

def poluchenie_catboostenc(data_raw_with_features_and_embed, data_val_with_features_and_embed):

    target_col = "resolution"  # укажи свою целевую колонку
    cat_cols = ["brand_name", "CommercialTypeName4"]  # какие категориальные кодируем
    cbe = CatBoostEncoder(cols=cat_cols, random_state=42, handle_unknown='ignore')

    cbe.fit(data_raw_with_features_and_embed[cat_cols], data_raw_with_features_and_embed[target_col])

    data_raw_with_features_and_embed_enc = data_raw_with_features_and_embed.copy()
    data_val_with_features_and_embed_enc = data_val_with_features_and_embed.copy()

    data_raw_with_features_and_embed_enc[cat_cols] = cbe.transform(data_raw_with_features_and_embed[cat_cols])
    data_val_with_features_and_embed_enc[cat_cols] = cbe.transform(data_val_with_features_and_embed[cat_cols])

    return data_raw_with_features_and_embed_enc, data_val_with_features_and_embed_enc

def poluchenie_frequency_enc(data_raw_with_features_and_embed, data_val_with_features_and_embed):
    target_col = "resolution"  # целевая колонка
    cat_cols = ["brand_name", "CommercialTypeName4"]  # категориальные колонки

    # Создаем копии датафреймов
    data_raw_with_features_and_embed_fenc = data_raw_with_features_and_embed.copy()
    data_val_with_features_and_embed_fenc = data_val_with_features_and_embed.copy()

    for col in cat_cols:
        freq = data_raw_with_features_and_embed[col].value_counts()
        data_raw_with_features_and_embed_fenc[col] = data_raw_with_features_and_embed[col].map(freq)
        data_val_with_features_and_embed_fenc[col] = data_val_with_features_and_embed[col].map(freq).fillna(0)

    return data_raw_with_features_and_embed_fenc, data_val_with_features_and_embed_fenc


def poluchenie_hash_enc(data_raw_with_features_and_embed, data_val_with_features_and_embed, n_features=8):
    cat_cols = ["brand_name", "CommercialTypeName4"]  # категориальные колонки

    data_raw_with_features_and_embed_henc = data_raw_with_features_and_embed.copy()
    data_val_with_features_and_embed_henc = data_val_with_features_and_embed.copy()

    for col in cat_cols:
        hasher = FeatureHasher(n_features=n_features, input_type='string')

        # Хешируем train
        hashed_train = hasher.transform(data_raw_with_features_and_embed[col].astype(str))
        hashed_train_df = pd.DataFrame(hashed_train.toarray(),
                                       columns=[f"{col}_hash_{i}" for i in range(n_features)],
                                       index=data_raw_with_features_and_embed.index)

        # Хешируем val
        hashed_val = hasher.transform(data_val_with_features_and_embed[col].astype(str))
        hashed_val_df = pd.DataFrame(hashed_val.toarray(),
                                     columns=[f"{col}_hash_{i}" for i in range(n_features)],
                                     index=data_val_with_features_and_embed.index)

        # Заменяем колонку на хешированные признаки
        data_raw_with_features_and_embed_henc = data_raw_with_features_and_embed_henc.drop(columns=[col]).join(hashed_train_df)
        data_val_with_features_and_embed_henc = data_val_with_features_and_embed_henc.drop(columns=[col]).join(hashed_val_df)

    return data_raw_with_features_and_embed_henc, data_val_with_features_and_embed_henc


def full_preprocess(path_train, path_test, get_encoder='CatEncoder'):

    data_raw, data_val, data_raw_full, data_val_full, data_name_rus_raw, data_name_rus_val = load_and_fill_data(path_train, path_test)
    data_raw_with_features, data_val_with_features, name_rus_raw_with_features, name_rus_val_with_features = extract_features(data_raw, data_val, data_name_rus_raw, data_name_rus_val)
    data_raw_with_features_and_embed, data_val_with_features_and_embed, name_rus_raw_with_features_and_embed, name_rus_val_with_features_and_embed = vectoriz_text(data_raw_with_features, data_val_with_features, name_rus_raw_with_features, name_rus_val_with_features)
    data_raw_with_features_and_embed, data_val_with_features_and_embed = clean_small_brands(data_raw_with_features_and_embed, data_val_with_features_and_embed)
    data_raw_with_features_and_embed, data_val_with_features_and_embed = contact_name_rus_with_data(data_raw_with_features_and_embed, data_val_with_features_and_embed, name_rus_raw_with_features_and_embed, name_rus_val_with_features_and_embed)
    if get_encoder == 'CatEncoder':
        data_raw_with_features_and_embed_enc, data_val_with_features_and_embed_enc = poluchenie_catboostenc(data_raw_with_features_and_embed, data_val_with_features_and_embed)
        return data_raw_with_features_and_embed_enc, data_val_with_features_and_embed_enc
    elif get_encoder == 'ohe':
        data_raw_with_features_and_embed_OHE, data_val_with_features_and_embed_OHE = poluchenie_dummy(data_raw_with_features_and_embed, data_val_with_features_and_embed, data_raw_with_features)
        return data_raw_with_features_and_embed_OHE, data_val_with_features_and_embed_OHE
    elif get_encoder == 'freq':
        data_raw_with_features_and_embed_fenc, data_val_with_features_and_embed_fenc = poluchenie_frequency_enc(data_raw_with_features_and_embed, data_val_with_features_and_embed)
        return data_raw_with_features_and_embed_fenc, data_val_with_features_and_embed_fenc
    elif get_encoder == 'hash':
        data_raw_with_features_and_embed_henc, data_val_with_features_and_embed_henc = poluchenie_hash_enc(data_raw_with_features_and_embed, data_val_with_features_and_embed)
        return data_raw_with_features_and_embed_henc, data_val_with_features_and_embed_henc


