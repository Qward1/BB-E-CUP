import numpy as np
import pandas as pd
from typing import Optional, Union, Generator
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import re
import gc
from pathlib import Path
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class CPUOptimizedVectorizer:
    """
    Векторизатор оптимизированный для CPU и больших данных
    Исправлена проблема с сохранением в parquet
    """

    def __init__(self, strategy: str = 'auto'):
        """
        Args:
            strategy: 'fast' (TF-IDF), 'balanced' (Sentence-BERT),
                     'quality' (XLM-RoBERTa), 'auto' (автовыбор)
        """
        self.strategy = strategy
        self.html_pattern = re.compile(r'<[^>]+>')

        # Инициализируем нужную модель
        if strategy == 'auto':
            self._auto_select_strategy()
        else:
            self._init_model(strategy)

    def _auto_select_strategy(self):
        """Автоматический выбор стратегии на основе доступных ресурсов"""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        except:
            available_memory = 8  # По умолчанию

        has_cuda = torch.cuda.is_available()

        print(f"Available memory: {available_memory:.1f} GB")
        print(f"CUDA available: {has_cuda}")

        if has_cuda:
            self.strategy = 'quality'
        elif available_memory > 16:
            self.strategy = 'balanced'
        else:
            self.strategy = 'fast'

        print(f"Selected strategy: {self.strategy}")
        self._init_model(self.strategy)

    def _init_model(self, strategy: str):
        """Инициализация модели в зависимости от стратегии"""

        if strategy == 'fast':
            # TF-IDF + SVD для быстрой векторизации
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                dtype=np.float32  # Используем float32 для совместимости
            )
            self.svd = TruncatedSVD(n_components=300)
            self.embedding_dim = 300
            print("Initialized TF-IDF + SVD (fast mode)")

        elif strategy == 'balanced':
            # Sentence-BERT - хороший баланс
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.embedding_dim = 384
                self.model.max_seq_length = 256
                print("Initialized Sentence-BERT (balanced mode)")
            except Exception as e:
                print(f"Failed to load Sentence-BERT: {e}")
                print("Falling back to TF-IDF")
                self.strategy = 'fast'
                self._init_model('fast')

        elif strategy == 'quality':
            # XLM-RoBERTa для максимального качества
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                self.model = AutoModel.from_pretrained('xlm-roberta-base')
                self.model.eval()
                self.embedding_dim = 768

                # Оптимизация для CPU
                if not torch.cuda.is_available():
                    # Квантизация для CPU
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )

                print("Initialized XLM-RoBERTa (quality mode)")
            except Exception as e:
                print(f"Failed to load XLM-RoBERTa: {e}")
                print("Falling back to balanced mode")
                self.strategy = 'balanced'
                self._init_model('balanced')

    def clean_text(self, text: str) -> str:
        """Очистка текста от HTML"""
        if pd.isna(text) or text is None:
            return ""

        text = str(text)
        text = self.html_pattern.sub(' ', text)
        text = text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')
        text = ' '.join(text.split())

        # Ограничиваем длину
        if len(text) > 1000:
            text = text[:1000]

        return text

    def encode_batch_fast(self, texts: list) -> np.ndarray:
        """Быстрое кодирование через TF-IDF"""
        cleaned = [self.clean_text(t) for t in texts]

        # Если модель не обучена
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(cleaned)
            tfidf_matrix = self.vectorizer.transform(cleaned)
            self.svd.fit(tfidf_matrix)
        else:
            tfidf_matrix = self.vectorizer.transform(cleaned)

        # Преобразуем в плотную матрицу и применяем SVD
        embeddings = self.svd.transform(tfidf_matrix)

        # Конвертируем в float32 для совместимости с parquet
        return embeddings.astype(np.float32)

    def encode_batch_balanced(self, texts: list) -> np.ndarray:
        """Кодирование через Sentence-BERT"""
        cleaned = [self.clean_text(t) for t in texts]
        embeddings = self.model.encode(cleaned, batch_size=32, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def encode_batch_quality(self, texts: list, batch_size: int = 4) -> np.ndarray:
        """Кодирование через XLM-RoBERTa"""
        cleaned = [self.clean_text(t) for t in texts]
        embeddings = []

        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                embeddings.append(pooled.cpu().numpy())

        result = np.vstack(embeddings) if embeddings else np.empty((0, self.embedding_dim))
        return result.astype(np.float32)

    def encode(self, texts: Union[list, pd.Series], batch_size: int = 100) -> np.ndarray:
        """Универсальный метод кодирования"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]

            try:
                if self.strategy == 'fast':
                    embeddings = self.encode_batch_fast(batch)
                elif self.strategy == 'balanced':
                    embeddings = self.encode_batch_balanced(batch)
                elif self.strategy == 'quality':
                    embeddings = self.encode_batch_quality(batch)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                all_embeddings.append(embeddings)

            except Exception as e:
                print(f"Error encoding batch {i // batch_size}: {e}")
                # Создаем пустые эмбеддинги для проблемного батча
                empty_embeddings = np.zeros((len(batch), self.embedding_dim), dtype=np.float32)
                all_embeddings.append(empty_embeddings)

            # Очистка памяти
            if i % 1000 == 0:
                gc.collect()

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim))

    def process_large_file(self,
                           file_path: str,
                           text_column: str = 'description',
                           chunksize: int = 10000,
                           save_dir: str = 'embeddings/',
                           save_format: str = 'csv') -> None:
        """
        Обработка большого файла по частям с сохранением на диск

        Args:
            file_path: Путь к CSV/Parquet файлу
            text_column: Колонка с текстами
            chunksize: Размер чанка
            save_dir: Директория для сохранения
            save_format: Формат сохранения ('parquet' или 'csv')
        """
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        # Определяем формат файла
        file_ext = Path(file_path).suffix.lower()

        print(f"Processing file: {file_path}")
        print(f"File format: {file_ext}")
        print(f"Save format: {save_format}")

        # Читаем файл по частям
        if file_ext == '.parquet':
            try:
                df_iterator = pd.read_parquet(file_path)
                # Разбиваем на чанки вручную
                total_rows = len(df_iterator)
                df_iterator = [df_iterator.iloc[i:i + chunksize]
                               for i in range(0, total_rows, chunksize)]
            except:
                print("Cannot read parquet in chunks, trying to read full file...")
                df = pd.read_parquet(file_path)
                df_iterator = [df.iloc[i:i + chunksize]
                               for i in range(0, len(df), chunksize)]
        else:
            df_iterator = pd.read_csv(file_path, chunksize=chunksize)

        chunk_num = 0

        for chunk in tqdm(df_iterator, desc="Processing chunks"):
            try:
                # Проверяем наличие колонки
                if text_column not in chunk.columns:
                    print(f"Warning: Column '{text_column}' not found in chunk {chunk_num}")
                    print(f"Available columns: {chunk.columns.tolist()}")
                    continue

                # Получаем эмбеддинги
                texts = chunk[text_column].fillna('')
                embeddings = self.encode(texts)

                # Создаем DataFrame с эмбеддингами
                embed_cols = [f'embed_{i}' for i in range(embeddings.shape[1])]
                embed_df = pd.DataFrame(
                    embeddings,
                    columns=embed_cols,
                    dtype=np.float32  # Явно указываем тип данных
                )

                # Добавляем ID если есть
                if 'item_id' in chunk.columns:
                    embed_df['item_id'] = chunk['item_id'].values
                else:
                    # Создаем индекс
                    start_idx = chunk_num * chunksize
                    embed_df['item_id'] = range(start_idx, start_idx + len(chunk))

                # Сохраняем чанк
                if save_format == 'parquet':
                    save_path = Path(save_dir) / f'embeddings_chunk_{chunk_num:04d}.parquet'
                    try:
                        embed_df.to_parquet(save_path, compression='snappy', engine='pyarrow')
                    except Exception as e:
                        print(f"Failed to save as parquet: {e}")
                        print("Trying fastparquet engine...")
                        try:
                            embed_df.to_parquet(save_path, compression='snappy', engine='fastparquet')
                        except:
                            print("Falling back to CSV format...")
                            save_path = Path(save_dir) / f'embeddings_chunk_{chunk_num:04d}.csv'
                            embed_df.to_csv(save_path, index=False)
                else:
                    save_path = Path(save_dir) / f'embeddings_chunk_{chunk_num:04d}.csv'
                    embed_df.to_csv(save_path, index=False)

                print(f"Saved chunk {chunk_num} to {save_path}")
                chunk_num += 1

                # Очистка памяти
                del embeddings, embed_df, chunk
                gc.collect()

            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {e}")
                chunk_num += 1
                continue

        print(f"Processed {chunk_num} chunks. Saved to {save_dir}")

        # Объединение чанков
        try:
            self._merge_chunks(save_dir, save_format)
        except Exception as e:
            print(f"Failed to merge chunks: {e}")
            print("Chunks are saved separately in the directory")

    def _merge_chunks(self, save_dir: str, save_format: str = 'parquet'):
        """Объединение чанков в один файл"""
        import glob

        # Ищем файлы чанков
        if save_format == 'parquet':
            chunk_files = sorted(glob.glob(f"{save_dir}/embeddings_chunk_*.parquet"))
        else:
            chunk_files = sorted(glob.glob(f"{save_dir}/embeddings_chunk_*.csv"))

        if not chunk_files:
            print("No chunk files found to merge")
            return

        print(f"Merging {len(chunk_files)} chunks...")

        # Читаем и объединяем
        all_chunks = []
        for file in tqdm(chunk_files, desc="Reading chunks"):
            try:
                if save_format == 'parquet':
                    chunk = pd.read_parquet(file)
                else:
                    chunk = pd.read_csv(file)
                all_chunks.append(chunk)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

        if not all_chunks:
            print("No chunks could be read")
            return

        # Объединяем
        print("Concatenating chunks...")
        final_df = pd.concat(all_chunks, ignore_index=True)

        # Сохраняем финальный файл
        if save_format == 'parquet':
            final_path = Path(save_dir) / 'embeddings_final.parquet'
            try:
                final_df.to_parquet(final_path, compression='snappy', engine='pyarrow')
            except:
                try:
                    final_df.to_parquet(final_path, compression='snappy', engine='fastparquet')
                except:
                    final_path = Path(save_dir) / 'embeddings_final.csv'
                    final_df.to_csv(final_path, index=False)
        else:
            final_path = Path(save_dir) / 'embeddings_final.csv'
            final_df.to_csv(final_path, index=False)

        print(f"Final embeddings saved to {final_path}")
        print(f"Shape: {final_df.shape}")

        # Удаляем чанки
        print("Cleaning up chunk files...")
        for file in chunk_files:
            try:
                Path(file).unlink()
            except:
                pass

    def save_model(self, path: str):
        """Сохранение модели"""
        Path(path).parent.mkdir(exist_ok=True, parents=True)

        model_data = {
            'strategy': self.strategy,
            'embedding_dim': self.embedding_dim
        }

        if self.strategy == 'fast':
            model_data['vectorizer'] = self.vectorizer
            model_data['svd'] = self.svd

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str):
        """Загрузка модели"""
        model_data = joblib.load(path)

        instance = cls(strategy=model_data['strategy'])

        if model_data['strategy'] == 'fast':
            instance.vectorizer = model_data['vectorizer']
            instance.svd = model_data['svd']
            instance.embedding_dim = model_data['embedding_dim']

        return instance