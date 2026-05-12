"""
Тесты для проверки воспроизводимости и целостности пайплайна.

Запуск:
    pytest tests/
    или
    python -m pytest tests/
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Добавляем корень проекта в PYTHONPATH (если нужно)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_files_exist():
    """Проверяем, что все необходимые файлы данных присутствуют."""
    required_files = [
        '../data/processed/train.csv',
        '../data/processed/val.csv',
        '../data/processed/test.csv'
    ]
    for f in required_files:
        assert os.path.exists(f), f"Файл {f} не найден"


def test_train_data_no_nulls():
    """В train.csv не должно быть пропусков."""
    df = pd.read_csv('../data/processed/train.csv')
    total_nulls = df.isnull().sum().sum()
    assert total_nulls == 0, f"Найдено {total_nulls} пропусков в train.csv"


def test_val_data_no_nulls():
    """В val.csv не должно быть пропусков."""
    df = pd.read_csv('../data/processed/val.csv')
    total_nulls = df.isnull().sum().sum()
    assert total_nulls == 0, f"Найдено {total_nulls} пропусков в val.csv"


def test_test_data_no_nulls():
    """В test.csv не должно быть пропусков."""
    df = pd.read_csv('../data/processed/test.csv')
    total_nulls = df.isnull().sum().sum()
    assert total_nulls == 0, f"Найдено {total_nulls} пропусков в test.csv"


def test_target_range():
    """Целевая переменная rating должна быть в диапазоне 800–3500."""
    train = pd.read_csv('../data/processed/train.csv')
    val = pd.read_csv('../data/processed/val.csv')
    test = pd.read_csv('../data/processed/test.csv')
    for name, df in [('train', train), ('val', val), ('test', test)]:
        assert df['rating'].between(800, 3500).all(), \
            f"В {name} есть rating вне диапазона 800-3500"


def test_feature_columns_consistency():
    """Набор признаков должен быть одинаков в train/val/test."""
    train = pd.read_csv('../data/processed/train.csv')
    val = pd.read_csv('../data/processed/val.csv')
    test = pd.read_csv('../data/processed/test.csv')

    target = 'rating'
    exclude_cols = ['contestId', 'index', 'name', 'tags', 'tags_list', 'top_tag', 'statement', target]
    train_features = set([c for c in train.columns if c not in exclude_cols])
    val_features = set([c for c in val.columns if c not in exclude_cols])
    test_features = set([c for c in test.columns if c not in exclude_cols])

    assert train_features == val_features, "Признаки train и val различаются"
    assert train_features == test_features, "Признаки train и test различаются"


def test_model_exists_and_loads():
    """Проверяем, что обученная модель существует и загружается."""
    model_path = '../models/best_model.pkl'
    assert os.path.exists(model_path), f"Модель {model_path} не найдена"
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Загруженный объект не является моделью с методом predict()"


def test_split_sizes_are_reasonable():
    """Проверяем разумность размеров сплита (train ~70%, val ~15%, test ~15%)."""
    train = pd.read_csv('../data/processed/train.csv')
    val = pd.read_csv('../data/processed/val.csv')
    test = pd.read_csv('../data/processed/test.csv')
    total = len(train) + len(val) + len(test)
    train_ratio = len(train) / total
    val_ratio = len(val) / total
    test_ratio = len(test) / total
    # Допускаем небольшие отклонения из-за сплита по контестам
    assert 0.65 <= train_ratio <= 0.75
    assert 0.12 <= val_ratio <= 0.18
    assert 0.12 <= test_ratio <= 0.18


if __name__ == '__main__':
    # Для запуска без pytest
    test_data_files_exist()
    test_train_data_no_nulls()
    test_val_data_no_nulls()
    test_test_data_no_nulls()
    test_target_range()
    test_feature_columns_consistency()
    test_model_exists_and_loads()
    test_split_sizes_are_reasonable()
    print("Все тесты пройдены успешно!")