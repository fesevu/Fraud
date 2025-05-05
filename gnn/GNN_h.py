# %% [markdown]
# # Установка зависимостей

# %%
# 0. Обновление pip и инструментов сборки
%pip install -U pip setuptools wheel

# 1. Установка PyTorch (CPU-версия)
%pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Обновление typing-extensions
%pip install -U typing-extensions

# 3. Установка зависимостей PyTorch Geometric
%pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
%pip install torch-geometric==2.2.0

# 4. Установка утилит
%pip install -U scikit-learn pandas networkx matplotlib rich tqdm
%pip install graphlime focal-loss-torch stella tensorflow

# %%
import os, random, numpy as np
import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, recall_score, confusion_matrix
import torch.optim as optim
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import networkx as nx
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_
import joblib
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import DirectedGraphSAGE
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from stellargraph.mapper import DirectedGraphSAGENodeGenerator
from stellargraph.layer import DirectedGraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.layer.gcn import GCN
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# %% [markdown]
# # Загрузка датасета

# %%
#@title 📂 Загрузка датасета (идемпотентно)
import os

# Папка, куда будет сохранён репозиторий
data_dir = 'data/'

import os
import shutil

data_dir = 'data/'

# Проверяем, существует ли папка
if os.path.exists(data_dir):
    # Если папка существует, удаляем все её содержимое
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
else:
    # Если папка не существует, создаем её
    os.makedirs(data_dir)


# Клонирование репозитория в папку data
!git clone https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git {data_dir}


import zipfile, pathlib
zip_path = "data/Dataset.zip"
extract_path = pathlib.Path("data/unpacked")
with zipfile.ZipFile(zip_path) as zf: zf.extractall(extract_path)
print("✅ Распаковка завершена")

# %%
#@title 🧹 Чтение CSV
import pandas as pd
dataset_csv_path = extract_path / "Dataset" / "Dataset.csv"
df = pd.read_csv(dataset_csv_path)
print(df.shape, "строк")
print(df.columns.tolist())

# %% [markdown]
# # Инженерия данных

# %%
# %% [data scrubbing]
# Список столбцов, в которых НЕ хотим допускать NaN
cols_required = [col for col in df.columns 
                 if col not in ['from_category', 'to_category']]

# Считаем пропуски по этим столбцам
missing_counts_req = df[cols_required].isnull().sum()
print("Missing values in required columns:\n", missing_counts_req)

# Дропаем строки, где есть хоть одно NaN в обязательных столбцах
df = df.dropna(subset=cols_required)
print("After dropping rows with missing in required cols:", df.shape)

# Проверка и дроп дубликатов
dup_count = df.duplicated().sum()
print("Duplicate rows count:", dup_count)
if dup_count > 0:
    df = df.drop_duplicates()
    print("Dropped duplicate rows. New shape:", df.shape)
else:
    print("No duplicate rows found.")

# %%
# Считаем количество 0 и 1 в from_scam / to_scam
from_counts = df['from_scam'].value_counts().rename({0: 'non_scam', 1: 'scam'})
to_counts   = df['to_scam'  ].value_counts().rename({0: 'non_scam', 1: 'scam'})

print("Sender (from_scam):")
print(from_counts)
print("\nRecipient (to_scam):")
print(to_counts)

# Если вас интересуют транзакции, где есть хоть одна пометка scam:
df['any_scam'] = ((df['from_scam'] == 1) | (df['to_scam'] == 1)).astype(int)
any_counts = df['any_scam'].value_counts().rename({0: 'no_scam', 1: 'has_scam'})
print("\nTransactions with any scam flag:")
print(any_counts)

# %%
import pandas as pd

# # Допустим, df уже загружен и очищен от NaN/дубликатов

# # 1) Балансировка по from_scam
# n_from_scam = df['from_scam'].sum()  # должно быть 2617
# df_from_pos = df[df['from_scam'] == 1]
# df_from_neg = df[df['from_scam'] == 0].sample(n=n_from_scam, random_state=42)
# df_balanced_from = pd.concat([df_from_pos, df_from_neg]) \
#                      .sample(frac=1, random_state=42) \
#                      .reset_index(drop=True)

# print("Balanced from_scam:")
# print(df_balanced_from['from_scam'].value_counts())

# # 2) Балансировка по to_scam
# n_to_scam = df['to_scam'].sum()  # 11638
# df_to_pos = df[df['to_scam'] == 1]
# df_to_neg = df[df['to_scam'] == 0].sample(n=n_to_scam, random_state=42)
# df_balanced_to = pd.concat([df_to_pos, df_to_neg]) \
#                    .sample(frac=1, random_state=42) \
#                    .reset_index(drop=True)

# print("\nBalanced to_scam:")
# print(df_balanced_to['to_scam'].value_counts())

# Если нужно ровно 2617 для обоих (независимо от фактического числа):
desired_n = 2617

# По from_scam
df_from_pos = df[df['from_scam'] == 1]
df_from_neg = df[df['from_scam'] == 0].sample(n=desired_n, random_state=42)
df_sampled_from_exact = pd.concat([df_from_pos, df_from_neg]) \
                           .sample(frac=1, random_state=42) \
                           .reset_index(drop=True)

# По to_scam
df_to_pos = df[df['to_scam'] == 1].sample(n=desired_n, random_state=42)
df_to_neg = df[df['to_scam'] == 0].sample(n=desired_n, random_state=42)
df_sampled_to_exact = pd.concat([df_to_pos, df_to_neg]) \
                         .sample(frac=1, random_state=42) \
                         .reset_index(drop=True)

print("\nExact 2617 each for from_scam:")
print(df_sampled_from_exact['from_scam'].value_counts())
print("\nExact 2617 each for to_scam:")
print(df_sampled_to_exact['to_scam'].value_counts())

# %%

df['block_year'] = (
    df['block_timestamp']
      .str.extract(r'^(\d{4})')[0]
      .astype(int)
)

# 2. Квантильное разбиение стоимости
df['value_cat'] = pd.qcut(df['value'], q=3, labels=False)

# 3. Адрес‑уровень: год регистрации и максимальное значение
addr_years = pd.concat([
        df[['from_address','block_year']].rename(columns={'from_address':'address'}),
        df[['to_address','block_year']].rename(columns={'to_address':'address'})
])
addr_reg_year  = addr_years.groupby('address')['block_year'].min().rename('reg_year')

addr_vals = pd.concat([
        df[['from_address','value']].rename(columns={'from_address':'address'}),
        df[['to_address','value']].rename(columns={'to_address':'address'})
])
addr_max_value = addr_vals.groupby('address')['value'].max().rename('max_value')

address_features = (
    pd.concat([addr_reg_year, addr_max_value], axis=1)
      .reset_index()                     # колонка index → address
      .rename(columns={'index':'address'})
)

current_year = df['block_year'].max()
address_features['years_in_operation'] = (
        current_year - address_features['reg_year']
)

# 4. Присоединяем к транзакциям
df = (df
      .merge(address_features.add_prefix('from_'),
             left_on='from_address', right_on='from_address', how='left')
      .merge(address_features.add_prefix('to_'),
             left_on='to_address',   right_on='to_address',   how='left')
)

edge_features = [
    'gas','gas_price','receipt_cumulative_gas_used','receipt_gas_used',
    'block_number','value_cat',
    'from_years_in_operation','to_years_in_operation'
]
node_features = [
    'from_reg_year','to_reg_year',
    'from_max_value','to_max_value'
]
response_cols = ['from_scam','to_scam']

print(df.shape)
print("Edge :", edge_features)
print("Node :", node_features)

# %%
# %% [data preview]
# Show sample records from beginning, middle, and end for clarity
sample_idxs = [0, df.shape[0] // 2, df.shape[0] - 1]
preview_df = df.loc[sample_idxs, edge_features + node_features + response_cols]
print("\nPreview of selected features at various positions:\n")
print(preview_df.to_string(index=True))


