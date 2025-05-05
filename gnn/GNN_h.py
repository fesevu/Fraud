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
from stellargraph.layer.gcn import GCN
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

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
# Check for missing values and drop if any
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)
if missing_counts.any():
    df = df.dropna()
    print("Dropped rows with missing values. New shape:", df.shape)
else:
    print("No missing values found.")

# Check for duplicate rows and drop if any
dup_count = df.duplicated().sum()
print("Duplicate rows count:", dup_count)
if dup_count > 0:
    df = df.drop_duplicates()
    print("Dropped duplicate rows. New shape:", df.shape)
else:
    print("No duplicate rows found.")
# %% [feature engineering]
# Reload dataframe if block_timestamp column was dropped in a previous run
if 'block_timestamp' not in df.columns:
    df = pd.read_csv(dataset_csv_path)
# Extract year directly from block_timestamp string
df['block_year'] = df['block_timestamp'].str[:4].astype(int)

# Categorize transaction values into 3 quantile-based categories
df['value_cat'] = pd.qcut(df['value'], q=3, labels=False)

# Compute registration year (first seen block year) per node address
from_reg = df.groupby('from_address')['block_year'].min().rename('from_reg_year')
to_reg   = df.groupby('to_address')['block_year'].min().rename('to_reg_year')
df = df.merge(from_reg, on='from_address', how='left').merge(to_reg, on='to_address', how='left')

# Compute years in operation based on the latest block year in the dataset
current_year = df['block_year'].max()
df['from_years_in_operation'] = current_year - df['from_reg_year']
df['to_years_in_operation']   = current_year - df['to_reg_year']

# Compute max transaction value per node address
from_max = df.groupby('from_address')['value'].max().rename('from_max_value')
to_max   = df.groupby('to_address')['value'].max().rename('to_max_value')
df = df.merge(from_max, on='from_address', how='left').merge(to_max, on='to_address', how='left')

# Drop unneeded columns
cols_to_drop = [
    'hash', 'nonce', 'transaction_index', 'input',
    'block_timestamp', 'block_hash',
    'from_category', 'to_category', 'value'
]
df = df.drop(columns=cols_to_drop)

# Define feature sets
edge_features   = [
    'gas', 'gas_price', 'receipt_cumulative_gas_used', 'receipt_gas_used',
    'block_number', 'value_cat', 'from_years_in_operation', 'to_years_in_operation'
]
node_features   = ['from_reg_year', 'to_reg_year', 'from_max_value', 'to_max_value']
response_cols   = ['from_scam', 'to_scam']

print("Dataframe after feature engineering:", df.shape)
print("Edge features:", edge_features)
print("Node features:", node_features)
print("Response columns:", response_cols)

# %% [data preview]
# Show sample records from beginning, middle, and end for clarity
sample_idxs = [0, df.shape[0] // 2, df.shape[0] - 1]
preview_df = df.loc[sample_idxs, edge_features + node_features + response_cols]
print("\nPreview of selected features at various positions:\n")
print(preview_df.to_string(index=True))

# %% [model definition]
# Build StellarGraph for directed transactions
# Prepare edge DataFrame
edges_sg = pd.DataFrame({
    "source": df["from_address"],
    "target": df["to_address"],
    **{feat: df[feat] for feat in edge_features}
})

# Prepare node features by averaging duplicates
node_feat_from = df[["from_address"] + node_features].rename(columns={"from_address": "id"}).set_index("id")
node_feat_to   = df[["to_address"]   + node_features].rename(columns={"to_address":   "id"}).set_index("id")
node_features_df = pd.concat([node_feat_from, node_feat_to]).groupby(level=0).mean()
# Scale node features to zero mean and unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
node_features_df[:] = scaler.fit_transform(node_features_df)

# Create StellarGraph object
G = sg.StellarGraph(
    nodes=node_features_df,
    edges=edges_sg,
    node_type_default="address",
    edge_type_default="transaction"
)


import tensorflow as tf
# %% [model definition]
# Full-batch generator with self-loops and dense adjacency via transform
generator = FullBatchNodeGenerator(
    G,
    method="self_loops",
    sparse=True,
    transform=lambda features, A: (features, tf.sparse.to_dense(A))
)

# Define GCN model with specified layer sizes
layer_sizes = [4, 32, 16, 16, 16, 4]
gcn = GCN(
    layer_sizes=layer_sizes,
    activations=["relu"] * len(layer_sizes),
    generator=generator,
    dropout=0.5
)

# Build input/output tensors and add output layer
x_inp, x_out = gcn.in_out_tensors()
prediction = Dense(units=2, activation="softmax")(x_out)

# Assemble and compile model
model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss=categorical_crossentropy,
    metrics=["acc"]
)

# Display model architecture
model.summary()

# %% [model training]
# Prepare node labels for classification: aggregate scam flag per node
from sklearn.model_selection import train_test_split

# Aggregate scam labels by node (1 if any transaction flagged)
from_labels = df.groupby("from_address")["from_scam"].max().rename("scam")
to_labels   = df.groupby("to_address")["to_scam"].max().rename("scam")
node_labels = pd.concat([from_labels, to_labels]).groupby(level=0).max()

# One-hot encode labels
node_targets = pd.get_dummies(node_labels).sort_index()

# Split nodes into train (50%), validation (25%), test (25%)
nodes = node_targets.index
train_nodes, temp_nodes = train_test_split(
    nodes,
    train_size=0.5,
    stratify=node_targets,
    random_state=42
)
val_nodes, test_nodes = train_test_split(
    temp_nodes,
    train_size=0.5,
    stratify=node_targets.loc[temp_nodes],
    random_state=42
)

train_targets = node_targets.loc[train_nodes]
val_targets   = node_targets.loc[val_nodes]
test_targets  = node_targets.loc[test_nodes]

# Create generators for training, validation, and testing
train_gen = generator.flow(train_nodes, train_targets.values)
val_gen   = generator.flow(val_nodes,   val_targets.values)
test_gen  = generator.flow(test_nodes,  test_targets.values)

# Train the model for 20 epochs with validation
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    verbose=2
)

# Evaluate on test set
loss, accuracy = model.evaluate(test_gen, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# %%
