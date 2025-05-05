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

# %% [markdown]
# # Определяем модель

# %%
# %% [define model]
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GCN, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.convs = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i+1]))
        self.activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)
        return x

# Instantiate model
input_dim = len(node_features)
hidden_dims = [32, 16, 16, 16, 4]
output_dim = len(response_cols)
model = GCN(input_dim, hidden_dims, output_dim)

# Debug information
print("Model architecture:\\n", model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# %%
# %% [set optimizer]
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer initialized:", optimizer)

# %%
# %% [define loss]
criterion = torch.nn.BCEWithLogitsLoss()
print("Loss function defined:", criterion)

# %%
# %% [debug forward]
# Test model forward pass on dummy data
dummy_nodes = 5
dummy_edges = 10
dummy_x = torch.randn(dummy_nodes, input_dim)
# Random edge_index for dummy graph
source = torch.randint(0, dummy_nodes, (dummy_edges,))
target = torch.randint(0, dummy_nodes, (dummy_edges,))
dummy_edge_index = torch.stack([source, target], dim=0)
out = model(dummy_x, dummy_edge_index)
print("Forward pass on dummy data output shape:", out.shape)

# %% [markdown]
# # Обучение модели

# %%
# %% [model training]
import tensorflow as tf
from stellargraph.mapper import FullBatchNodeGenerator

# Define the full-batch generator with self-loops
generator = FullBatchNodeGenerator(
    G,
    method="self_loops",
    sparse=False
)

# %% [Keras GCN model definition]
from stellargraph.layer.gcn import GCN as StellarGCN
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# Build and compile the StellarGraph GCN model
layer_sizes = [32, 16, 16, 16, 4]
gcn_keras = StellarGCN(
    layer_sizes=layer_sizes,
    activations=["relu"] * len(layer_sizes),
    generator=generator,
    dropout=0.5,
)
x_inp, x_out = gcn_keras.in_out_tensors()
prediction = Dense(units=2, activation="softmax")(x_out)
model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss=categorical_crossentropy,
    metrics=["acc"],
)
model.summary()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Prepare node labels by merging 'from_scam' and 'to_scam'
node_labels_from = df.groupby('from_address')['from_scam'].max()
node_labels_to   = df.groupby('to_address')['to_scam'].max()
node_labels      = pd.concat([node_labels_from, node_labels_to], axis=1).max(axis=1)

# One-hot encode labels
node_targets = pd.get_dummies(node_labels.astype(int))

# Stratified split into train, validation, and test sets using sklearn
from sklearn.model_selection import train_test_split

# First split: 75% train+val, 25% test
train_val_nodes, test_nodes = train_test_split(
    node_labels.index,
    test_size=0.25,
    stratify=node_labels,
    random_state=42
)
# Second split: of train_val into 2/3 train (50% total) and 1/3 val (25% total)
train_nodes, val_nodes = train_test_split(
    train_val_nodes,
    test_size=1/3,
    stratify=node_labels.loc[train_val_nodes],
    random_state=42
)

train_targets = node_targets.loc[train_nodes].values
val_targets   = node_targets.loc[val_nodes].values
test_targets  = node_targets.loc[test_nodes].values

# Create data generators
train_gen = generator.flow(train_nodes, train_targets)
val_gen   = generator.flow(val_nodes,   val_targets)
test_gen  = generator.flow(test_nodes,  test_targets)

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate on test set
loss, accuracy = model.evaluate(test_gen)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Detailed metrics
# Get model predictions
y_pred = model.predict(test_gen)

# Squeeze out the batch dimension if present (full-batch sequence yields shape (1, N, classes))
if isinstance(y_pred, np.ndarray) and y_pred.ndim == 3:
    y_pred = y_pred[0]

# Compute predicted classes and true labels
preds = np.argmax(y_pred, axis=1)
truths = np.argmax(test_targets, axis=1)

# Print classification metrics
print("Recall:", recall_score(truths, preds))
print("Precision:", precision_score(truths, preds))
print("F1 score:", f1_score(truths, preds))
print("Confusion Matrix:\n", confusion_matrix(truths, preds))



