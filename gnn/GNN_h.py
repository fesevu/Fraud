# %% [markdown]
# # Установка зависимостей

# %%
# 0. Обновление pip и инструментов сборки
import pathlib
import zipfile
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import os
%pip install - U pip setuptools wheel

# 4. Установка утилит
%pip install - U scikit-learn pandas networkx matplotlib rich tqdm
%pip install graphlime tensorflow

%pip install torch == 2.6.0 torchvision torchaudio - -index-url https: // download.pytorch.org/whl/cu126
%pip install torch_geometric
%pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv - f https: // data.pyg.org/whl/torch-2.6.0+cu126.html


# %% [markdown]
# # Загрузка датасета

# %%
# @title 📂 Загрузка датасета (идемпотентно)

# Папка, куда будет сохранён репозиторий
data_dir = 'data/'


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
!git clone https: // github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git {data_dir}


zip_path = "data/Dataset.zip"
extract_path = pathlib.Path("data/unpacked")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(extract_path)
print("✅ Распаковка завершена")

# %%
# @title 🧹 Чтение CSV
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
to_counts = df['to_scam'].value_counts().rename({0: 'non_scam', 1: 'scam'})

print("Sender (from_scam):")
print(from_counts)
print("\nRecipient (to_scam):")
print(to_counts)

# Если вас интересуют транзакции, где есть хоть одна пометка scam:
df['any_scam'] = ((df['from_scam'] == 1) | (df['to_scam'] == 1)).astype(int)
any_counts = df['any_scam'].value_counts().rename(
    {0: 'no_scam', 1: 'has_scam'})
print("\nTransactions with any scam flag:")
print(any_counts)

# %%

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
# desired_n = 2617

# # По from_scam
# df_from_pos = df[df['from_scam'] == 1]
# df_from_neg = df[df['from_scam'] == 0].sample(n=desired_n, random_state=42)
# df_sampled_from_exact = pd.concat([df_from_pos, df_from_neg]) \
#                            .sample(frac=1, random_state=42) \
#                            .reset_index(drop=True)

# # По to_scam
# df_to_pos = df[df['to_scam'] == 1].sample(n=desired_n, random_state=42)
# df_to_neg = df[df['to_scam'] == 0].sample(n=desired_n, random_state=42)
# df_sampled_to_exact = pd.concat([df_to_pos, df_to_neg]) \
#                          .sample(frac=1, random_state=42) \
#                          .reset_index(drop=True)

# print("\nExact 2617 each for from_scam:")
# print(df_sampled_from_exact['from_scam'].value_counts())
# print("\nExact 2617 each for to_scam:")
# print(df_sampled_to_exact['to_scam'].value_counts())


# Предполагаем, что df уже загружен и очищен от NaN/дубликатов
desired_n = 2617

# 1) Баланс по from_scam
pos_from = df[df["from_scam"] == 1]
neg_from = df[df["from_scam"] == 0].sample(n=desired_n, random_state=42)

df_from_balanced = (
    pd.concat([pos_from, neg_from])
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

print("FROM balance:\n", df_from_balanced["from_scam"].value_counts())

# 2) Баланс по to_scam без пересечений с df_from_balanced
pos_to = df[df["to_scam"] == 1].sample(n=desired_n, random_state=42)

# из кандидатов на negative-to исключаем уже взятые в df_from_balanced
neg_to_candidates = df[df["to_scam"] == 0]
neg_to_candidates = neg_to_candidates.loc[
    ~neg_to_candidates.index.isin(df_from_balanced.index)
]

neg_to = neg_to_candidates.sample(n=desired_n, random_state=42)

df_to_balanced = (
    pd.concat([pos_to, neg_to])
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

print("\nTO balance:\n", df_to_balanced["to_scam"].value_counts())

# 3) Проверка пересечений индексов между двумя наборами
intersection = set(df_from_balanced.index) & set(df_to_balanced.index)
print("\nПересечение индексов:", intersection)

# 4) Если нужен единый датасет — объединяем и удаляем дубликаты
df_balanced = (
    pd.concat([df_from_balanced, df_to_balanced])
      .drop_duplicates()
      .reset_index(drop=True)
)
print("\nИтоговый сбалансированный датасет:", df_balanced.shape)

df = df_balanced

# %%
# %% [feature engineering]

# 0) Предполагаем, что df уже загружен и очищен от NaN/дубликатов

# 1) Извлекаем год из ISO-строки block_timestamp
df['block_year'] = df['block_timestamp'].str[:4].astype(int)

# 2) Категории value (3 квантиля)
df['value_cat'] = pd.qcut(df['value'], q=3, labels=False)

# 3) Считаем node-признаки на уровне адреса
# -------------------------------------------------------------------
# Собираем все вхождения адрес → год первой транзакции
addr_years = pd.concat([
    df[['from_address', 'block_year']].rename(
        columns={'from_address': 'address'}),
    df[['to_address',   'block_year']].rename(
        columns={'to_address':   'address'}),
], ignore_index=True)
reg_year = addr_years.groupby('address')['block_year'].min().rename('reg_year')

# Собираем все вхождения адрес → максимальное значение транзакции
addr_vals = pd.concat([
    df[['from_address', 'value']].rename(columns={'from_address': 'address'}),
    df[['to_address',   'value']].rename(columns={'to_address':   'address'}),
], ignore_index=True)
max_value = addr_vals.groupby('address')['value'].max().rename('max_value')

# Склеиваем node-таблицу
node_features_df = (
    pd.concat([reg_year, max_value], axis=1)
      .reset_index()               # index → address
)
node_features_df.columns = ['address', 'reg_year', 'max_value']

# 4) Считаем edge-признаки, мерджим туда годы работы из node-таблицы
# -------------------------------------------------------------------
# Сначала добавим «from_» признаки
edge_df = df[[
    'from_address', 'to_address',
    'gas', 'gas_price',
    'receipt_cumulative_gas_used', 'receipt_gas_used',
    'block_number', 'value_cat',
    'from_scam', 'to_scam'
]].copy()

# прицепляем год регистрации и max_value, затем считаем years_in_operation
edge_df = edge_df.merge(
    node_features_df.rename(columns={
        'address': 'from_address',
        'reg_year': 'from_reg_year',
        'max_value': 'from_max_value'
    }),
    on='from_address', how='left'
)
edge_df = edge_df.merge(
    node_features_df.rename(columns={
        'address': 'to_address',
        'reg_year': 'to_reg_year',
        'max_value': 'to_max_value'
    }),
    on='to_address', how='left'
)

current_year = edge_df['block_year'].max(
) if 'block_year' in edge_df else df['block_year'].max()
# Но нам нужен block_year в edge_df для расчёта лет в эксплуатации:
edge_df['block_year'] = df['block_year']  # просто копируем
edge_df['from_years_in_operation'] = current_year - edge_df['from_reg_year']
edge_df['to_years_in_operation'] = current_year - edge_df['to_reg_year']

# 5) Итоговые списки признаков
edge_feature_cols = [
    'gas', 'gas_price',
    'receipt_cumulative_gas_used', 'receipt_gas_used',
    'block_number', 'value_cat',
    'from_years_in_operation', 'to_years_in_operation'
]

node_feature_cols = [
    'address',    # ID узла
    'reg_year',   # год регистрации
    'max_value',  # макс. сумма транзакций
]

response_cols = ['from_scam', 'to_scam']

# 6) Проверим, что всё на месте
print("Edge DataFrame shape:", edge_df.shape)
print("Edge feature columns:", edge_feature_cols)
print("Node DataFrame shape:", node_features_df.shape)
print("Node feature columns:", node_feature_cols)
print("Response columns:", response_cols)

# При желании можно отделить чисто две таблицы:
node_df = node_features_df.set_index('address')
edge_df = edge_df[['from_address', 'to_address'] +
                  edge_feature_cols + response_cols]

# %% [markdown]
# # Определяем модель

# %%
print("DataFrame columns:", df.columns.tolist())

# %%
# ───────────── Подготовка меток и разбиение ─────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) метка «хотя бы одна отметка scam у узла»
from_label = df.groupby("from_address")["from_scam"].max()
to_label = df.groupby("to_address")["to_scam"].max()

node_label = (
    from_label.reindex(node_df.index, fill_value=0).astype(int)
    | to_label.reindex(node_df.index, fill_value=0).astype(int)
)

# 2) train / val / test = 50 / 25 / 25 %, стратифицированно
train_ids, rest_ids = train_test_split(
    node_df.index,
    train_size=0.5,
    stratify=node_label,
    random_state=SEED,
)
val_ids, test_ids = train_test_split(
    rest_ids,
    train_size=0.5,
    stratify=node_label.reindex(rest_ids),
    random_state=SEED,
)

print(
    f"train / val / test sizes: {len(train_ids)} | {len(val_ids)} | {len(test_ids)}")

# %%


# 1) ─────────────── Подготовка графа ──────────────────────────
#   node_df     – как в вашем пайплайне (index = address)
#   edge_df     – столбцы ["from_address", "to_address", …]
#   node_label  – Series(address → 0/1, «scam‑узел»)

addr2idx = {addr: i for i, addr in enumerate(node_df.index)}
num_nodes = len(addr2idx)

# Node‑признаки: возьмём два, что вы уже сделали (reg_year, max_value)
x = torch.tensor(
    node_df[["reg_year", "max_value"]].values,
    dtype=torch.float32,
)

# Список рёбер (направленный граф, как в исходном варианте)
edge_index = torch.tensor(
    [
        [addr2idx[s], addr2idx[t]]
        for s, t in zip(edge_df["from_address"], edge_df["to_address"])
        if s in addr2idx and t in addr2idx
    ],
    dtype=torch.long,
).t().contiguous()

# Добавим self‑loops (помогает сходиться GraphSAGE)
edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

# Массив меток узлов
y = torch.tensor(node_label.values, dtype=torch.long)

# ─ Split на train / val / test (адреса у вас уже выбраны) ────


def mask_from_ids(id_list):
    m = torch.zeros(num_nodes, dtype=torch.bool)
    m[[addr2idx[a] for a in id_list]] = True
    return m


train_mask = mask_from_ids(train_ids)
val_mask = mask_from_ids(val_ids)
test_mask = mask_from_ids(test_ids)

data = Data(
    x=x, edge_index=edge_index, y=y,
    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
)

# 2) ─────────────── Определение модели ───────────────────────


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_ch, 4),
            SAGEConv(4, 32, aggr="mean"),
            SAGEConv(32, 16, aggr="mean"),
            SAGEConv(16, 16, aggr="mean"),
            SAGEConv(16, 16, aggr="mean"),
            SAGEConv(16,  4, aggr="mean"),
        ])
        self.lin = torch.nn.Linear(4, 2)          # 2 класса
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(
                x, p=self.dropout, training=self.training
            )
        return torch.log_softmax(self.lin(x), dim=-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_ch=data.num_features).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = torch.nn.NLLLoss()

print(model)

# %% [markdown]
# # Обучение модели

# %%
# 3) ─────────────── Обучение ─────────────────────────────────
EPOCHS = 50
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # валидация
    model.eval()
    with torch.no_grad():
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] ==
                     data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]
                   ).float().mean().item()

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | loss {loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

# 4) ─────────────── Тест + отчёты ────────────────────────────
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_acc = (pred[data.test_mask] ==
                data.y[data.test_mask]).float().mean().item()

print(f"\nTEST accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(
    classification_report(
        data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), digits=4
    )
)
print("Confusion matrix:\n",
      confusion_matrix(
          data.y[data.test_mask].cpu(), pred[data.test_mask].cpu()
      ))
