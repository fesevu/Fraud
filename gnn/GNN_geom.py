# %%
%pip uninstall pyg-lib torch-sparse -y
# 2.1 PyTorch (CPU) – подмените индекс, если нужна CUDA
%pip install torch==2.6.0 torchvision torchaudio

# 2.2 PyG: с 2.3+ внешних библиотек почти нет, ставим одной строкой
%pip install torch_geometric 
%pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html


# 2.4 остальное
%pip install xgboost pandas matplotlib graphviz scikit-learn tqdm numpy networkx seaborn

# 2.5 если drawSchema выдаёт ошибку “Graphviz executable not found”:
#   Ubuntu: sudo apt-get install graphviz
#   macOS:  brew install graphviz

# %%
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

print(torch.__version__)

# %% [markdown]
# # Чтение датасета

# %%
from pathlib import Path
ACCOUNTS_CSV = Path('./tmp/Ethereum/account.csv')      # обновите при необходимости
TXS_CSV      = Path('./tmp/Ethereum/transaction.csv')   # обновите при необходимости

assert ACCOUNTS_CSV.exists(), f'{ACCOUNTS_CSV} not found'
assert TXS_CSV.exists(), f'{TXS_CSV} not found'

# %%
acc_df = pd.read_csv(ACCOUNTS_CSV, header=None, names=['id', 'label'])
tx_df  = pd.read_csv(TXS_CSV, header=None, names=['src', 'dst', 'amount', 'timestamp'])

display(acc_df.head())
display(tx_df.head())

# %% [markdown]
# ## 🏗️ Сборка графа

# %%
G = nx.DiGraph()

# add nodes
for _, row in acc_df.iterrows():
    G.add_node(row.id, label=int(bool(row.label)))

# add edges
for _, row in tx_df.iterrows():
    G.add_edge(row.src, row.dst, amount=float(row.amount), ts=float(row.timestamp))

print(f'Граф содержит {G.number_of_nodes():,} узлов и {G.number_of_edges():,} рёбер')

# %%
# Структурные признаки
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

# Финансовые суммы
sent_sum = {n: 0.0 for n in G.nodes()}
recv_sum = {n: 0.0 for n in G.nodes()}
for u, v, d in G.edges(data=True):
    amt = d['amount']
    sent_sum[u] += amt
    recv_sum[v] += amt

# PageRank и кластерный коэффициент
pr = nx.pagerank(G, alpha=0.85)
clust = nx.clustering(G.to_undirected())

# Записываем фичи в граф
for n in G.nodes():
    G.nodes[n].update({
        'in_deg':      in_deg.get(n, 0),
        'out_deg':     out_deg.get(n, 0),
        'sent_sum':    sent_sum[n],
        'recv_sum':    recv_sum[n],
        'net_sum':     sent_sum[n] - recv_sum[n],
        'pagerank':    pr[n],
        'clustering':  clust[n],
    })

print('Пример фич узла:', list(G.nodes(data=True))[0])

# %%
# 🔧 Делаем так, чтобы у всех узлов была метка
nx.set_node_attributes(G, -1, "label")        # сначала всем ставим -1
for _, row in acc_df.iterrows():              # затем переопределяем тем, что есть в accounts.csv
    G.nodes[row.id]["label"] = int(bool(row.label))   

# %% [markdown]
# ## 🔄 Конвертация в `torch_geometric.data.Data`

# %%
num_attr_keys = ['in_deg','out_deg','sent_sum','recv_sum','net_sum','pagerank','clustering']

for n in G.nodes():
    G.nodes[n]['x'] = torch.tensor([float(G.nodes[n][k]) for k in num_attr_keys], dtype=torch.float)

data = from_networkx(G, group_node_attrs=['x'])
data.y = torch.tensor([G.nodes[n].get('label', -1) for n in G.nodes()], dtype=torch.long)

print(data)

# %% [markdown]
# ## ✂️ Train / Val / Test сплит

# %%
torch.manual_seed(42)
N = data.num_nodes
perm = torch.randperm(N)
n_train = int(0.6 * N)
n_val   = int(0.2 * N)

data.train_mask = torch.zeros(N, dtype=torch.bool)
data.val_mask   = torch.zeros(N, dtype=torch.bool)
data.test_mask  = torch.zeros(N, dtype=torch.bool)

data.train_mask[perm[:n_train]]           = True
data.val_mask[perm[n_train:n_train+n_val]] = True
data.test_mask[perm[n_train+n_val:]]      = True

print(f'Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}')

# %%
OUT_PATH = Path('./artifacts/eth_graph.pt')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, OUT_PATH)
print(f'Сериализовано в {OUT_PATH.resolve()}')

# %% [markdown]
# ## 🧠 Определение GNN‑модели (GCN)

# %%
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

print('GCN model ready')

# %% [markdown]
# ## 🏃‍♂️ Обучение и валидация

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_node_features, 64, int(data.y.max().item()) + 1).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

labeled = (data.y != -1)
perm = torch.randperm(int(labeled.sum()))
idx  = labeled.nonzero(as_tuple=False).view(-1)[perm]

n_train = int(0.6 * len(idx))
n_val   = int(0.2 * len(idx))

data.train_mask[:] = False
data.val_mask[:]   = False
data.test_mask[:]  = False

data.train_mask[idx[:n_train]]           = True
data.val_mask[idx[n_train:n_train+n_val]] = True
data.test_mask[idx[n_train+n_val:]]      = True

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def accuracy(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == data.y[mask]).sum().item()
    return correct / int(mask.sum())

for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        train_acc = accuracy(data.train_mask)
        val_acc = accuracy(data.val_mask)
        print(f'Epoch {epoch:03d} | Loss {loss:.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}')

test_acc = accuracy(data.test_mask)
print(f'✅ Test Accuracy: {test_acc:.3f}')

# %% [markdown]
# ## 📊 Визуализация результатов
# Вычислим метрики на тестовой выборке и отобразим матрицу ошибок.

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model.eval()
logits = model(data.x, data.edge_index)
preds  = logits.argmax(dim=1)

mask = (data.test_mask) & (data.y != -1)
y_true, y_pred = data.y[mask].cpu(), preds[mask].cpu()

print(classification_report(y_true, y_pred, digits=3))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %% [markdown]
# ## 🕵️‍♂️ Интерпретация предсказаний с **GNNExplainer**
# Выберем произвольный корректно классифицированный узел из тестовой выборки и посмотрим, какие рёбра и признаки были наиболее важны для модели.

# %%
from torch_geometric.explain import Explainer, GNNExplainer

# --------------------- 1. забираем подграф вокруг нужного адреса ---------------------
target_addr = "0x5093c4029acab3aff80140023099f5ec6ca7d52f"
# train_loader.fetch вернёт список Data-объектов; берём первый
data = train_loader.fetch([{"type": "Account", "primary_id": target_addr}])[0]

# --------------------- 2. включаем eval-режим модели ---------------------------------
gs.model.eval()                       # важно: ни dropout, ни BN-апдейтов

# --------------------- 3. настраиваем Explainer (новое API) --------------------------
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=gs.model,                          # ваша обученная GNN
    algorithm=GNNExplainer(epochs=200),      # 200 эпох достаточно
    explanation_type="model",                # объясняем предсказание модели
    node_mask_type="attributes",             # важность признаков узла
    edge_mask_type="object",                 # важность рёбер
    model_config=dict(
        mode="binary_classification",        # 0/1 метки
        task_level="node",
        return_type="raw",                   # модель выдаёт logits
    ),
)

# --------------------- 4. считаем объяснение -----------------------------------------
# Если подграф уже обрезан до одной вершины – index=None.
# Если там несколько вершин, укажите локальный индекс целевой (обычно 0).
explanation = explainer(
    x=data.x.float(), 
    edge_index=data.edge_index, 
    index=None,              # или 0 / int(<нужная_вершина>)
    target=data.y[0].item()  # какой класс объясняем
)

# --------------------- 5. визуализация ------------------------------------------------
explanation.visualize_feature_importance(
    top_k=len(train_loader.v_in_feats),
    feat_labels=train_loader.v_in_feats,
)
explanation.visualize_graph()  # покажет субграф и веса рёбер


# %% [markdown]
# ## 💾 Сохранение обученной модели

# %%
MODEL_PATH = Path('./checkpoints/gcn_model.pt')
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f'Модель сохранена в {MODEL_PATH.resolve()}')


