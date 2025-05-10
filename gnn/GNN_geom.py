# %%
%pip uninstall torch torch_geometric torch_scatter torch_sparse pyg_lib torch_cluster torch_spline_conv -y
# 2.1 PyTorch (CPU) – подмените индекс, если нужна CUDA
%pip install torch==2.3.0

# 2.2 PyG: с 2.3+ внешних библиотек почти нет, ставим одной строкой
%pip install torch_geometric
%pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html


# 2.4 остальное
%pip install xgboost pandas matplotlib graphviz scikit-learn tqdm numpy networkx seaborn


# %%
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected
import math
import torch_cluster

print(torch.__version__)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# --- после расчёта pr и clust --------------------------------------------
print('⏳ Calculating extra graph features…')

# 1) Betweenness (approx)
btw = nx.betweenness_centrality(G, k=10_000, seed=42, normalized=True)

# 2) Weakly-connected component size
G_u = G.to_undirected()
wcc = {n: 0 for n in G}
for comp in nx.connected_components(G_u):
    size = len(comp)
    for n in comp:
        wcc[n] = size

# 3) Min amounts
send_min = {n: math.inf for n in G}
recv_min = {n: math.inf for n in G}
for u, v, d in G.edges(data=True):
    amt = d['amount']
    send_min[u] = min(send_min[u], amt)
    recv_min[v] = min(recv_min[v], amt)
# заменяем inf → 0.0 (узлы без операций)
send_min = {n: 0.0 if math.isinf(v) else v for n, v in send_min.items()}
recv_min = {n: 0.0 if math.isinf(v) else v for n, v in recv_min.items()}

# --- записываем все новые фичи в вершины ----------------------------------
for n in G.nodes():
    G.nodes[n].update({
        'btw_centr':   btw[n],
        'wcc_size':    wcc[n],
        'send_min':    send_min[n],
        'recv_min':    recv_min[n],
    })

# %%
 # --- перед обучением Node2Vec: создаём маппинг id → idx -------------------
id2idx = {n: i for i, n in enumerate(G.nodes())}

# --- строим edge_index по числовым индексам ------------------------------
edges = list(G.edges())
row = [id2idx[u] for u, v in edges]
col = [id2idx[v] for u, v in edges]
edge_index = torch.tensor([row, col], dtype=torch.long)
edge_index = to_undirected(edge_index).contiguous()

# --- теперь обучаем Node2Vec на неориентированном графе --------------------
print('⏳ Training Node2Vec…')
n2v = Node2Vec(
    edge_index=edge_index,
    num_nodes=G.number_of_nodes(),
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=5,
    num_negative_samples=1,
    sparse=True,
    p=1, q=1
).to(device)

loader = n2v.loader(batch_size=1024, shuffle=True)
opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)

n2v.train()
for epoch in range(25):  # можете увеличить число эпох
    total_loss = 0
    for pos_rw, neg_rw in loader:
        opt.zero_grad()
        loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss {total_loss:.4f}')

# %%
# --- извлекаем эмбеддинги и записываем обратно в G ------------------------
z = n2v.embedding.weight.detach().cpu()
for n, idx in id2idx.items():
    G.nodes[n]['n2v'] = z[idx]

# --- при формировании data.x ---------------------------------------------
# Наконец, собираем x
num_attr_keys = [
    'in_deg','out_deg','sent_sum','recv_sum','net_sum',
    'pagerank','clustering',
    'btw_centr','wcc_size','send_min','recv_min'
]

for n in G.nodes():
    # проверяем, что всё есть
    missing = [k for k in num_attr_keys if k not in G.nodes[n]]
    if missing:
        raise RuntimeError(f"У узла {n} нет фичей {missing}")
    base = [float(G.nodes[n][k]) for k in num_attr_keys]
    emb  = G.nodes[n]['n2v'].tolist()           # 128-мерный эмбеддинг
    G.nodes[n]['x'] = torch.tensor(base + emb, dtype=torch.float)

print('Пример фич узла:', list(G.nodes(data=True))[0])

# %%
# 🔧 Делаем так, чтобы у всех узлов была метка
nx.set_node_attributes(G, -1, "label")        # сначала всем ставим -1
for _, row in acc_df.iterrows():              # затем переопределяем тем, что есть в accounts.csv
    G.nodes[row.id]["label"] = int(bool(row.label))   

# %%
# после формирования всех численных признаков, до отправки в PyG
from sklearn.preprocessing import StandardScaler
X = np.stack([G.nodes[n]['x'].numpy() for n in G.nodes()])
scaler = StandardScaler()
X = scaler.fit_transform(X)
for n, vec in zip(G.nodes(), X):
    G.nodes[n]['x'] = torch.tensor(vec, dtype=torch.float)

# %% [markdown]
# ## 🔄 Конвертация в `torch_geometric.data.Data`

# %%
num_attr_keys = ['in_deg','out_deg','sent_sum','recv_sum','net_sum','pagerank','clustering']

for n in G.nodes():
    G.nodes[n]['x'] = torch.tensor([float(G.nodes[n][k]) for k in num_attr_keys], dtype=torch.float)

data = from_networkx(G, group_node_attrs=['x'])
data.y = torch.tensor([G.nodes[n].get('label', -1) for n in G.nodes()], dtype=torch.long)

print("Num features:", data.num_node_features)  
print(data)

# %%
import torch

# 1) Список «старых» фичей уже в data.x: shape [N,7]
X_base = data.x                # float32, device может быть CPU или CUDA

# 2) Достаём остальные скалярные фичи и приводим к форме [N,1]
btw      = data.btw_centr.view(-1,1)   # [N] → [N,1]
wcc      = data.wcc_size.view(-1,1)
send_min = data.send_min.view(-1,1)
recv_min = data.recv_min.view(-1,1)

# 3) Node2Vec: shape [N,128]
X_n2v    = data.n2v               # уже [N,128]

# 4) Конкатенируем всё в один [N, 7+4+128 = 139]
data.x = torch.cat([X_base, btw, wcc, send_min, recv_min, X_n2v], dim=1)

print("Now num features:", data.x.size(1))  # должно быть 139

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) выгружаем data.x в NumPy
X = data.x.cpu().numpy()   # shape [N,139]

# 2) лог-трансформации неотрицательных признаков
log_idxs = [0,1,2,3,7,8,9,10]
X[:, log_idxs] = np.log1p(X[:, log_idxs])

# 3) чистим NaN/Inf → конечные числа
#    nan → 0.0, +inf → max_float32, -inf → min_float32
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=np.finfo(np.float32).max,
    neginf=np.finfo(np.float32).min
)

# (можно проверить, что теперь все конечные)
assert np.isfinite(X).all(), "Есть ещё не-конечные элементы!"

# 4) стандартизация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5) обратно в TorchTensor
import torch
data.x = torch.tensor(X, dtype=torch.float32).to(device)

# %% [markdown]
# ## ✂️ Train / Val / Test сплит

# %%
import hashlib
import numpy as np
import torch

# 1) Собираем список узлов в порядке G.nodes()
nodes = list(G.nodes())

# 2) Хешируем каждый ID (md5 → целое), берём mod 100 → [0..99]
hash_vals = np.array([
    int(hashlib.md5(str(n).encode()).hexdigest(), 16) % 100
    for n in nodes
])

# 3) Порог: [0,59] → train, [60,79] → val, [80,99] → test
train_mask = torch.tensor(hash_vals < 60, dtype=torch.bool)
val_mask   = torch.tensor((hash_vals >= 60) & (hash_vals < 80), dtype=torch.bool)
test_mask  = torch.tensor(hash_vals >= 80, dtype=torch.bool)

# после заполнения data.y (с -1 для неметочных) и до создания loader’ов
label_mask = data.y != -1               # булевый тензор: True там, где есть 0/1
train_mask = (hash_vals < 60)  & label_mask.cpu().numpy()
val_mask   = ((hash_vals >= 60) & (hash_vals < 80)) & label_mask.cpu().numpy()
test_mask  = (hash_vals >= 80) & label_mask.cpu().numpy()

# преобразуем обратно в torch.bool
data.train_mask = torch.tensor(train_mask, dtype=torch.bool, device=device)
data.val_mask   = torch.tensor(val_mask,   dtype=torch.bool, device=device)
data.test_mask  = torch.tensor(test_mask,  dtype=torch.bool, device=device)

# Проверим доли
print(f"Train: {train_mask.sum().item()/len(nodes):.2%}, "
      f"Val: {val_mask.sum().item()/len(nodes):.2%}, "
      f"Test: {test_mask.sum().item()/len(nodes):.2%}")

train_loader = NeighborLoader(
    data,
    input_nodes=data.train_mask,        # корневые узлы для train
    num_neighbors=[200, 200],             # сколько соседей на каждом слое
    batch_size=1024,
    shuffle=True,
)
val_loader = NeighborLoader(
    data,
    input_nodes=data.val_mask,
    num_neighbors=[200, 200],
    batch_size=1024,
    shuffle=False,
)
test_loader = NeighborLoader(
    data,
    input_nodes=data.test_mask,
    num_neighbors=[200, 200],
    batch_size=1024,
    shuffle=False,
)


# %%
OUT_PATH = Path('./artifacts/eth_graph.pt')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, OUT_PATH)
print(f'Сериализовано в {OUT_PATH.resolve()}')

# %% [markdown]
# ## 🧠 Определение GNN‑модели (GCN)

# %%
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# 1) Определяем GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, out_ch)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.dropout2(x)
        return x

# %% [markdown]
# ## 🏃‍♂️ Обучение и валидация

# %%
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

labels = data.y[data.y != -1]
counts = torch.bincount(labels)
weight = torch.tensor([1.0, counts[0].float() / counts[1].float()], device=device)
print(f"Class weights: normal={weight[0]:.2f}, fraud={weight[1]:.2f}")

# 3) Инициализируем модель, optimizer, criterion (с весами из шага 2)
model = GraphSAGE(
    in_ch=data.num_node_features,
    hidden_ch=256,
    out_ch=int(data.y.max().item())+1,
    dropout=0.45
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss(weight=weight)

# 4) Цикл обучения
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        y = batch.y[:batch.batch_size]
        mask = y != -1                    # True для валидных меток
        if mask.sum() == 0:
            continue
        # только первые batch.batch_size предсказаний — для корневых узлов
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
    return total_loss / int(data.train_mask.sum())

@torch.no_grad()
def eval_loader(loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        logits = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        # Только размеченные
        mask = (y != -1)
        if mask.sum() == 0:
            continue

        preds = logits.argmax(dim=1)
        correct += int((preds[mask] == y[mask]).sum())
        total   += int(mask.sum())

    return correct / total if total > 0 else 0.0

num_epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs     = [], []

# Функция для вычисления средней потери на валидации
@torch.no_grad()
def eval_loss(loader):
    model.eval()
    total_loss, total = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        logits = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]
        mask = (y != -1)
        if mask.sum() == 0:
            continue
        loss = criterion(logits[mask], y[mask])
        total_loss += loss.item() * mask.sum().item()
        total += mask.sum().item()
    return total_loss / total if total > 0 else 0.0

scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',
                              factor=0.5,
                              patience=3,
                              verbose=True)

best_val_loss = float('inf')
patience_cnt = 0
early_stop_patience = 7

for epoch in range(1, num_epochs+1):
    loss = train_epoch()
    train_acc = eval_loader(train_loader)
    val_loss  = eval_loss(val_loader)
    val_acc   = eval_loader(val_loader)

    train_losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Шедулер по валидационной потере
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= early_stop_patience:
            print(f"Early stopping на эпохе {epoch}")
            break

    print(f'Epoch {epoch:02d} | Train Loss {loss:.4f} | Val Loss {val_loss:.4f} '
          f'| Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}')


test_acc = eval_loader(test_loader)
print(f'✅ Test Accuracy: {test_acc:.3f}')

# %%

# ─── Визуализация кривых обучения ───────────────────────────────────────────────
import matplotlib.pyplot as plt

epochs = list(range(1, len(train_losses) + 1))  # вместо num_epochs
plt.figure(figsize=(8,4))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Val Loss')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs,   label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 📊 Визуализация результатов
# Вычислим метрики на тестовой выборке и отобразим матрицу ошибок.

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

model.eval()
all_preds, all_labels = [], []
for batch in test_loader:
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index)
    logits = out[:batch.batch_size]
    y = batch.y[:batch.batch_size]
    mask = (y != -1)

    all_preds.append(logits.argmax(dim=1)[mask].cpu())
    all_labels.append(y[mask].cpu())

y_true = torch.cat(all_labels).numpy()
y_pred = torch.cat(all_preds).numpy()
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
# %% 
# ## 🕵️‍♂️ Интерпретация предсказаний с GNNExplainer (inline, с именами фичей)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer

# 1) Найти первый корректно классифицированный узел
model.eval()
target_node = None
for batch in test_loader:
    batch = batch.to(device)
    out   = model(batch.x, batch.edge_index)
    preds = out[:batch.batch_size].argmax(dim=1)
    y     = batch.y[:batch.batch_size]
    mask  = (y != -1) & (preds == y)
    if mask.any():
        idx_local   = mask.nonzero(as_tuple=True)[0][0].item()
        target_node = batch.n_id[idx_local].item()
        break

print(f'🔍 Объясняем решение для узла {target_node}')

# 2) Подготовить Explainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
)

# 3) Сгенерировать Explanation
explanation = explainer(
    x=data.x, 
    edge_index=data.edge_index, 
    index=target_node,
)

# 4) Бар-граф топ-10 фич с человеческими названиями
feat_names = [
    'in_deg','out_deg','sent_sum','recv_sum','net_sum',
    'pagerank','clustering',
    'btw_centr','wcc_size','send_min','recv_min'
] + [f'n2v_{i}' for i in range(128)]

# расплющиваем в 1D-массив
feat_imp = explanation.node_mask.cpu().numpy().flatten()
top_k    = 10

# теперь argsort вернёт одномерный список индексов
idxs = np.argsort(feat_imp)[-top_k:][::-1].tolist()
vals = feat_imp[idxs]
names = [feat_names[i] for i in idxs]

plt.figure(figsize=(8,4))
plt.barh(names, vals)
plt.xlabel('Importance')
plt.title(f'Top {top_k} Feature Importances for node {target_node}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 5) Визуализация объясняющего субграфа как на вашем примере

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Достаём directed edge-mask и edge_index
edge_mask = explanation.edge_mask.cpu().numpy().flatten()
E = data.edge_index.cpu().numpy()
mask = edge_mask > np.percentile(edge_mask, 100 - 20)  # топ-20 самых важных рёбер

# Формируем DiGraph из наиболее важных рёбер
sub_edges = list(zip(E[0][mask], E[1][mask]))
G_sub = nx.DiGraph()
G_sub.add_edges_from(sub_edges)

# Достаём layout
pos = nx.spring_layout(G_sub, seed=42)

# Рисуем всё inline
fig, ax = plt.subplots(figsize=(6,6))
# рамка
ax.add_patch(Rectangle(
    ( -1.05, -1.05),   # чуть-чуть за границы layout [-1..1]
    2.1, 2.1,
    fill=False, lw=1.5, edgecolor='black'
))

# узлы
nx.draw_networkx_nodes(
    G_sub, pos,
    ax=ax,
    node_size=800,
    node_color='white',
    edgecolors='black',
    linewidths=1.2
)

# стрелки
nx.draw_networkx_edges(
    G_sub, pos,
    ax=ax,
    arrowstyle='-|>',
    arrowsize=12,
    width=1.2,
    edge_color='black',
    connectionstyle='arc3,rad=0.1'
)

# подписи внутри кружков
nx.draw_networkx_labels(
    G_sub, pos,
    ax=ax,
    labels={n:str(n) for n in G_sub.nodes()},
    font_size=10
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')
plt.title(f'Explanation Subgraph for node {target_node}', pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 💾 Сохранение обученной модели

# %%
MODEL_PATH = Path('./checkpoints/gcn_model.pt')
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f'Модель сохранена в {MODEL_PATH.resolve()}')

# %%
import torch
import pickle
import numpy as np
from pathlib import Path

# 1) Папка для артефактов
artifacts_dir = Path("./artifacts")
artifacts_dir.mkdir(exist_ok=True)

# 2) Внутри цикла обучения (при early stopping) сохраняем лучшие весa
best_val_loss = float('inf')
for epoch in range(1, num_epochs+1):
    train_epoch()
    val_loss = eval_loss(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            artifacts_dir / "gcn_best_weights.pt"
        )
        print(f"[Epoch {epoch}] Сохранили лучшие веса (val_loss={val_loss:.4f})")

# 3) После окончания обучения сохраняем:
# 3.1) финальные веса
torch.save(
    model.state_dict(),
    artifacts_dir / "gcn_final_weights.pt"
)
print("Финальные веса сохранены в artifacts/gcn_final_weights.pt")

# 3.2) весь объект модели (если нужно удобно загрузить без объявления класса)
torch.save(
    model,
    artifacts_dir / "gcn_model_full.pt"
)
print("Полная модель сохранена в artifacts/gcn_model_full.pt")

# 3.3) scaler (StandardScaler из sklearn)
with open(artifacts_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler сохранён в artifacts/scaler.pkl")

# 3.4) список фичей в том же порядке, что вы собирали в x
#      возьмите ваш DATA_FEATS список (или просто data.num_node_features названий)
feature_list = feat_names  # например ['in_degree','out_degree',…,'wcc_size']
with open(artifacts_dir / "features_list.pkl", "wb") as f:
    pickle.dump(feature_list, f)
print("Список признаков сохранён в artifacts/features_list.pkl")

# 4) Сохранение эмбеддингов (Node2Vec или финальных x-векторов)
#    — либо берём z = n2v.embedding.weight.detach().cpu().numpy()
#    — либо ваши итоговые node representations data.x.cpu().numpy()
embeddings = n2v.embedding.weight.detach().cpu().numpy()
np.save(artifacts_dir / "node2vec_embeddings.npy", embeddings)
print("Эмбеддинги сохранены в artifacts/node2vec_embeddings.npy")

# %% [markdown]
# # Загрузка модели

# %%
# 1) Модель и весa
model = GraphSAGE(...)    # объявить ту же архитектуру
model.load_state_dict(torch.load("./artifacts/gcn_best_weights.pt"))
model.eval()

# или, если вы сохраняли весь объект:
model = torch.load("./artifacts/gcn_model_full.pt")
model.eval()

# 2) Scaler
with open("./artifacts/scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# 3) Список фичей
with open("./artifacts/features_list.pkl","rb") as f:
    feature_list = pickle.load(f)

# 4) Эмбеддинги
embeddings = np.load("./artifacts/node2vec_embeddings.npy")


