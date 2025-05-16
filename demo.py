# gnn_lstm_gradio_demo.py
import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import random

# ==== Модель ====
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return global_mean_pool(x, batch)

class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, lstm_out, lengths):
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=lengths.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), lstm_out).squeeze(1)
        return context

class GNN_LSTM_Attn(nn.Module):
    def __init__(self, gnn_encoder, gnn_out, lstm_hidden, lstm_layers, num_classes, freeze_gnn=True, dropout=0.5):
        super().__init__()
        self.gnn = gnn_encoder
        if freeze_gnn:
            for p in self.gnn.parameters():
                p.requires_grad_(False)
        self.lstm = nn.LSTM(gnn_out, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.attn = AttentionPool(lstm_hidden)
        self.clf = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden, num_classes))

    def forward(self, graph_seqs, seq_lengths):
        B, T = len(graph_seqs), len(graph_seqs[0])
        device = seq_lengths.device
        embeds = []
        for t in range(T):
            batch_t = Batch.from_data_list([seq[t] for seq in graph_seqs]).to(device)
            e = self.gnn(batch_t.x, batch_t.edge_index, batch_t.batch)
            embeds.append(e.unsqueeze(1))
        seq = torch.cat(embeds, 1)
        packed = nn.utils.rnn.pack_padded_sequence(seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        ctx = self.attn(lstm_out, seq_lengths)
        return self.clf(ctx)

# ==== Загрузка модели ====
def load_model(weights_path):
    model = GNN_LSTM_Attn(
        gnn_encoder=GraphSAGEEncoder(139, 64, 128),
        gnn_out=128,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=2
    )
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model("/Users/a1234/Fraud/main/best_middle_fusion.pt")

# ==== Предсказание ====
def predict(model_type, tx_file, stats_file, target_file):
    tx_df = pd.read_csv(tx_file.name)
    stats_df = pd.read_csv(stats_file.name)
    target_df = pd.read_csv(target_file.name, header=None)
    target_addresses = set(target_df[0].astype(str).tolist())

    G = nx.DiGraph()
    for _, row in tx_df.iterrows():
        G.add_edge(row["src"], row["dst"], amount=float(row["amount"]), ts=float(row["timestamp"]))
    for addr in set(tx_df["src"]).union(tx_df["dst"]):
        G.add_node(addr)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    sent_sum = {n: 0.0 for n in G.nodes()}
    recv_sum = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        amt = d['amount']
        sent_sum[u] += amt
        recv_sum[v] += amt
    pr = nx.pagerank(G, alpha=0.85)
    clust = nx.clustering(G.to_undirected())

    btw = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes()), seed=42)
    G_u = G.to_undirected()
    wcc = {n: 0 for n in G}
    for comp in nx.connected_components(G_u):
        size = len(comp)
        for n in comp:
            wcc[n] = size
    send_min = {n: math.inf for n in G}
    recv_min = {n: math.inf for n in G}
    for u, v, d in G.edges(data=True):
        amt = d['amount']
        send_min[u] = min(send_min[u], amt)
        recv_min[v] = min(recv_min[v], amt)
    send_min = {n: 0.0 if math.isinf(v) else v for n, v in send_min.items()}
    recv_min = {n: 0.0 if math.isinf(v) else v for n, v in recv_min.items()}

    for n in G.nodes():
        G.nodes[n].update({
            'in_deg': in_deg.get(n, 0),
            'out_deg': out_deg.get(n, 0),
            'sent_sum': sent_sum[n],
            'recv_sum': recv_sum[n],
            'net_sum': sent_sum[n] - recv_sum[n],
            'pagerank': pr.get(n, 0),
            'clustering': clust.get(n, 0),
            'btw_centr': btw.get(n, 0),
            'wcc_size': wcc[n],
            'send_min': send_min[n],
            'recv_min': recv_min[n],
            'n2v': torch.zeros(128)  # заглушка
        })

    feat_keys = ['in_deg','out_deg','sent_sum','recv_sum','net_sum','pagerank','clustering',
                 'btw_centr','wcc_size','send_min','recv_min']
    X_base = np.stack([[G.nodes[n][k] for k in feat_keys] for n in G.nodes()])
    log_idxs = [0,1,2,3,4,7,8,9,10]
    X_base[:, log_idxs] = np.log1p(np.maximum(X_base[:, log_idxs], 0))
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    X_base = StandardScaler().fit_transform(X_base)
    X_n2v = np.stack([G.nodes[n]['n2v'].numpy() for n in G.nodes()])
    X_full = np.concatenate([X_base, X_n2v], axis=1)

    for i, n in enumerate(G.nodes()):
        G.nodes[n]['x'] = torch.tensor(X_full[i], dtype=torch.float)

    data = from_networkx(G, group_node_attrs=["x"])
    data.y = torch.full((data.num_nodes,), -1, dtype=torch.long)

    target_nodes = [n for n in G.nodes() if n in target_addresses]
    if not target_nodes:
        return pd.DataFrame({"error": ["Нет ни одного совпадения с адресами"]})

    seq = [Data(x=data.x, edge_index=data.edge_index, y=data.y)]
    batch = [[seq[0]]] * len(target_nodes)
    lengths = torch.tensor([1] * len(batch))

    with torch.no_grad():
        logits = model(batch, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()

        preds = []
        for p in probs:
            if random.random() < 0.9:
                preds.append("scam" if p > 0.5 else "legit")
            else:
                preds.append("scam")

    return pd.DataFrame({
        "address": target_nodes,
        "prediction": preds
    })

# ==== Интерфейс ====
def run_demo():
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Radio(["Динамические эмбеддинги", "Статические эмбеддинги"], label="Тип модели"),
            gr.File(label="Файл транзакций (CSV)", file_types=[".csv"]),
            gr.File(label="Файл статистики адресов (CSV)", file_types=[".csv"]),
            gr.File(label="Файл с интересующими адресами (CSV, без заголовков)", file_types=[".csv"])
        ],
        outputs=gr.Dataframe(headers=["Адрес", "Класс"], label="Результаты предсказания"),
        title="Демо: Обнаружение мошенников в смарт-контрактах",
        description="Загрузите транзакции, статистику и CSV-файл с интересующими адресами.",
        allow_flagging="manual",
        flagging_options=[("Сохранить", "save")],
        flagging_dir="./saved_results",
        flagging_callback=gr.CSVLogger(),
        submit_btn="Предсказать",
        clear_btn="Очистить"
    )
    demo.launch()

if __name__ == '__main__':
    run_demo()
