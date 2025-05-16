# %% [markdown]
# # Берем готовый датасет lstm и с помощью bigquery собираем датасет для gnn

# %%
# Установить клиентскую библиотеку
%pip uninstall -y pandas
# свежий pip подтянет правильный wheel
%pip install --upgrade pip           
%pip install "pandas>=2.2.0"

%pip install google-cloud-bigquery


# %%
# Шаг 1: Авторизация (если не делал)
!gcloud auth application-default login --quiet

# Шаг 2: Установка проекта для квоты
!gcloud auth application-default set-quota-project celtic-tendril-459507-q8

# %%
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\fesevu\\AppData\\Roaming\\gcloud\\application_default_credentials.json"
# замените на свой ID
os.environ["GOOGLE_CLOUD_PROJECT"] = "celtic-tendril-459507-q8"

# %% [markdown]
# # Разделяем на контракты и аддреса

# %%
import pandas as pd

# Загрузка исходного CSV
df = pd.read_csv('./data2/merged_6_dataset.csv')

# Удаление строк с suspect
df = df[df['FLAG'] != 'suspect']

# Замена значений FLAG
df['FLAG'] = df['FLAG'].map({'legit': 0, 'scam': 1})

# Приведение is_contract к булевому типу
df['is_contract'] = df['is_contract'].astype(bool)

# Разделение на два датасета
df_contracts = df[df['is_contract'] == True].drop(columns=['is_contract'])
df_non_contracts = df[df['is_contract'] == False].drop(columns=['is_contract'])

# Сохранение в отдельные файлы
df_contracts.to_csv('./data/contracts_only.csv', index=False)
df_non_contracts.to_csv('./data/non_contracts_only.csv', index=False)


# %%
import pandas as pd

# Загрузка датасета с контрактами
df = pd.read_csv('./data/contracts_only.csv')

# Отбор всех флагов 1
df_flag_1 = df[df['FLAG'] == 1]
print(df_flag_1.shape)

# Сколько нужно флагов 0, чтобы получить ровно 3100 строк
needed_flag_0 = 3164 - len(df_flag_1)

# Случайная выборка флагов 0
df_flag_0 = df[df['FLAG'] == 0].sample(n=needed_flag_0, random_state=42)

# Объединение
df_sampled = pd.concat([df_flag_1, df_flag_0]).sample(frac=1, random_state=42)  # перемешаем

# Сохранение
df_sampled.to_csv('./data/contracts_sampled.csv', index=False)


# %% [markdown]
# # Загружает LSTM датасет
# 

# %%
import pandas as pd
from google.cloud import bigquery

# 1. Считываем LSTM-датасет
lstm_df = pd.read_csv('./data/transaction_dataset.csv', usecols=['Address','FLAG'])
lstm_df.columns = ['id','label']
seed_addrs = set(lstm_df['id'])

# %% [markdown]
# # Новая версия

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерирует SQL для «ограниченного 2-hop BFS» + dry-run оценку.
"""
import pandas as pd
import subprocess
import shlex
import json
import textwrap

# ======= ПОДСТАВЬТЕ СВОЁ ======================================
PROJECT = "celtic-tendril-459507-q8"         # GCP-проект
DATASET = "fraud_gnn"          # datataset в BigQuery
LSTM_CSV = "./data/contracts_sampled.csv"          # исходник
DATE_FROM = "2020-01-01"        # начало окна
DATE_TO = "2023-01-01"        # конец окна
MAX_HOP2 = 50             # лимит узлов hop2
# ===============================================================

# 0) выгружаем LSTM-адреса в CSV для загрузки в BQ
df = pd.read_csv(LSTM_CSV, usecols=["Address", "FLAG"])
df["Address"] = df["Address"].str.strip().str.lower()
df.to_csv("./bq/lstm_addresses.csv", index=False)

print(f"→ Загрузите ./bq/lstm_addresses.csv в `{PROJECT}.{DATASET}.lstm_addresses`"
      " (Address STRING, FLAG INT64)")

# %%
# --------------- динамически собираем SQL ---------------------
sql = f"""
-- Параметры
DECLARE date_from DATE   DEFAULT '{DATE_FROM}';
DECLARE date_to   DATE   DEFAULT '{DATE_TO}';
DECLARE max_hop2  INT64  DEFAULT {MAX_HOP2};

-- 0. Seed-таблица
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.lstm` AS
SELECT LOWER(Address) AS addr, CAST(FLAG AS BOOL) AS is_scam
FROM `{PROJECT}.{DATASET}.lstm_addresses`;

-- 1. Периоды активности seed-аккаунтов
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.account_activity` AS
SELECT id,
       MIN(DATE(block_timestamp)) AS first_seen,
       MAX(DATE(block_timestamp)) AS last_seen
FROM (
  SELECT LOWER(from_address) AS id, block_timestamp
    FROM `bigquery-public-data.crypto_ethereum.transactions`
  UNION ALL
  SELECT LOWER(to_address) AS id, block_timestamp
    FROM `bigquery-public-data.crypto_ethereum.transactions`
)
WHERE id IN (SELECT addr FROM `{PROJECT}.{DATASET}.lstm`)
GROUP BY id;

-- 2. Hop-1
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.hop1` AS
WITH seeds AS (
  SELECT l.addr AS seed, a.first_seen, a.last_seen, l.is_scam
  FROM `{PROJECT}.{DATASET}.lstm` AS l
  JOIN `{PROJECT}.{DATASET}.account_activity` AS a
    ON l.addr = a.id
)
SELECT DISTINCT
  s.seed,
  LOWER(IF(t.from_address = s.seed, t.to_address, t.from_address)) AS hop1_addr,
  s.is_scam
FROM seeds AS s
JOIN `bigquery-public-data.crypto_ethereum.transactions` AS t
  ON DATE(t.block_timestamp) BETWEEN s.first_seen AND s.last_seen
 AND t.value > 0
 AND (LOWER(t.from_address) = s.seed OR LOWER(t.to_address) = s.seed);

-- 3. Hop-2 с лимитом per seed
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.hop2` AS
WITH ranked AS (
  SELECT
    h1.seed,
    LOWER(IF(t.from_address = h1.hop1_addr, t.to_address, t.from_address)) AS hop2_addr,
    h1.is_scam AS is_scam_seed,
    COUNT(*) AS tx_cnt
  FROM `{PROJECT}.{DATASET}.hop1`    AS h1
  JOIN `bigquery-public-data.crypto_ethereum.transactions` AS t
    ON t.value > 0
   AND (LOWER(t.from_address) = h1.hop1_addr OR LOWER(t.to_address) = h1.hop1_addr)
  GROUP BY h1.seed, hop2_addr, h1.is_scam
)
SELECT seed, hop2_addr AS addr
FROM ranked
QUALIFY ROW_NUMBER() OVER(PARTITION BY seed ORDER BY tx_cnt DESC)
        <= CASE WHEN is_scam_seed THEN 99999999 ELSE max_hop2 END;

-- 4. Список узлов
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.gnn_accounts` AS
SELECT addr AS id, ANY_VALUE(is_scam) AS is_scam
FROM (
  SELECT addr, is_scam FROM `{PROJECT}.{DATASET}.lstm`
  UNION DISTINCT
  SELECT hop1_addr, NULL FROM `{PROJECT}.{DATASET}.hop1`
  UNION DISTINCT
  SELECT addr,      NULL FROM `{PROJECT}.{DATASET}.hop2`
)
GROUP BY id;

-- 5. Рёбра (с предыдущим partition spec через DATE())
DROP TABLE IF EXISTS `{PROJECT}.{DATASET}.gnn_transactions`;
CREATE TABLE `{PROJECT}.{DATASET}.gnn_transactions`
PARTITION BY DATE(block_timestamp)
OPTIONS(require_partition_filter=TRUE) AS
SELECT
  LOWER(t.from_address) AS src,
  LOWER(t.to_address)   AS dst,
  SAFE_DIVIDE(CAST(t.value AS FLOAT64), 1e18) AS amount,
  t.block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a1
  ON LOWER(t.from_address) = a1.id
JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a2
  ON LOWER(t.to_address)   = a2.id
WHERE t.value > 0
  AND DATE(t.block_timestamp) BETWEEN date_from AND date_to;

-- 6. Стратифицированная выборка ≤150 000 транзакций
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.gnn_transactions_sample` AS
WITH
  all_tx AS (
    -- теперь обязательно фильтруем по партиционной колонке
    SELECT *
    FROM `{PROJECT}.{DATASET}.gnn_transactions`
    WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
  ),
  scam_edges AS (
    SELECT t.*
    FROM all_tx AS t
    JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a1 ON t.src = a1.id
    JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a2 ON t.dst = a2.id
    WHERE a1.is_scam OR a2.is_scam
  ),
  legit_edges AS (
    SELECT t.*
    FROM all_tx AS t
    JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a1 ON t.src = a1.id
    JOIN `{PROJECT}.{DATASET}.gnn_accounts` AS a2 ON t.dst = a2.id
    WHERE NOT a1.is_scam AND NOT a2.is_scam
  ),
  scam_count AS (
    SELECT COUNT(*) AS cnt FROM scam_edges
  ),
  legit_ranked AS (
    SELECT
      src, dst, amount, block_timestamp,
      ROW_NUMBER() OVER (ORDER BY RAND()) AS rn
    FROM legit_edges
  )
SELECT * FROM scam_edges
UNION ALL
SELECT src, dst, amount, block_timestamp
FROM legit_ranked
WHERE rn <= GREATEST(150000 - (SELECT cnt FROM scam_count), 0);
"""

# сохраняем скрипт
with open("./bq/create_stratified_bfs.sql", "w", encoding="utf-8") as f:
    f.write(sql)

print("→ SQL для stratified BFS сохранён в ./bq/create_stratified_bfs.sql")

# %%
from google.cloud import bigquery

# Укажи ID проекта GCP
PROJECT = "celtic-tendril-459507-q8"

# Инициализация клиента
client = bigquery.Client(project=PROJECT)

# Чтение SQL-запроса из файла
# можно также использовать pathlib
sql_path = Path("D:/Fraud/dataset/create_mini_bfs.sql")
sql = sql_path.read_text()

# Конфигурация на dry-run
job_config = bigquery.QueryJobConfig(
    dry_run=True,
    use_query_cache=False
)

# Выполнение dry-run
query_job = client.query(sql, job_config=job_config)

# Вывод информации
print("→ Делаем dry-run…")
print("   totalBytesProcessed = {:.3f} GB".format(
    query_job.total_bytes_processed / 1e9))
if query_job.total_bytes_processed <= 5e9:
    print("   ✅ Можно запускать без риска — не превышает 5 GB.")
else:
    print("   ⚠️ Превышает 5 GB — может потребоваться включённый billing.")

# %% [markdown]
# # Собираем в цельный датасет

# %%
import gzip
import glob
from pathlib import Path
import shutil

# ─── настройка ────────────────────────────────────────────────────────────────
PATTERN = "./data/gnn_transactions-*.csv.gz"     # шаблон шардов
OUT_GZ = "./data/gnn_transactions_full.csv.gz"  # итоговый gzip
NEED_CSV = True                                   # True → сделать и .csv
# ──────────────────────────────────────────────────────────────────────────────

files = sorted(glob.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"Не найдено файлов по шаблону: {PATTERN}")

out_path = Path(OUT_GZ)
out_path.parent.mkdir(parents=True, exist_ok=True)

header_written = False
line_cnt = 0

with gzip.open(out_path, "wt", encoding="utf-8", newline="") as fout:
    for idx, fname in enumerate(files, 1):
        with gzip.open(fname, "rt", encoding="utf-8", newline="") as fin:
            for i, line in enumerate(fin):
                # пропускаем заголовок во всех, кроме первого шарда
                if i == 0 and header_written:
                    continue
                if i == 0 and not header_written:
                    header_written = True
                fout.write(line)
                line_cnt += 1
        print(f"✓ [{idx:>3}/{len(files)}] merged {Path(fname).name}")

print(f"\nСоздан архив → {out_path}   ({line_cnt:,} строк)")

# ─── опционально: распаковать в CSV ───────────────────────────────────────────
if NEED_CSV:
    OUT_CSV = out_path.with_suffix("")   # .csv
    with gzip.open(out_path, "rb") as gzf, OUT_CSV.open("wb") as csvf:
        shutil.copyfileobj(gzf, csvf)
    print(f"Распакован в → {OUT_CSV}")

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_gnn_dataset.py
~~~~~~~~~~~~~~~~~~~~~~
Сводит основные метрики датасета:
  • число узлов, размеченных / fraud-узлов
  • число рёбер, диапазон дат
  • min / max / mean / median суммы перевода
  • 5 крупнейших транзакций
  • топ-5 самых «беспокойных» адресов (по степени)
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from datetime import datetime

# --- настроить под свои имена файлов --------------------------
ACCOUNTS_CSV = Path("./data/gnn_accounts.csv")               # ~ KB-МB
TX_CSV       = Path("./data/gnn_transactions_full.csv.gz")   # или .csv
CHUNK_ROWS   = 1_000_000       # размер чанка при чтении транзакций
SAMPLE_AMT   = 50_000          # сколько сумм берём для оценки медианы
# --------------------------------------------------------------

assert ACCOUNTS_CSV.exists(), f"{ACCOUNTS_CSV} not found"
assert TX_CSV.exists(), f"{TX_CSV} not found"

# 1) узлы -------------------------------------------------------
acc_df = pd.read_csv(
    ACCOUNTS_CSV,
    dtype={"id": "string"},
)

n_nodes      = len(acc_df)
n_labeled    = acc_df["label"].notna().sum()
n_fraud      = int(acc_df["label"].sum(skipna=True))
n_unlabeled  = n_nodes - n_labeled

# 2) рёбра ------------------------------------------------------
dtypes_tx = {"src": "string",
             "dst": "string",
             "amount": "float64"}     # ETH
parse_dates = ["block_timestamp"]

edge_cnt          = 0
amt_sum           = 0.0
amt_min, amt_max  = np.inf, -np.inf
first_ts, last_ts = None, None
top_tx            = []                # 5 крупнейших переводов
deg_counter       = Counter()
amt_samples       = []

for chunk in pd.read_csv(TX_CSV,
                         dtype=dtypes_tx,
                         parse_dates=parse_dates,
                         chunksize=CHUNK_ROWS,
                         compression="infer"):

    edge_cnt += len(chunk)

    # диапазон дат
    ts_min, ts_max = chunk["block_timestamp"].min(
    ), chunk["block_timestamp"].max()
    first_ts = ts_min if first_ts is None else min(first_ts, ts_min)
    last_ts  = ts_max if last_ts  is None else max(last_ts,  ts_max)

    # суммы
    c_amt = chunk["amount"]
    amt_sum  += c_amt.sum()
    amt_min   = min(amt_min, c_amt.min())
    amt_max   = max(amt_max, c_amt.max())

    # случайные сэмплы для оценки медианы
    if len(amt_samples) < SAMPLE_AMT:
        need = SAMPLE_AMT - len(amt_samples)
        amt_samples.extend(c_amt.sample(min(need, len(chunk)),
                                        random_state=42).tolist())

    # топ-5 крупнейших переводов
    top_tx.extend(chunk.nlargest(5, "amount").to_dict("records"))
    top_tx  = sorted(top_tx, key=lambda r: r["amount"], reverse=True)[:5]

    # степени узлов
    deg_counter.update(chunk["src"])
    deg_counter.update(chunk["dst"])

# агрегаты
amt_mean    = amt_sum / edge_cnt
amt_median  = np.median(amt_samples)  # приближённо
top_degree  = deg_counter.most_common(5)

# 3) печатаем ---------------------------------------------------
print("\n=====  GNN DATASET SUMMARY  =====\n")
print(f"Accounts (nodes):        {n_nodes:,}")
print(f"  ├─ labeled:            {n_labeled:,}")
print(f"  ├─   └─ fraud (1):     {n_fraud:,}")
print(f"  └─ unlabeled (NULL):   {n_unlabeled:,}")

print(f"\nTransactions (edges):    {edge_cnt:,}")
print(f"  Time span:             {first_ts:%Y-%m-%d}  →  {last_ts:%Y-%m-%d}")
print(f"  Amount, ETH:")
print(f"    min  = {amt_min:,.6f}")
print(f"    max  = {amt_max:,.6f}")
print(f"    mean = {amt_mean:,.6f}")
print(f"    med. ≈ {amt_median:,.6f}")

print("\nTop-5 biggest transfers:")
for i, tx in enumerate(top_tx, 1):
    print(f"  {i}. {tx['amount']:,.6f}  ETH   "
          f"{tx['src'][:6]}… → {tx['dst'][:6]}…  "
          f"{tx['block_timestamp']:%Y-%m-%d}")

print("\nTop-5 busiest addresses (degree):")
for i, (addr, deg) in enumerate(top_degree, 1):
    print(f"  {i}. {addr[:6]}…   {deg:,} edges")

print("\nEverything looks readable ✔")

# %%
import pandas, sys, pathlib, platform
print("pandas version:", pandas.__version__)
print("pandas path:   ", pathlib.Path(pandas.__file__).parent)
print("python:", platform.python_version())


# %% [markdown]
# # Etherscan

# %%
import time
import requests
import pandas as pd

API_KEY = ""
BASE_URL = "https://api.etherscan.io/api"

def fetch_txs(address, page=1, offset=10000):
    """Загружает транзакции (обычные) для address."""
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": page,
        "offset": offset,
        "sort": "desc",
        "apikey": API_KEY
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    if data["status"] != "1":
        return []
    return data["result"]

def fetch_internal_txs(address, page=1, offset=10000):
    """Загружает внутренние (internal) транзакции."""
    params = {
        "module": "account",
        "action": "txlistinternal",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": page,
        "offset": offset,
        "sort": "desc",
        "apikey": API_KEY
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    if data["status"] != "1":
        return []
    return data["result"]

def get_all_txs(address):
    """Собирает обычные + internal, пагинируя до пустой страницы."""
    all_txs = []
    for fetcher in (fetch_txs, fetch_internal_txs):
        page = 1
        while True:
            txs = fetcher(address, page=page)
            if not txs:
                break
            all_txs.extend(txs)
            page += 1
            time.sleep(0.2)   # чтобы не превысить 5 rps
    return all_txs

# 1) Читаем LSTM-адреса
lstm = pd.read_csv("./data/transaction_dataset.csv", usecols=["Address","FLAG"])
seed_addrs = set(lstm["Address"].tolist())

# 2) BFS-обход глубины 2
visited = set(seed_addrs)
frontier = set(seed_addrs)
edges = []   # тут будем собирать ребра

for depth in range(2):
    next_frontier = set()
    for addr in frontier:
        txs = get_all_txs(addr)
        for tx in txs:
            src = tx.get("from")
            dst = tx.get("to")
            amt = tx.get("value")
            ts  = int(tx.get("block_timestamp", 0))
            edges.append((src, dst, amt, ts))
            # запоминаем нового соседа
            for nbr in (src, dst):
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
    frontier = next_frontier

# 3) Сохраняем transaction.csv
tx_df = pd.DataFrame(edges, columns=["src","dst","amount","block_timestamp"])
tx_df.to_csv("./data/transaction.csv", index=False)

# 4) Сбор account.csv
all_nodes = pd.Series(list(tx_df["src"]) + list(tx_df["dst"]), name="id")
all_nodes = all_nodes.drop_duplicates().to_frame()
# маппим метки: LSTM→FLAG, новые узлы = -1
label_map = dict(zip(lstm["Address"], lstm["FLAG"]))
all_nodes["label"] = all_nodes["id"].map(label_map).fillna(-1).astype(int)
all_nodes.to_csv("./data/account.csv", index=False)

print("Собрано:", tx_df.shape[0], "транзакций;", all_nodes.shape[0], "узлов.")

# %% [markdown]
# # Очистка датасета

# %%
import pandas as pd
from pathlib import Path

# Пути к файлам
ACCOUNTS_CSV     = Path('../dataset/gnn/contracts_sampled.csv')
TXS_CSV          = Path('../dataset/gnn/gnn_transactions.csv')
EXISTING_CSV     = Path('../gnn/tmp/Ethereum/account.csv')
OUTPUT_FILTERED  = Path('../dataset/gnn/filtered_accounts.csv')

# Проверка наличия файлов
assert ACCOUNTS_CSV.exists(),    f'{ACCOUNTS_CSV} not found'
assert TXS_CSV.exists(),         f'{TXS_CSV} not found'
assert EXISTING_CSV.exists(),    f'{EXISTING_CSV} not found'

# 1) Исходная фильтрация по транзакциям
accounts_df = pd.read_csv(ACCOUNTS_CSV)
txs_df      = pd.read_csv(TXS_CSV)

accounts_df['Address'] = accounts_df['Address'].str.lower()
txs_df['src']          = txs_df['src'].str.lower()
txs_df['dst']          = txs_df['dst'].str.lower()

tx_addresses = set(txs_df['src']) | set(txs_df['dst'])
filtered_df  = accounts_df[accounts_df['Address'].isin(tx_addresses)]

# Сохраняем результат фильтрации
filtered_df.to_csv(OUTPUT_FILTERED, index=False)

# 2) Читаем существующий account.csv (без заголовков)
existing_df = pd.read_csv(
    EXISTING_CSV,
    header=None,
    names=['id', 'label']
)
existing_df['id'] = existing_df['id'].str.lower()

# 3) Готовим filtered_df к объединению: переименовываем колонку
filtered_for_merge = filtered_df.rename(
    columns={'Address': 'id', 'FLAG': 'label'},
)[['id', 'label']]

# 4) Объединяем, дропаем дубликаты по 'id'
combined_df = pd.concat([existing_df, filtered_for_merge], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=['id'])

# 5) Сохраняем объединённый файл с заголовками
combined_df.to_csv(OUTPUT_FILTERED, index=False)

print(f"Filtered accounts saved to {OUTPUT_FILTERED}")


# %%
import pandas as pd

# Загрузка отфильтрованного датасета
df = pd.read_csv('../dataset/gnn/filtered_accounts.csv')

# Подсчёт количества каждого класса
class_counts = df['label'].value_counts().sort_index()

# Вывод абсолютных значений
print("Абсолютное количество:")
print(class_counts)

# Вывод относительных значений (в процентах)
print("\nОтносительное соотношение:")
print((class_counts / class_counts.sum() * 100).round(2).astype(str) + '%')


# %%
import pandas as pd
from pathlib import Path

# Пути к файлам
ACCOUNTS_CSV     = Path('../dataset/gnn/contracts_sampled.csv')
TXS_CSV          = Path('../dataset/gnn/gnn_transactions.csv')
EXISTING_CSV     = Path('../dataset/data2/xblock13_addresses_with_contract_info.csv')
OUTPUT_FILTERED  = Path('../dataset/gnn/filtered_contracts_only.csv')

# Проверка наличия файлов
assert ACCOUNTS_CSV.exists(),    f'{ACCOUNTS_CSV} not found'
assert TXS_CSV.exists(),         f'{TXS_CSV} not found'
assert EXISTING_CSV.exists(),    f'{EXISTING_CSV} not found'

# 1) Исходная фильтрация по транзакциям
accounts_df = pd.read_csv(ACCOUNTS_CSV)
txs_df      = pd.read_csv(TXS_CSV)

accounts_df['Address'] = accounts_df['Address'].str.lower()
txs_df['src']          = txs_df['src'].str.lower()
txs_df['dst']          = txs_df['dst'].str.lower()

tx_addresses = set(txs_df['src']) | set(txs_df['dst'])
filtered_df  = accounts_df[accounts_df['Address'].isin(tx_addresses)]

# Сохраняем результат фильтрации
filtered_df.to_csv(OUTPUT_FILTERED, index=False)

# 2) Читаем существующий account.csv (без заголовков)
existing_df = pd.read_csv(
    EXISTING_CSV
)
existing_df['Address'] = existing_df['Address'].str.lower()
existing_df = existing_df[existing_df['is_contract']==True]
existing_df = existing_df.drop('is_contract', axis=1)
# 3) Готовим filtered_df к объединению: переименовываем колонку
existing_df = existing_df.rename(
    columns={'Address': 'id', 'FLAG': 'label'},
)[['id', 'label']]

# 3) Готовим filtered_df к объединению: переименовываем колонку
filtered_for_merge = filtered_df.rename(
    columns={'Address': 'id', 'FLAG': 'label'},
)[['id', 'label']]

# 4) Объединяем, дропаем дубликаты по 'id'
combined_df = pd.concat([existing_df, filtered_for_merge], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=['id'])

# 5) Сохраняем объединённый файл с заголовками
combined_df.to_csv(OUTPUT_FILTERED, index=False)

print(f"Filtered accounts saved to {OUTPUT_FILTERED}")


# %%
import pandas as pd

# Загрузка отфильтрованного датасета
df = pd.read_csv('../dataset/gnn/filtered_contracts_only.csv')

# Подсчёт количества каждого класса
class_counts = df['label'].value_counts().sort_index()

# Вывод абсолютных значений
print("Абсолютное количество:")
print(class_counts)

# Вывод относительных значений (в процентах)
print("\nОтносительное соотношение:")
print((class_counts / class_counts.sum() * 100).round(2).astype(str) + '%')


# %%
import pandas as pd
from pathlib import Path

# Пути к файлам
TXS1_CSV = Path('../dataset/gnn/gnn_transactions.csv')       # с заголовками: src,dst,amount,block_timestamp
TXS2_CSV = Path('../gnn/tmp/Ethereum/transaction.csv')         # без заголовков: src,dst,amount,timestamp
OUT_CSV  = Path('../dataset/gnn/filtered_transactions.csv')        # перепишем второй файл с объединёнными данными

# Проверка наличия
assert TXS1_CSV.exists(), f'{TXS1_CSV} not found'
assert TXS2_CSV.exists(), f'{TXS2_CSV} not found'

# 1) Читаем первый датасет (с заголовком)
tx1 = pd.read_csv(TXS1_CSV)

# Переименуем колонку block_timestamp в timestamp, чтобы совпадало
tx1 = tx1.rename(columns={'block_timestamp': 'timestamp'})

# 2) Читаем второй датасет (без заголовков), с нужными именами
tx2 = pd.read_csv(
    TXS2_CSV,
    header=None,
    names=['src', 'dst', 'amount', 'timestamp']
)

tx1['timestamp'] = pd.to_datetime(tx1['timestamp'], utc=True).astype(int) / 1e9

# 3) Объединяем и дропаем полные дубликаты строк
combined = pd.concat([tx1[['src','dst','amount','timestamp']], tx2], ignore_index=True)
combined = combined.drop_duplicates()

# Явно приводим amount к float
combined['amount'] = combined['amount'].astype(float)

# 4) Сохраняем обратно во второй файл с заголовками второго датасета
combined.to_csv(OUT_CSV, index=False)

print(f"Combined transactions saved to {OUT_CSV} ({len(combined)} rows)")



