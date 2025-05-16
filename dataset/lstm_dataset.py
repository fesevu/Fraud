# %%
# Установить клиентскую библиотеку
%pip uninstall -y pandas
# свежий pip подтянет правильный wheel
%pip install --upgrade pip           
%pip install "pandas>=2.2.0"

%pip install web3 tqdm


# %% [markdown]
# # Собираем датасет для lstm
# Используем
# 1. vagifa/ethereum-frauddetection-dataset c kaggle
# 2. https://github.com/surajsjain/ethereum-fraud-datasets/tree/main
# 3. https://github.com/eltontay/Ethereum-Fraud-Detection/tree/main (https://arxiv.org/abs/1908.07886)
# 4. https://xblock.pro/#/dataset/13
# 5. https://xblock.pro/#/dataset/25
# 7. https://xblock.pro/#/dataset/50

# %% [markdown]
# # Проверяем пересечения датасета 1 и 4

# %%
import pandas as pd

# Загрузим два CSV
df1 = pd.read_csv("./data2/lstm_with_contract_info.csv")  # замените на свой путь
df2 = pd.read_csv("./data2/xblock13_addresses_with_contract_info.csv")
# Приведение адресов к нижнему регистру и удаление пробелов
df1['Address'] = df1['Address'].str.lower().str.strip()
df2['Address'] = df2['Address'].str.lower().str.strip()

# Уникальные адреса
unique1 = set(df1['Address'])
unique2 = set(df2['Address'])

# Пересечение
common = unique1 & unique2

# Фильтрация по is_contract = True
# Фильтрация по is_contract=True в df1 и попаданию в общие адреса
df1_contracts = df1[(df1['is_contract'] == True) & (df1['Address'].isin(common))]

# Из них — флаг=1
df1_contracts_flag1 = df1_contracts[df1_contracts['FLAG'] == 1]

# Аналогично: из df2 — флаг=True
df2_contracts_flag_true = df2[(df2['Address'].isin(df1_contracts['Address'])) & (df2['FLAG'] == True)]

common_contracts = set(df1_contracts['Address']) & common

# Вывод
print(f"Общее количество общих address: {len(common)}")
print(f"Из них с is_contract=True в df1: {len(df1_contracts)}")
print(f"Из них FLAG=1 (мошенники) в df1: {len(df1_contracts_flag1)}")
print(f"Из них FLAG=True (мошенники) в df2: {len(df2_contracts_flag_true)}")
print(f"Из них is_contract=True: {len(common_contracts)}")


common_verified = set(df1_contracts_flag1['Address']) & set(df2_contracts_flag_true['Address'])
print(f"Совпадающих мошенников is_contract=True в обоих CSV: {len(common_verified)}")



# %% [markdown]
# ## Обьединяем датасеты (2 из 3)

# %%
import pandas as pd

# Загрузка только нужных колонок
df1 = pd.read_csv('./original/address_data_e.csv', usecols=['Address', 'FLAG'])
df2 = pd.read_csv('./original/lstm.csv', usecols=['Address', 'FLAG'])

# Объединение
merged_df = pd.concat([df1, df2], ignore_index=True)

# Удалим дубликаты по адресу (если один и тот же адрес есть в обоих и флаги совпадают — оставим один)
merged_df = merged_df.drop_duplicates()

# Сохраняем
merged_df.to_csv('./data2/merged_flags_only.csv', index=False)

print(f"Готово: {len(merged_df)} строк сохранено в merged_flags_only.csv")


# %% [markdown]
# ## Проверяем на контракт

# %%
from web3 import Web3
import pandas as pd

# Подключение к Ethereum через Infura (или другой провайдер)
w3 = Web3(Web3.HTTPProvider(
    'https://mainnet.infura.io/v3/7d52e5a07e9a4d9985e195ed49f26d5d'))

# Функция для проверки, является ли адрес смарт-контрактом


def is_contract(address):
    try:
        # Пропускаем пустые и некорректные значения
        if not isinstance(address, str) or not address.startswith('0x') or len(address) != 42:
            return False
        # Преобразуем в checksum и получаем код
        checksum_address = Web3.to_checksum_address(address)
        code = w3.eth.get_code(checksum_address)
        return len(code) > 2  # '0x' или пустое => не контракт
    except Exception as e:
        print(f"Ошибка при обработке {address}: {e}")
        return False


# Загружаем свой датасет с адресами (например, из CSV)
df = pd.read_csv('./original/lstm.csv')  # Заменить на свой путь

# Предположим, что в столбце 'address' находятся адреса Ethereum
df['is_contract'] = df['Address'].apply(lambda x: is_contract(x))

# Сохраняем результаты в новый CSV
df.to_csv('./data2/lstm_with_contract_info.csv', index=False)

print(df.head())

# %%


# %%
from web3 import Web3
import pandas as pd

# Подключение к Ethereum через Infura (или другой провайдер)
w3 = Web3(Web3.HTTPProvider(
    'https://mainnet.infura.io/v3/YOUR_API_KEY'))

# Функция для проверки, является ли адрес смарт-контрактом


def is_contract(address):
    try:
        # Пропускаем пустые и некорректные значения
        if not isinstance(address, str) or not address.startswith('0x') or len(address) != 42:
            return False
        # Преобразуем в checksum и получаем код
        checksum_address = Web3.to_checksum_address(address)
        code = w3.eth.get_code(checksum_address)
        return len(code) > 2  # '0x' или пустое => не контракт
    except Exception as e:
        print(f"Ошибка при обработке {address}: {e}")
        return False


# Загружаем свой датасет с адресами (например, из CSV)
df = pd.read_csv('./data2/merged_flags_only.csv')  # Заменить на свой путь

# Предположим, что в столбце 'address' находятся адреса Ethereum
df['is_contract'] = df['Address'].apply(lambda x: is_contract(x))

# Сохраняем результаты в новый CSV
df.to_csv('./data2/addresses_with_contract_info.csv', index=False)

print(df.head())

# %%
import pandas as pd

# Загружаем оба датасета
gt_df = pd.read_csv('./data2/addresses_with_contract_info.csv')

gt_df = gt_df[gt_df["is_contract"]==True]

print(gt_df.shape)

gt_df_s = gt_df[gt_df["FLAG"]==True]
print(gt_df_s.shape)
print(gt_df.shape[0] - gt_df_s.shape[0])

# %% [markdown]
# # Проверяем какие поля есть в датасете
# 3 датасет

# %%
import pandas as pd

# Загружаем CSV
df = pd.read_csv('./original/Eth_labeled_addresses.csv')

# Настройки отображения без сокращений
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Исключаем столбец Address из анализа
columns_to_analyze = [col for col in df.columns if col.lower() != 'address']

# Просмотр всех уникальных значений и их частот по каждому оставшемуся столбцу
for column in columns_to_analyze:
    print(f"\n=== Столбец: {column} ===")
    print(df[column].value_counts(dropna=False))


# %%
import pandas as pd

# Загружаем CSV
df = pd.read_csv('./original/Eth_labeled_addresses.csv')

# Настройки отображения без сокращений
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Оставляем только нужные поля
columns_to_analyze = [
    'Account Type',   # смарт-контракт или нет
    'Label',          # подозрение (Dodgy)
    'Tags',           # признаки мошенничества
    'Name',           # возможные указания на хакеров
    'Unnamed: 7',     # вторичные признаки мошенничества
    'Unnamed: 8'      # вторичные признаки мошенничества
]

# Просмотр всех уникальных значений и их частот
for column in columns_to_analyze:
    if column in df.columns:
        print(f"\n=== Столбец: {column} ===")
        print(df[column].value_counts(dropna=False))

# %% [markdown]
# ## Определям из 3 датасета скам, не скам и смарт контракты

# %%
import re
import pandas as pd


# Загрузка исходного файла

df = pd.read_csv('./original/Eth_labeled_addresses.csv')



fraud_keywords = ['scam', 'scamming', 'phish', 'phishing',
                  'hack', 'heist', 'compromised', 'ponzi', 'fake']


def detect_first_fraud_type(row):
    # Собираем текст из подозрительных колонок
    text_sources = [
        str(row.get(col, '')).lower()
        for col in ['Tags', 'Unnamed: 7', 'Unnamed: 8']
    ]
    combined_text = ' '.join(text_sources)

    # Удаляем спецсимволы и нормализуем
    cleaned_text = re.sub(r'[^a-z0-9]+', ' ', combined_text)

    # Создаём множество токенов
    tokens = set(cleaned_text.split())

    # Возвращаем первый найденный ключ в порядке fraud_keywords
    for keyword in fraud_keywords:
        if keyword in tokens:
            return keyword

    return None


# Смарт-контракт или нет

df['is_contract'] = df['Account Type'].fillna('').str.strip().str.lower() == 'smart contract'


# Тип скама

df['scam_type'] = df.apply(detect_first_fraud_type, axis=1)


# Мошенник или нет

def classify_status(row):

    if pd.notna(row['scam_type']):
        return 'scam'

    label = str(row.get('Label', '')).lower().strip()

    if label == 'dodgy':
        return 'suspect'
    return 'not_scam'


df['scam_status'] = df.apply(classify_status, axis=1)


# Оставляем нужные поля

final_df = df[['Address', 'is_contract', 'scam_status', 'scam_type']].rename(columns={'Address': 'address'})


# Удаляем дубликаты и мусорные строки

final_df = final_df[final_df['address'].str.startswith('0x', na=False)]  # только валидные адреса

final_df = final_df.drop_duplicates(subset='address')


# Сохраняем

final_df.to_csv('./data2/eth_fraud_labeled_categorized.csv', index=False)

print("✅ Готово! Сохранено в eth_fraud_labeled_categorized.csv")

# %% [markdown]
# ## Объединяем 3 датасета

# %%
import pandas as pd

# Загрузка датасетов
df1 = pd.read_csv('./data2/addresses_with_contract_info.csv')  # содержит Address, FLAG, is_contract
df2 = pd.read_csv('./data2/eth_fraud_labeled_categorized.csv')  # содержит address, is_contract, scam_status, scam_type

# Приведение столбцов к одному стилю
df1 = df1.rename(columns={'Address': 'Address'})
df2 = df2.rename(columns={'address': 'Address'})

# Приведение значений FLAG
flag_map_1 = {1: 'scam', 0: 'legit'}
flag_map_2 = {'scam': 'scam', 'not_scam': 'legit', 'suspect': 'suspect', 'suspend': 'suspend'}

df1['FLAG'] = df1['FLAG'].map(flag_map_1)
df1['scam_type'] = None  # в первом датасете нет этой информации

df2['FLAG'] = df2['scam_status'].map(flag_map_2)
df2 = df2.drop(columns=['scam_status'])

# Объединение и удаление дубликатов
combined = pd.concat([df1[['Address', 'FLAG', 'is_contract', 'scam_type']], 
                      df2[['Address', 'FLAG', 'is_contract', 'scam_type']]])

combined = combined.drop_duplicates(subset='Address', keep='first')

# Сохранение результата
combined.to_csv('./data2/merged_3_dataset.csv', index=False)

print(f'Итоговый датасет содержит {len(combined)} уникальных адресов')


# %% [markdown]
# ## Работа с 4 датасетом

# %%
from web3 import Web3
import pandas as pd

# Подключение к Ethereum через Infura (или другой провайдер)
w3 = Web3(Web3.HTTPProvider(
    'https://mainnet.infura.io/v3/YOUR_API_KEY'))

# Функция для проверки, является ли адрес смарт-контрактом
def is_contract(address):
    try:
        if not isinstance(address, str) or not address.startswith('0x') or len(address) != 42:
            return False
        checksum_address = Web3.to_checksum_address(address)
        code = w3.eth.get_code(checksum_address)
        return len(code) > 2
    except Exception as e:
        print(f"Ошибка при обработке {address}: {e}")
        return False

# Загружаем датасет без заголовков и задаём имена колонок
df = pd.read_csv('./original/xblock13_account.csv', header=None, names=['Address', 'FLAG'])

# Приводим адреса к нижнему регистру (на всякий случай)
df['Address'] = df['Address'].str.lower()

# Проверка адресов на принадлежность к смарт-контрактам
df['is_contract'] = df['Address'].apply(is_contract)

# Сохраняем результат
df.to_csv('./data2/xblock13_addresses_with_contract_info.csv', index=False)

print(df.head())


# %%
import pandas as pd

# Загружаем оба датасета
gt_df = pd.read_csv('./data2/xblock13_addresses_with_contract_info.csv')

gt_df = gt_df[gt_df["is_contract"]==True]

print(gt_df.shape)

gt_df_s = gt_df[gt_df["FLAG"]==True]
print("Скам", gt_df_s.shape)
print(gt_df.shape[0] - gt_df_s.shape[0])

# %%
import pandas as pd

# Загружаем оба датасета
gt_df = pd.read_csv('./data2/xblock13_addresses_with_contract_info.csv')
main_df = pd.read_csv('./data2/merged_3_dataset.csv')

# Приводим названия колонок к одному формату
gt_df.rename(columns={'address': 'Address'}, inplace=True)

# Приводим адреса к нижнему регистру
gt_df['Address'] = gt_df['Address'].str.lower()
main_df['Address'] = main_df['Address'].str.lower()

flag_map_1 = {True: 'scam', False: 'legit'}

gt_df['FLAG'] = gt_df['FLAG'].map(flag_map_1)

# Добавляем нужные поля к gt_df
gt_df['scam_type'] = 'phish'

# Оставляем только нужные поля
gt_df = gt_df[['Address', 'FLAG', 'is_contract', 'scam_type']]

# Объединяем с основным датасетом
merged_df = pd.concat([main_df, gt_df], ignore_index=True)

# Удаляем дубликаты по адресу, оставляя первую встречающуюся строку
merged_df = merged_df.drop_duplicates(subset='Address', keep='first')

# Сохраняем результат
merged_df.to_csv('./data2/merged_4_dataset.csv', index=False)

print('Объединение завершено. Файл сохранён в ./data2/merged_4_dataset.csv')
print(f'Итоговый датасет содержит {len(merged_df)} уникальных адресов')


# %% [markdown]
# ## Работа с 5 датасетом

# %%
import pandas as pd

# Загружаем оба датасета
gt_df = pd.read_csv('./original/PonziDataset_20221114.csv')
main_df = pd.read_csv('./data2/merged_4_dataset.csv')

# Приводим названия колонок к одному формату
gt_df.rename(columns={'address': 'Address'}, inplace=True)

# Приводим адреса к нижнему регистру
gt_df['Address'] = gt_df['Address'].str.lower()
main_df['Address'] = main_df['Address'].str.lower()

# Формируем поля по правилам
gt_df['FLAG'] = gt_df['label'].apply(lambda x: 'scam' if x == 1 else 'legit')
gt_df['is_contract'] = True
gt_df['scam_type'] = gt_df['label'].apply(lambda x: 'phish' if x == 1 else '')

# Оставляем только нужные колонки
gt_df = gt_df[['Address', 'FLAG', 'is_contract', 'scam_type']]

# Объединяем с основным датасетом
merged_df = pd.concat([main_df, gt_df], ignore_index=True)

# Удаляем дубликаты по адресу, оставляя первую встречающуюся строку
merged_df = merged_df.drop_duplicates(subset='Address', keep='first')

# Сохраняем результат
merged_df.to_csv('./data2/merged_5_dataset.csv', index=False)

print('Объединение завершено. Файл сохранён в ./data2/merged_5_dataset.csv')
print(f'Итоговый датасет содержит {len(merged_df)} уникальных адресов')


# %% [markdown]
# ## Преобразуем 6 датасет

# %%
import pandas as pd

# Загружаем оба датасета
gt_df = pd.read_csv('./original/groundTruth.csv', sep=';')
main_df = pd.read_csv('./data2/merged_5_dataset.csv')

# Приводим названия колонок к одному формату
gt_df.rename(columns={'address': 'Address'}, inplace=True)

# Приводим адреса к нижнему регистру
gt_df['Address'] = gt_df['Address'].str.lower()
main_df['Address'] = main_df['Address'].str.lower()

# Добавляем нужные поля к gt_df
gt_df['FLAG'] = 'scam'
gt_df['is_contract'] = True
gt_df['scam_type'] = 'rug_pull'

# Оставляем только нужные поля
gt_df = gt_df[['Address', 'FLAG', 'is_contract', 'scam_type']]

# Объединяем с основным датасетом
merged_df = pd.concat([main_df, gt_df], ignore_index=True)

# Удаляем дубликаты по адресу, оставляя первую встречающуюся строку
merged_df = merged_df.drop_duplicates(subset='Address', keep='first')

# Сохраняем результат
merged_df.to_csv('./data2/merged_6_dataset.csv', index=False)

print('Объединение завершено. Файл сохранён в ./data2/merged_6_dataset.csv')
print(f'Итоговый датасет содержит {len(merged_df)} уникальных адресов')


# %% [markdown]
# # Проверяем соотношение

# %%
# Подсчёт количества контрактов и не-контрактов
df = pd.read_csv('./data2/merged_6_dataset.csv')  # Заменить на свой путь

counts = df['is_contract'].value_counts()
total = counts.sum()

contracts = counts.get(True, 0)
externals = counts.get(False, 0)

contracts_pct = contracts / total * 100
externals_pct = externals / total * 100

print("\n📊 Статистика адресов:")
print(f"Смарт-контрактов     : {contracts:,} ({contracts_pct:.2f}%)")
print(f"Обычных аккаунтов    : {externals:,} ({externals_pct:.2f}%)")
print(f"Всего адресов        : {total:,}")

# Фильтруем только смарт-контракты
contracts_df = df[df['is_contract'] == True]

# Считаем количество скамерских и нескамерских
scam_contracts = contracts_df[contracts_df['FLAG'] == "scam"].shape[0]
legit_contracts = contracts_df[contracts_df['FLAG'] == "legit"].shape[0]
total_contracts = contracts_df.shape[0]

# Проценты
scam_pct = scam_contracts / total_contracts * 100 if total_contracts else 0
legit_pct = legit_contracts / total_contracts * 100 if total_contracts else 0

# Вывод
print("\n🔍 Статистика смарт-контрактов:")
print(f"Скам-контрактов      : {scam_contracts:,} ({scam_pct:.2f}%)")
print(f"Не скам-контрактов   : {legit_contracts:,} ({legit_pct:.2f}%)")
print(f"Всего контрактов     : {total_contracts:,}")

# Фильтруем только смарт-контракты
contracts_df = df[df['is_contract'] == False]

# Считаем количество скамерских и нескамерских
scam_contracts = contracts_df[contracts_df['FLAG'] == "scam"].shape[0]
legit_contracts = contracts_df[contracts_df['FLAG'] == "legit"].shape[0]
total_contracts = contracts_df.shape[0]

# Проценты
scam_pct = scam_contracts / total_contracts * 100 if total_contracts else 0
legit_pct = legit_contracts / total_contracts * 100 if total_contracts else 0

print("\n🔍 Статистика аккаунтов:")
print(f"Скам-аккаунтов      : {scam_contracts:,} ({scam_pct:.2f}%)")
print(f"Не скам-аккаунтов  : {legit_contracts:,} ({legit_pct:.2f}%)")
print(f"Всего аккаунтов     : {total_contracts:,}")


# %% [markdown]
# # Формируем запрос для BIG Query

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерирует SQL-скрипт create_address_features.sql для BigQuery
и выгружает CSV address_labels, оставляя текстовые метки
FLAG = scam / legit / suspect + числовую flag_num.
"""

import pandas as pd
import textwrap
from pathlib import Path

# ───────────────────────── КОНФИГ ──────────────────────────
PROJECT      = "celtic-tendril-459507-q8"
DATASET      = "fraud_lstm"
SRC_CSV      = "./data2/merged_6_dataset.csv"

# «сбалансированный» диапазон — 2 последних года
DATE_TO      = "2025-01-01"
DATE_FROM    = "2015-01-01"

WINDOW_DAYS  = 7       # длительность окна (дней)
MAX_STEPS    = 40      # сколько последних окон брать
# ────────────────────────────────────────────────────────────

# from datetime import date, timedelta

# DATE_TO   = date.today().isoformat()                # сегодня
# DATE_FROM = (date.today() - timedelta(days=210)).isoformat()


# CSV для BigQuery ─────────────────────────
df = pd.read_csv(SRC_CSV, dtype=str)
df.columns = [c.strip() for c in df.columns]
df["Address"]     = df["Address"].str.strip().str.lower()
df["FLAG"]        = df["FLAG"].str.strip().str.lower()
df["is_contract"] = df["is_contract"].str.strip().str.lower()
outfile = Path("./bq/bq_addresses.csv")
outfile.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(outfile, index=False)

print(f"\nCSV готов → {outfile.resolve()}\n")
print(
    "Загрузите его командой (одна строка):\n"
    f"bq --location=US load --source_format=CSV --skip_leading_rows=1 "
    "--field_delimiter=',' --quote='\"' "
    "--schema='Address:STRING,FLAG:STRING,is_contract:BOOLEAN,scam_type:STRING' "
    f"{PROJECT}:{DATASET}.address_labels {outfile}\n"
)

# %%
sql = textwrap.dedent(f"""
/* ========================================================================
   Address-window feature dataset with “meaningful zeros” и proper NULLs
   (→ NaN в pandas) where a metric could not be рассчитана.
   ======================================================================== */

-- 0. Таблицы-справочники
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.addresses` AS
SELECT
  LOWER(Address)                        AS address,
  LOWER(FLAG)                           AS flag_text,
  CASE LOWER(FLAG)
    WHEN 'scam'    THEN 1
    WHEN 'legit'   THEN 0
    WHEN 'suspect' THEN 2
  END                                   AS flag_num,
  CAST(is_contract AS BOOL)             AS is_contract,
  scam_type
FROM `{PROJECT}.{DATASET}.address_labels`;

CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.all_contracts` AS
SELECT DISTINCT LOWER(address) AS contract_addr
FROM `bigquery-public-data.crypto_ethereum.contracts`;

-- 1. Построение address_sequences с индивидуальными окнами и учётом любой активности
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.address_sequences` AS

WITH
  address_labels AS (
    SELECT address
    FROM `{PROJECT}.{DATASET}.addresses`
  ),

  all_activity AS (
    -- ETH in
    SELECT LOWER(from_address) AS address, DATE(block_timestamp) AS dt
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE LOWER(from_address) IN (SELECT address FROM address_labels)
    UNION ALL
    -- ETH out
    SELECT LOWER(to_address), DATE(block_timestamp)
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE LOWER(to_address) IN (SELECT address FROM address_labels)
    UNION ALL
    -- ERC20 in
    SELECT LOWER(from_address), DATE(block_timestamp)
    FROM `bigquery-public-data.crypto_ethereum.token_transfers`
    WHERE LOWER(from_address) IN (SELECT address FROM address_labels)
    UNION ALL
    -- ERC20 out
    SELECT LOWER(to_address), DATE(block_timestamp)
    FROM `bigquery-public-data.crypto_ethereum.token_transfers`
    WHERE LOWER(to_address) IN (SELECT address FROM address_labels)
  ),

  address_bounds AS (
    SELECT
      address,
      MIN(dt) AS date_from,
      MAX(dt) AS date_to
    FROM all_activity
    GROUP BY address
  ),

  windows AS (
    SELECT
      address,
      GENERATE_DATE_ARRAY(date_from, date_to, INTERVAL {WINDOW_DAYS} DAY) AS arr
    FROM address_bounds
  ),
  win AS (
    SELECT
      address,
      d AS window_start,
      LEAD(d) OVER(PARTITION BY address ORDER BY d) - 1 AS window_end
    FROM windows, UNNEST(arr) AS d
  ),

  -- ETH raw и теги
  eth_raw AS (
    SELECT
      LOWER(from_address)               AS addr,
      LOWER(to_address)                 AS peer,
      block_timestamp                   AS ts,
      CAST(value AS FLOAT64)/1e18       AS val,
      TRUE                              AS is_sender,
      receipt_contract_address IS NOT NULL AS created_contract
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE LOWER(from_address) IN (SELECT address FROM address_bounds)
       OR LOWER(to_address)   IN (SELECT address FROM address_bounds)
  ),
  eth_tag AS (
    SELECT e.*,
           IF(c.contract_addr IS NULL, FALSE, TRUE) AS peer_is_contract
    FROM eth_raw e
    LEFT JOIN `{PROJECT}.{DATASET}.all_contracts` c
      ON e.peer = c.contract_addr
  ),

  -- рассчитываем GAP между транзакциями
  eth_lag AS (
    SELECT
      e.*,
      TIMESTAMP_DIFF(
        ts,
        LAG(ts) OVER (PARTITION BY addr, is_sender ORDER BY ts),
        MINUTE
      ) AS gap
    FROM eth_tag e
  ),

  -- агрегация ETH по оконной схеме
  eth_win AS (
    SELECT
      w.address,
      w.window_start,
      COUNTIF(is_sender)                               AS Sent_tnx,
      COUNTIF(NOT is_sender)                           AS Received_tnx,
      COUNTIF(created_contract AND is_sender)          AS Number_of_Created_Contracts,
      COUNT(DISTINCT IF(is_sender, peer, NULL))        AS Unique_Sent_To_Addresses,
      COUNT(DISTINCT IF(NOT is_sender, peer, NULL))    AS Unique_Received_From_Addresses,
      MIN(IF(NOT is_sender, val, NULL))                AS Min_Value_Received,
      MAX(IF(NOT is_sender, val, NULL))                AS Max_Value_Received,
      AVG(IF(NOT is_sender, val, NULL))                AS Avg_Value_Received,
      MIN(IF(is_sender, val, NULL))                    AS Min_Val_Sent,
      MAX(IF(is_sender, val, NULL))                    AS Max_Val_Sent,
      AVG(IF(is_sender, val, NULL))                    AS Avg_Val_Sent,
      MIN(IF(is_sender AND peer_is_contract, val, NULL)) AS Min_Value_Sent_To_Contract,
      MAX(IF(is_sender AND peer_is_contract, val, NULL)) AS Max_Value_Sent_To_Contract,
      AVG(IF(is_sender AND peer_is_contract, val, NULL)) AS Avg_Value_Sent_To_Contract,
      SUM(IF(is_sender AND peer_is_contract, val, 0))   AS Total_Ether_Sent_Contracts,
      SUM(IF(is_sender, val, 0))                       AS Total_Ether_Sent,
      SUM(IF(NOT is_sender, val, 0))                   AS Total_Ether_Received,
      COUNT(*)                                         AS Total_Transactions_Including_Tnx_to_Create_Contract,
      TIMESTAMP_DIFF(MAX(ts), MIN(ts), MINUTE)         AS Time_Diff_between_first_and_last_Mins,
      AVG(IF(is_sender, gap, NULL))                    AS Avg_min_between_sent_tnx,
      AVG(IF(NOT is_sender, gap, NULL))                AS Avg_min_between_received_tnx
    FROM eth_lag e
    JOIN win w
      ON e.addr = w.address
     AND DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY w.address, w.window_start
  ),

  -- ERC-20 raw и теги
  tok AS (
    SELECT LOWER(address) AS token_addr,
           COALESCE(SAFE_CAST(decimals AS INT64), 0) AS decs
    FROM `bigquery-public-data.crypto_ethereum.tokens`
  ),
  erc_raw AS (
    SELECT
      LOWER(from_address)  AS sender,
      LOWER(to_address)    AS receiver,
      block_timestamp      AS ts,
      CAST(value AS FLOAT64)/POW(10, t.decs) AS val
    FROM `bigquery-public-data.crypto_ethereum.token_transfers` r
    JOIN tok t
      ON LOWER(r.token_address) = t.token_addr
    WHERE LOWER(from_address) IN (SELECT address FROM address_bounds)
       OR LOWER(to_address)   IN (SELECT address FROM address_bounds)
  ),
  erc_tag AS (
    SELECT e.*,
           IF(c.contract_addr IS NULL, FALSE, TRUE) AS peer_is_contract
    FROM erc_raw e
    LEFT JOIN `{PROJECT}.{DATASET}.all_contracts` c
      ON e.receiver = c.contract_addr
  ),
  erc_win AS (
    SELECT
      w.address,
      w.window_start,
      COUNTIF(sender = w.address)               AS ERC20_Total_Ether_Sent_Tnx,
      COUNTIF(receiver = w.address)             AS ERC20_Total_Ether_Recv_Tnx,
      SUM(IF(sender = w.address, val, 0))       AS ERC20_Total_Ether_Sent,
      SUM(IF(receiver = w.address, val, 0))     AS ERC20_Total_Ether_Received,
      SUM(IF(sender = w.address AND peer_is_contract, val, 0)) AS ERC20_Total_Ether_Sent_Contract,
      COUNT(DISTINCT IF(sender = w.address, receiver, NULL))  AS ERC20_Uniq_Sent_Addr,
      COUNT(DISTINCT IF(receiver = w.address, sender, NULL))  AS ERC20_Uniq_Rec_Addr,
      COUNT(DISTINCT IF(receiver = w.address AND peer_is_contract, sender, NULL))
                                               AS ERC20_Uniq_Rec_Contract_Addr,
      MIN(IF(receiver = w.address, val, NULL)) AS ERC20_Min_Val_Rec,
      MAX(IF(receiver = w.address, val, NULL)) AS ERC20_Max_Val_Rec,
      AVG(IF(receiver = w.address, val, NULL)) AS ERC20_Avg_Val_Rec,
      MIN(IF(sender = w.address, val, NULL))   AS ERC20_Min_Val_Sent,
      MAX(IF(sender = w.address, val, NULL))   AS ERC20_Max_Val_Sent,
      AVG(IF(sender = w.address, val, NULL))   AS ERC20_Avg_Val_Sent
    FROM erc_tag e
    JOIN win w
      ON (e.sender = w.address OR e.receiver = w.address)
     AND DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY w.address, w.window_start
  ),

  -- Объединение ETH + ERC20
  full_win AS (
    SELECT
      w.address,
      w.window_start,

      -- ETH метрики
      COALESCE(e.Sent_tnx, 0)                              AS Sent_tnx,
      COALESCE(e.Received_tnx, 0)                          AS Received_tnx,
      COALESCE(e.Number_of_Created_Contracts, 0)           AS Number_of_Created_Contracts,
      COALESCE(e.Unique_Sent_To_Addresses, 0)              AS Unique_Sent_To_Addresses,
      COALESCE(e.Unique_Received_From_Addresses, 0)        AS Unique_Received_From_Addresses,
      e.Min_Value_Received                                 AS Min_Value_Received,
      e.Max_Value_Received                                 AS Max_Value_Received,
      e.Avg_Value_Received                                 AS Avg_Value_Received,
      e.Min_Val_Sent                                       AS Min_Val_Sent,
      e.Max_Val_Sent                                       AS Max_Val_Sent,
      e.Avg_Val_Sent                                       AS Avg_Val_Sent,
      e.Min_Value_Sent_To_Contract                         AS Min_Value_Sent_To_Contract,
      e.Max_Value_Sent_To_Contract                         AS Max_Value_Sent_To_Contract,
      e.Avg_Value_Sent_To_Contract                         AS Avg_Value_Sent_To_Contract,
      e.Total_Ether_Sent_Contracts                         AS Total_Ether_Sent_Contracts,
      e.Total_Ether_Sent                                   AS Total_Ether_Sent,
      e.Total_Ether_Received                               AS Total_Ether_Received,
      e.Time_Diff_between_first_and_last_Mins              AS Time_Diff_between_first_and_last_Mins,
      e.Avg_min_between_sent_tnx                           AS Avg_min_between_sent_tnx,
      e.Avg_min_between_received_tnx                       AS Avg_min_between_received_tnx,

      -- ERC-20 метрики
      COALESCE(r.ERC20_Total_Ether_Sent_Tnx, 0)             AS ERC20_Total_Ether_Sent_Tnx,
      COALESCE(r.ERC20_Total_Ether_Recv_Tnx, 0)             AS ERC20_Total_Ether_Recv_Tnx,
      COALESCE(r.ERC20_Total_Ether_Sent, 0)                 AS ERC20_Total_Ether_Sent,
      COALESCE(r.ERC20_Total_Ether_Received, 0)             AS ERC20_Total_Ether_Received,
      COALESCE(r.ERC20_Total_Ether_Sent_Contract, 0)        AS ERC20_Total_Ether_Sent_Contract,
      COALESCE(r.ERC20_Uniq_Sent_Addr, 0)                   AS ERC20_Uniq_Sent_Addr,
      COALESCE(r.ERC20_Uniq_Rec_Addr, 0)                    AS ERC20_Uniq_Rec_Addr,
      COALESCE(r.ERC20_Uniq_Rec_Contract_Addr, 0)           AS ERC20_Uniq_Rec_Contract_Addr,
      r.ERC20_Min_Val_Rec                                   AS ERC20_Min_Val_Rec,
      r.ERC20_Max_Val_Rec                                   AS ERC20_Max_Val_Rec,
      r.ERC20_Avg_Val_Rec                                   AS ERC20_Avg_Val_Rec,
      r.ERC20_Min_Val_Sent                                  AS ERC20_Min_Val_Sent,
      r.ERC20_Max_Val_Sent                                  AS ERC20_Max_Val_Sent,
      r.ERC20_Avg_Val_Sent                                  AS ERC20_Avg_Val_Sent
    FROM win w
    LEFT JOIN eth_win e
      ON w.address = e.address AND w.window_start = e.window_start
    LEFT JOIN erc_win r
      ON w.address = r.address AND w.window_start = r.window_start
  ),

  -- 8. Нумерация окон и усечение до max_steps
  sequenced AS (
    SELECT
      fw.*,
      ROW_NUMBER() OVER(
        PARTITION BY fw.address
        ORDER BY fw.window_start DESC
      ) - 1 AS idx_desc
    FROM full_win fw
  )

SELECT
  a.address,
  a.flag_text   AS FLAG,
  a.flag_num    AS FLAG_NUM,
  a.is_contract,
  a.scam_type,
  ({MAX_STEPS} - 1) - seq.idx_desc                AS step_idx,
  seq.window_start,
  seq.Sent_tnx,
  seq.Received_tnx,
  seq.Number_of_Created_Contracts,
  seq.Unique_Sent_To_Addresses,
  seq.Unique_Received_From_Addresses,
  seq.Min_Value_Received,
  seq.Max_Value_Received,
  seq.Avg_Value_Received,
  seq.Min_Val_Sent,
  seq.Max_Val_Sent,
  seq.Avg_Val_Sent,
  seq.Min_Value_Sent_To_Contract,
  seq.Max_Value_Sent_To_Contract,
  seq.Avg_Value_Sent_To_Contract,
  seq.Total_Ether_Sent_Contracts,
  seq.Total_Ether_Sent,
  seq.Total_Ether_Received,
  seq.Time_Diff_between_first_and_last_Mins,
  seq.Avg_min_between_sent_tnx,
  seq.Avg_min_between_received_tnx,
  seq.ERC20_Total_Ether_Sent_Tnx,
  seq.ERC20_Total_Ether_Recv_Tnx,
  seq.ERC20_Total_Ether_Sent,
  seq.ERC20_Total_Ether_Received,
  seq.ERC20_Total_Ether_Sent_Contract,
  seq.ERC20_Uniq_Sent_Addr,
  seq.ERC20_Uniq_Rec_Addr,
  seq.ERC20_Uniq_Rec_Contract_Addr,
  seq.ERC20_Min_Val_Rec,
  seq.ERC20_Max_Val_Rec,
  seq.ERC20_Avg_Val_Rec,
  seq.ERC20_Min_Val_Sent,
  seq.ERC20_Max_Val_Sent,
  seq.ERC20_Avg_Val_Sent
FROM sequenced seq
JOIN `{PROJECT}.{DATASET}.addresses` a
  ON seq.address = a.address
WHERE seq.idx_desc < {MAX_STEPS}
ORDER BY a.address, step_idx;
""")

# Сохраняем в файл
out_sql = Path("./bq/create_address_sequences.sql")
out_sql.write_text(sql, encoding="utf-8")
print(f"SQL-файл сохранён → {out_sql.resolve()}")

# %%
import pandas as pd

# Загрузите CSV
df = pd.read_csv("./lstm/lstm_sequence.csv")  # замените на путь к вашему файлу

# Убедимся, что is_contract — булевый тип
df['is_contract'] = df['is_contract'].astype(bool)

# Фильтрация по условиям
filtered = df[(df['is_contract'] == True) & (df['step_idx'] == 29)]

# Общее количество
total_count = len(filtered)

# Количество scam и legit
scam_count = (filtered['FLAG'] == 'scam').sum()
legit_count = (filtered['FLAG'] == 'legit').sum()

# Вывод результатов
print(f"Всего is_contract=True и step_idx=29: {total_count}")
print(f"Из них scam: {scam_count}")
print(f"Из них legit: {legit_count}")


# %%
import pandas as pd
import numpy as np

# Загрузите свой CSV-файл (замените путь на ваш)
df = pd.read_csv("./lstm/lstm_sequence.csv") 

# Преобразуем типы
df['is_contract'] = df['is_contract'].astype(bool)
df['step_idx'] = pd.to_numeric(df['step_idx'], errors='coerce')

# Выбираем числовые столбцы
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Подсчёт статистик и выявление выбросов по методу IQR
summary = []
thresholds = {}
for col in num_cols:
    series = df[col].dropna()
    count = series.count()
    mean = series.mean()
    std = series.std()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    thresholds[col] = (lower, upper)
    outliers = ((series < lower) | (series > upper)).sum()
    missing = df[col].isnull().sum()
    pct_outliers = (outliers / count * 100) if count > 0 else 0

    summary.append({
        'column': col,
        'count': count,
        'mean': mean,
        'std': std,
        'min': series.min(),
        'max': series.max(),
        'missing': missing,
        'outlier_count': outliers,
        'outlier_pct': pct_outliers
    })

summary_df = pd.DataFrame(summary)
print(summary_df)  # или display(summary_df) в Jupyter


# Фильтрация подозрительных записей (любой выброс в числовых полях)
outlier_masks = {
    col: ((df[col] < lower) | (df[col] > upper))
    for col, (lower, upper) in thresholds.items()
}
outlier_any = pd.DataFrame(outlier_masks).any(axis=1)
df['outlier_any'] = outlier_any

# Подсчёт аномалий для scam-контрактов
scam_contracts = df[(df['is_contract']) & (df['FLAG'] == 'scam')]
scam_outliers = scam_contracts[scam_contracts['outlier_any']]

print(f"Всего записей scam-контрактов: {len(scam_contracts)}")
print(f"Аномалий в scam-контрактах (по IQR): {len(scam_outliers)}")
print("\nПримеры аномальных записей:")
print(scam_outliers.head())


# %% [markdown]
# # Etherscan

# %%
%pip install aiolimiter


# %%
import asyncio
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter

# Конфигурация
API_KEY = "NNBVITD832QR2IDMWYZJF12TRWQT6ZQ9TF"
BASE_URL = "https://api.etherscan.io/api"
INPUT_CSV = "./data2/merged_6_dataset.csv"
OUTPUT_CSV = "./data2/etherscan_dataset.csv"

WINDOW_SIZE_DAYS = 7
NUM_STEPS = 30
DAILY_LIMIT = 99950
RETRY_DELAY = 1

# Ограничения
limiter = AsyncLimiter(max_rate=5, time_period=1)
REQUEST_COUNT = 0

class DailyLimitReached(Exception):
    pass

async def fetch_with_retry(session, params, max_retries=10):
    global REQUEST_COUNT
    retries = 0
    while True:
        async with limiter:
            async with session.get(BASE_URL, params=params) as resp:
                data = await resp.json()
            REQUEST_COUNT += 1
            if REQUEST_COUNT >= DAILY_LIMIT:
                raise DailyLimitReached()

        status = data.get("status")
        message = data.get("message", "")

        if status == "1":
            return data.get("result", [])
        if status == "0" and "No transactions found" in message:
            return []

        retries += 1
        if retries >= max_retries:
            print(f"⚠️  Repeated failure for address: {params.get('address')} action: {params.get('action')}, message: {message}")
            return []  # ← безопасно вернуть пустой список

        await asyncio.sleep(RETRY_DELAY)


async def process_address(row, session):
    addr = row['Address']
    flag = row['FLAG']
    is_contract_flag = row['is_contract']

    params_tx = {"module":"account","action":"txlist","address":addr,
                 "startblock":0,"endblock":99999999,"sort":"asc","apikey":API_KEY}
    params_int = {"module":"account","action":"txlistinternal","address":addr,
                  "startblock":0,"endblock":99999999,"sort":"asc","apikey":API_KEY}
    params_erc = {"module":"account","action":"tokentx","address":addr,
                  "startblock":0,"endblock":99999999,"sort":"asc","apikey":API_KEY}
    params_bal = {"module":"account","action":"balance","address":addr,
                  "tag":"latest","apikey":API_KEY}

    txs, internals, erc20, balance_raw = await asyncio.gather(
        fetch_with_retry(session, params_tx),
        fetch_with_retry(session, params_int),
        fetch_with_retry(session, params_erc),
        fetch_with_retry(session, params_bal)
    )

    try:
        balance_eth = int(balance_raw) / 1e18
    except Exception:
        balance_eth = 0.0

    df_tx = pd.DataFrame(txs)
    if not df_tx.empty:
        df_tx['value_eth'] = df_tx['value'].astype(float) / 1e18
        df_tx['timestamp'] = pd.to_datetime(df_tx['timeStamp'].astype(int), unit='s')
    df_int = pd.DataFrame(internals)
    df_int['timestamp'] = pd.to_datetime(df_int['timeStamp'].astype(int), unit='s') if 'timeStamp' in df_int else pd.Series([pd.NaT] * len(df_int))
    df_erc = pd.DataFrame(erc20)
    if not df_erc.empty and 'timeStamp' in df_erc.columns:
        df_erc['timestamp'] = pd.to_datetime(df_erc['timeStamp'].astype(int), unit='s')
        df_erc['value_num'] = df_erc['value'].astype(float) / (
            10 ** pd.to_numeric(df_erc['tokenDecimal'], errors='coerce').fillna(0).astype(int)
        )
    else:
        df_erc['timestamp'] = pd.Series([pd.NaT] * len(df_erc))
        df_erc['value_num'] = pd.Series([0.0] * len(df_erc))

    if not df_tx.empty:
        first_activity = df_tx['timestamp'].min()
        last_activity = df_tx['timestamp'].max()
        window_starts = pd.date_range(
            start=first_activity,
            end=last_activity - pd.Timedelta(days=WINDOW_SIZE_DAYS),
            periods=NUM_STEPS
        ).to_pydatetime().tolist()
    else:
        first_activity = pd.NaT
        last_activity = pd.NaT
        window_starts = [pd.NaT] * NUM_STEPS

    results = []
    for step_idx, window_start in enumerate(window_starts):
        window_end = window_start + pd.Timedelta(days=WINDOW_SIZE_DAYS) if pd.notna(window_start) else pd.NaT

        if pd.isna(window_start):
            df_tx_w = pd.DataFrame(columns=df_tx.columns)
            df_int_w = pd.DataFrame(columns=df_int.columns)
            df_erc_w = pd.DataFrame(columns=df_erc.columns)
        else:
            df_tx_w = df_tx[(df_tx['timestamp'] >= window_start) & (df_tx['timestamp'] < window_end)]
            df_int_w = df_int[(df_int['timestamp'] >= window_start) & (df_int['timestamp'] < window_end)]
            df_erc_w = df_erc[(df_erc['timestamp'] >= window_start) & (df_erc['timestamp'] < window_end)]

        sent_tx = df_tx_w[df_tx_w['from'].str.lower() == addr.lower()] if not df_tx_w.empty else pd.DataFrame()
        recv_tx = df_tx_w[df_tx_w['to'].str.lower() == addr.lower()] if not df_tx_w.empty else pd.DataFrame()
        sent_count = len(sent_tx)
        recv_count = len(recv_tx)
        def stats(df, col):
            return (df[col].min() if not df.empty else 0,
                    df[col].max() if not df.empty else 0,
                    df[col].mean() if not df.empty else 0)
        min_sent, max_sent, avg_sent = stats(sent_tx, 'value_eth')
        min_recv, max_recv, avg_recv = stats(recv_tx, 'value_eth')
        unique_sent_to = sent_tx['to'].nunique() if sent_count else 0
        unique_recv_from = recv_tx['from'].nunique() if recv_count else 0

        created_contracts = df_int_w[df_int_w.get('type') == 'create'].shape[0] if not df_int_w.empty else 0
        total_with_create = sent_count + recv_count + created_contracts

        erc_total = len(df_erc_w)
        erc_recv = df_erc_w[df_erc_w['to'].str.lower() == addr.lower()] if erc_total else pd.DataFrame()
        erc_sent = df_erc_w[df_erc_w['from'].str.lower() == addr.lower()] if erc_total else pd.DataFrame()
        min_erc_recv, max_erc_recv, avg_erc_recv = stats(erc_recv, 'value_num')
        min_erc_sent, max_erc_sent, avg_erc_sent = stats(erc_sent, 'value_num')
        uniq_erc_sent = erc_sent['to'].nunique() if not erc_sent.empty else 0
        uniq_erc_recv = erc_recv['from'].nunique() if not erc_recv.empty else 0

        results.append({
            'Address': addr,
            'FLAG': flag,
            'is_contract': is_contract_flag,
            'first_activity': first_activity,
            'last_activity': last_activity,
            'step_idx': step_idx,
            'window_start': window_start,
            'Sent_tnx': sent_count,
            'Received_tnx': recv_count,
            'Number_of_Created_Contracts': created_contracts,
            'Unique_Sent_To_Addresses': unique_sent_to,
            'Unique_Received_From_Addresses': unique_recv_from,
            'Min_Value_Received': min_recv,
            'Max_Value_Received': max_recv,
            'Avg_Value_Received': avg_recv,
            'Min_Val_Sent': min_sent,
            'Max_Val_Sent': max_sent,
            'Avg_Val_Sent': avg_sent,
            'Total_Transactions_Including_Tnx_to_Create_Contract': total_with_create,
            'Total_Ether_Balance': balance_eth,
            'Total_ERC20_Tnxs': erc_total,
            'ERC20_Uniq_Sent_Addr': uniq_erc_sent,
            'ERC20_Uniq_Rec_Addr': uniq_erc_recv,
            'ERC20_Min_Val_Rec': min_erc_recv,
            'ERC20_Max_Val_Rec': max_erc_recv,
            'ERC20_Avg_Val_Rec': avg_erc_recv,
            'ERC20_Min_Val_Sent': min_erc_sent,
            'ERC20_Max_Val_Sent': max_erc_sent,
            'ERC20_Avg_Val_Sent': avg_erc_sent
        })
    return results

async def main():
    global REQUEST_COUNT
    df_input = pd.read_csv(INPUT_CSV, dtype={"Address": str, "FLAG": str})
    df_input['is_contract'] = df_input['is_contract'].map({True: True, False: False, 'True': True, 'False': False})

    try:
        df_existing = pd.read_csv(OUTPUT_CSV)
        last_address = df_existing['Address'].iloc[-1]
        df_existing = df_existing[df_existing['Address'] != last_address]
        df_existing.to_csv(OUTPUT_CSV, index=False)
        processed = set(df_existing['Address'].unique())
        start_index = df_input[df_input['Address'] == last_address].index[0]
        df_input = df_input.iloc[start_index:]
        print(f"Resuming from last address: {last_address} (index {start_index})")
    except (FileNotFoundError, IndexError):
        processed = set()
        df_existing = pd.DataFrame()
        pd.DataFrame(columns=[
            'Address','FLAG','is_contract','first_activity','last_activity',
            'step_idx','window_start','Sent_tnx','Received_tnx',
            'Number_of_Created_Contracts','Unique_Sent_To_Addresses',
            'Unique_Received_From_Addresses','Min_Value_Received','Max_Value_Received',
            'Avg_Value_Received','Min_Val_Sent','Max_Val_Sent','Avg_Val_Sent',
            'Total_Transactions_Including_Tnx_to_Create_Contract','Total_Ether_Balance',
            'Total_ERC20_Tnxs','ERC20_Uniq_Sent_Addr','ERC20_Uniq_Rec_Addr',
            'ERC20_Min_Val_Rec','ERC20_Max_Val_Rec','ERC20_Avg_Val_Rec',
            'ERC20_Min_Val_Sent','ERC20_Max_Val_Sent','ERC20_Avg_Val_Sent'
        ]).to_csv(OUTPUT_CSV, index=False)

    total = len(df_input)
    count = 0
    async with aiohttp.ClientSession() as session:
        for _, row in df_input.iterrows():
            addr = row['Address']
            count += 1
            try:
                results = await process_address(row, session)
            except DailyLimitReached:
                print(f"⛔️ DAILY LIMIT REACHED after {REQUEST_COUNT} requests. Stopping at address {addr}.")
                break
            pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', index=False, header=False)
            if count % 100 == 0 or count == total:
                print(f"✅ Processed {count}/{total} addresses. Output rows: {sum(1 for _ in open(OUTPUT_CSV)) - 1}")

    print(f"✅ Finished (or paused) at {count}/{total}. Total requests used: {REQUEST_COUNT}.")

# Для Jupyter:
await main()


# %%
import pandas as pd
import numpy as np

# Загрузка датасета
df = pd.read_csv('./data2/etherscan_dataset.csv')

# Список транзакционных столбцов
txn_columns = [
    'Sent_tnx', 'Received_tnx', 'Number_of_Created_Contracts',
    'Unique_Sent_To_Addresses', 'Unique_Received_From_Addresses',
    'Min_Value_Received', 'Max_Value_Received', 'Avg_Value_Received',
    'Min_Val_Sent', 'Max_Val_Sent', 'Avg_Val_Sent',
    'Total_Transactions_Including_Tnx_to_Create_Contract',
    'Total_Ether_Balance', 'Total_ERC20_Tnxs', 'ERC20_Uniq_Sent_Addr',
    'ERC20_Uniq_Rec_Addr', 'ERC20_Min_Val_Rec', 'ERC20_Max_Val_Rec', 'ERC20_Avg_Val_Rec',
    'ERC20_Min_Val_Sent', 'ERC20_Max_Val_Sent', 'ERC20_Avg_Val_Sent'
]

# Удаление строк с более чем 2/3 нулевых или NaN значений в транзакционных столбцах
threshold = int(2 / 3 * len(txn_columns))

def count_zero_or_nan(row):
    return sum((pd.isna(row[col]) or row[col] == 0) for col in txn_columns)

df_filtered = df[df.apply(count_zero_or_nan, axis=1) <= threshold]

# Сохранение результата
df_filtered.to_csv('./data2/filtered_etherscan_dataset.csv', index=False)



