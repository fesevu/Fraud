
-- Параметры
DECLARE date_from DATE DEFAULT '2023-01-01';
DECLARE date_to   DATE DEFAULT '2025-04-30';
DECLARE max_hop2  INT64 DEFAULT 300000;

-- 0. Базовая таблица LSTM
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.phishing.lstm` AS
SELECT LOWER(Address) AS addr, CAST(FLAG AS BOOL) AS label
FROM `celtic-tendril-459507-q8.phishing.lstm_addresses`;

-- 1. Hop-1 адреса
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.phishing.hop1` AS
SELECT DISTINCT LOWER(IF(t.from_address IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.lstm`),
                         t.to_address, t.from_address)) AS addr
FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
  AND value > 0
  AND (LOWER(t.from_address) IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.lstm`)
       OR LOWER(t.to_address)  IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.lstm`));

-- 2. Hop-2 (с лимитом по активности)
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.phishing.hop2` AS
SELECT addr FROM (
  SELECT LOWER(IF(t.from_address IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.hop1`),
                  t.to_address, t.from_address)) AS addr,
         COUNT(*) AS tx_cnt
  FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
  WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
    AND value > 0
    AND (LOWER(t.from_address) IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.hop1`)
         OR LOWER(t.to_address)  IN (SELECT addr FROM `celtic-tendril-459507-q8.phishing.hop1`))
  GROUP BY addr
  QUALIFY ROW_NUMBER() OVER(ORDER BY tx_cnt DESC) <= max_hop2
);

-- 3. Итоговые узлы (метка = NULL, если адрес не в LSTM)
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.phishing.gnn_accounts` AS
SELECT addr AS id,
       ANY_VALUE(label) AS label   -- BOOL, может быть NULL
FROM (
  SELECT * FROM `celtic-tendril-459507-q8.phishing.lstm`
  UNION DISTINCT
  SELECT addr, NULL AS label FROM `celtic-tendril-459507-q8.phishing.hop1`
  UNION DISTINCT
  SELECT addr, NULL AS label FROM `celtic-tendril-459507-q8.phishing.hop2`
)
GROUP BY id;

-- 4. Рёбра
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.phishing.gnn_transactions`
PARTITION BY DATE(block_timestamp)
AS
SELECT
  LOWER(t.from_address) AS src,
  LOWER(t.to_address)   AS dst,
  SAFE_DIVIDE(CAST(t.value AS FLOAT64), 1e18) AS amount,
  t.block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
JOIN `celtic-tendril-459507-q8.phishing.gnn_accounts` AS n1
  ON LOWER(t.from_address) = n1.id
JOIN `celtic-tendril-459507-q8.phishing.gnn_accounts` AS n2
  ON LOWER(t.to_address)   = n2.id
WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
  AND value > 0;
