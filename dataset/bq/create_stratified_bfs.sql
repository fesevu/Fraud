
-- Параметры
DECLARE date_from DATE   DEFAULT '2020-01-01';
DECLARE date_to   DATE   DEFAULT '2023-01-01';
DECLARE max_hop2  INT64  DEFAULT 50;

-- 0. Базовая таблица LSTM
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.lstm` AS
SELECT
  LOWER(Address)   AS addr,
  CAST(FLAG AS BOOL) AS is_scam
FROM `celtic-tendril-459507-q8.fraud_gnn.lstm_addresses`;

-- 1. Периоды активности seed-аккаунтов
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.account_activity` AS
SELECT
  id,
  MIN(DATE(block_timestamp)) AS first_seen,
  MAX(DATE(block_timestamp)) AS last_seen
FROM (
  SELECT LOWER(from_address) AS id, block_timestamp
    FROM `bigquery-public-data.crypto_ethereum.transactions`
  UNION ALL
  SELECT LOWER(to_address)   AS id, block_timestamp
    FROM `bigquery-public-data.crypto_ethereum.transactions`
)
WHERE id IN (SELECT addr FROM `celtic-tendril-459507-q8.fraud_gnn.lstm`)
GROUP BY id;

-- 2. Hop-1 по каждому seed-аккаунту
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.hop1` AS
WITH seeds AS (
  SELECT l.addr   AS seed,
         a.first_seen,
         a.last_seen,
         l.is_scam
  FROM `celtic-tendril-459507-q8.fraud_gnn.lstm`  AS l
  JOIN `celtic-tendril-459507-q8.fraud_gnn.account_activity` AS a
    ON l.addr = a.id
)
SELECT DISTINCT
  s.seed,
  LOWER(
    IF(t.from_address = s.seed, t.to_address, t.from_address)
  ) AS hop1_addr,
  s.is_scam
FROM seeds AS s
JOIN `bigquery-public-data.crypto_ethereum.transactions` AS t
  ON DATE(t.block_timestamp) BETWEEN s.first_seen AND s.last_seen
 AND t.value > 0
 AND (
       LOWER(t.from_address) = s.seed
    OR LOWER(t.to_address)   = s.seed
     );

-- 3. Hop-2 с лимитом per seed: безлимит для scam, до 50 для legit
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.hop2` AS
WITH ranked AS (
  SELECT
    h1.seed,
    LOWER(
      IF(t.from_address = h1.hop1_addr, t.to_address, t.from_address)
    ) AS hop2_addr,
    h1.is_scam         AS is_scam_seed,
    COUNT(*)           AS tx_cnt
  FROM `celtic-tendril-459507-q8.fraud_gnn.hop1`       AS h1
  JOIN `bigquery-public-data.crypto_ethereum.transactions` AS t
    ON t.value > 0
   AND (
         LOWER(t.from_address) = h1.hop1_addr
      OR LOWER(t.to_address)   = h1.hop1_addr
       )
  GROUP BY h1.seed, hop2_addr, h1.is_scam
)
SELECT seed, hop2_addr AS addr
FROM ranked
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY seed
    ORDER BY tx_cnt DESC
  ) <= CASE WHEN is_scam_seed THEN 99999999 ELSE 50 END;

-- 4. Собираем итоговый список узлов
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS
SELECT
  addr AS id,
  ANY_VALUE(is_scam) AS is_scam
FROM (
  SELECT addr, is_scam       FROM `celtic-tendril-459507-q8.fraud_gnn.lstm`
  UNION DISTINCT
  SELECT hop1_addr, NULL     FROM `celtic-tendril-459507-q8.fraud_gnn.hop1`
  UNION DISTINCT
  SELECT addr,      NULL     FROM `celtic-tendril-459507-q8.fraud_gnn.hop2`
)
GROUP BY id;

-- 5. Все рёбра между узлами (с тем же partitioning)
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.gnn_transactions`
PARTITION BY DATE(block_timestamp)
AS
SELECT
  LOWER(t.from_address) AS src,
  LOWER(t.to_address)   AS dst,
  SAFE_DIVIDE(CAST(t.value AS FLOAT64), 1e18) AS amount,
  t.block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions` AS t
JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a1
  ON LOWER(t.from_address) = a1.id
JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a2
  ON LOWER(t.to_address)   = a2.id
WHERE t.value > 0
  AND DATE(t.block_timestamp) BETWEEN date_from AND date_to;

-- 6. Стратифицированная выборка ≤150 000 транзакций
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_gnn.gnn_transactions_sample` AS
WITH
  all_tx AS (
    SELECT * FROM `celtic-tendril-459507-q8.fraud_gnn.gnn_transactions`
  ),
  scam_edges AS (
    SELECT t.*
    FROM all_tx AS t
    JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a1
      ON t.src = a1.id
    JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a2
      ON t.dst = a2.id
    WHERE a1.is_scam OR a2.is_scam
  ),
  legit_edges AS (
    SELECT t.*
    FROM all_tx AS t
    JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a1
      ON t.src = a1.id
    JOIN `celtic-tendril-459507-q8.fraud_gnn.gnn_accounts` AS a2
      ON t.dst = a2.id
    WHERE NOT a1.is_scam AND NOT a2.is_scam
  ),
  scam_count AS (
    SELECT COUNT(*) AS cnt FROM scam_edges
  ),
  legit_sample AS (
    SELECT *
    FROM legit_edges
    ORDER BY RAND()
    LIMIT GREATEST(150000 - (SELECT cnt FROM scam_count), 0)
  )
SELECT * FROM scam_edges
UNION ALL
SELECT * FROM legit_sample;
