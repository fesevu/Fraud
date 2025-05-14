
/* ========================================================================
   Address-window feature dataset with “meaningful zeros” и proper NULLs
   (→ NaN в pandas) where a metric could not be рассчитана.
   ======================================================================== */

-- 0. Таблицы-справочники
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.addresses` AS
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
FROM `celtic-tendril-459507-q8.fraud_lstm.address_labels`;

CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.all_contracts` AS
SELECT DISTINCT LOWER(address) AS contract_addr
FROM `bigquery-public-data.crypto_ethereum.contracts`;

-- 1. Построение address_sequences с индивидуальными окнами и учётом любой активности
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.address_sequences` AS

WITH
  address_labels AS (
    SELECT address
    FROM `celtic-tendril-459507-q8.fraud_lstm.addresses`
  ),

  all_activity AS (
    -- ETH in+out
    SELECT LOWER(from_address) AS address, DATE(block_timestamp) AS dt
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE LOWER(from_address) IN (SELECT address FROM address_labels)
    UNION ALL
    SELECT LOWER(to_address), DATE(block_timestamp)
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE LOWER(to_address) IN (SELECT address FROM address_labels)
    -- ERC20 in+out
    UNION ALL
    SELECT LOWER(from_address), DATE(block_timestamp)
    FROM `bigquery-public-data.crypto_ethereum.token_transfers`
    WHERE LOWER(from_address) IN (SELECT address FROM address_labels)
    UNION ALL
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
      GENERATE_DATE_ARRAY(date_from, date_to, INTERVAL 7 DAY) AS arr
    FROM address_bounds
  ),
  win AS (
    SELECT
      address,
      d AS window_start,
      LEAD(d) OVER(PARTITION BY address ORDER BY d) - 1 AS window_end
    FROM windows, UNNEST(arr) AS d
  ),

  -- ETH raw + tag
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
    LEFT JOIN `celtic-tendril-459507-q8.fraud_lstm.all_contracts` c
      ON e.peer = c.contract_addr
  ),

  -- Разворачиваем LAG() в отдельное CTE
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

  -- Собственно агрегация ETH по вашим окнам
  eth_win AS (
    SELECT
      w.address,
      w.window_start,
      COUNTIF(is_sender)                      AS Sent_tnx,
      COUNTIF(NOT is_sender)                  AS Received_tnx,
      COUNTIF(created_contract AND is_sender) AS Number_of_Created_Contracts,
      COUNT(DISTINCT IF(is_sender, peer, NULL))     AS Unique_Sent_To_Addresses,
      COUNT(DISTINCT IF(NOT is_sender, peer, NULL)) AS Unique_Received_From_Addresses,
      MIN(IF(NOT is_sender, val, NULL))       AS Min_Value_Received,
      MAX(IF(NOT is_sender, val, NULL))       AS Max_Value_Received,
      AVG(IF(NOT is_sender, val, NULL))       AS Avg_Value_Received,
      MIN(IF(is_sender, val, NULL))           AS Min_Val_Sent,
      MAX(IF(is_sender, val, NULL))           AS Max_Val_Sent,
      AVG(IF(is_sender, val, NULL))           AS Avg_Val_Sent,
      MIN(IF(is_sender AND peer_is_contract, val, NULL)) AS Min_Value_Sent_To_Contract,
      MAX(IF(is_sender AND peer_is_contract, val, NULL)) AS Max_Value_Sent_To_Contract,
      AVG(IF(is_sender AND peer_is_contract, val, NULL)) AS Avg_Value_Sent_To_Contract,
      SUM(IF(is_sender AND peer_is_contract, val, 0))    AS Total_Ether_Sent_Contracts,
      SUM(IF(is_sender, val, 0))              AS Total_Ether_Sent,
      SUM(IF(NOT is_sender, val, 0))          AS Total_Ether_Received,
      COUNT(*)                                AS Total_Transactions_Including_Tnx_to_Create_Contract,
      TIMESTAMP_DIFF(MAX(ts), MIN(ts), MINUTE) AS Time_Diff_between_first_and_last_Mins,
      AVG(IF(is_sender, gap, NULL))           AS Avg_min_between_sent_tnx,
      AVG(IF(NOT is_sender, gap, NULL))       AS Avg_min_between_received_tnx
    FROM eth_lag e
    JOIN win w
      ON e.addr       = w.address
     AND DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY w.address, w.window_start
  ),

  -- ERC-20: без изменений в логике gap
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
    LEFT JOIN `celtic-tendril-459507-q8.fraud_lstm.all_contracts` c
      ON e.receiver = c.contract_addr
  ),
  erc_win AS (
    SELECT
      w.address,
      w.window_start,
      COUNTIF(sender = w.address)           AS ERC20_Total_Ether_Sent_Tnx,
      COUNTIF(receiver = w.address)         AS ERC20_Total_Ether_Recv_Tnx,
      SUM(IF(sender = w.address, val, 0))   AS ERC20_Total_Ether_Sent,
      SUM(IF(receiver = w.address, val, 0)) AS ERC20_Total_Ether_Received,
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
      ON (e.sender   = w.address OR e.receiver = w.address)
     AND DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY w.address, w.window_start
  ),

  -- Объединяем ETH + ERC20
  full_win AS (
    SELECT
      w.address,
      w.window_start,
      COALESCE(e.Sent_tnx, 0)                AS Sent_tnx,
      COALESCE(e.Received_tnx, 0)            AS Received_tnx,
      /* … остальные поля по аналогии … */
      COALESCE(r.ERC20_Total_Ether_Sent_Tnx, 0)   AS ERC20_Total_Ether_Sent_Tnx,
      COALESCE(r.ERC20_Total_Ether_Recv_Tnx, 0)   AS ERC20_Total_Ether_Recv_Tnx,
      /* … и т.д. … */
    FROM win w
    LEFT JOIN eth_win e
      ON w.address = e.address AND w.window_start = e.window_start
    LEFT JOIN erc_win r
      ON w.address = r.address AND w.window_start = r.window_start
  ),

  -- Нумерация окон и усечение до max_steps
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
  (40 - 1) - seq.idx_desc  AS step_idx,
  seq.window_start,
  seq.Sent_tnx, seq.Received_tnx,
  /* … остальные поля … */
FROM sequenced seq
JOIN `celtic-tendril-459507-q8.fraud_lstm.addresses` a
  ON seq.address = a.address
WHERE seq.idx_desc < 40
ORDER BY a.address, step_idx;
