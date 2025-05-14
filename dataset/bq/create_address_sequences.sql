
---------------------------------------------------------------------------
DECLARE date_from DATE DEFAULT DATE '2015-01-01';
DECLARE date_to   DATE DEFAULT DATE '2025-01-01';
DECLARE win_days  INT64 DEFAULT 7;
DECLARE max_steps INT64 DEFAULT 30;
---------------------------------------------------------------------------

-- 0. Справочники
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

-- 1–5. Создаём полные LSTM-последовательности
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.address_sequences` AS
WITH
  -- 1. Окна
  windows AS (
    SELECT GENERATE_DATE_ARRAY(date_from, date_to, INTERVAL win_days DAY) AS arr
  ),
  win AS (
    SELECT d AS window_start,
           LEAD(d) OVER (ORDER BY d) - 1 AS window_end
    FROM windows, UNNEST(arr) AS d
  ),

  -- 2. Нормальные ETH-транзакции
  eth_raw AS (
    SELECT
      LOWER(from_address) AS addr,
      LOWER(to_address)   AS peer,
      block_timestamp     AS ts,
      CAST(value AS FLOAT64)/1e18 AS val,
      TRUE                AS is_sender,
      receipt_contract_address IS NOT NULL AS created_contract
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
    UNION ALL
    SELECT
      LOWER(to_address),
      LOWER(from_address),
      block_timestamp,
      CAST(value AS FLOAT64)/1e18,
      FALSE,
      FALSE
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
  ),
  eth_tag AS (
    SELECT e.*,
           IF(c.contract_addr IS NULL, FALSE, TRUE) AS peer_is_contract
    FROM eth_raw e
    LEFT JOIN `celtic-tendril-459507-q8.fraud_lstm.all_contracts` c
      ON e.peer = c.contract_addr
  ),
  eth_win AS (
    SELECT
      a.address,
      w.window_start,

      COUNTIF(is_sender)                      AS Sent_tnx,
      COUNTIF(NOT is_sender)                  AS Received_tnx,
      COUNTIF(created_contract AND is_sender) AS Number_of_Created_Contracts,
      COUNT(DISTINCT IF(is_sender, peer, NULL))
                                             AS Unique_Sent_To_Addresses,
      COUNT(DISTINCT IF(NOT is_sender, peer, NULL))
                                             AS Unique_Received_From_Addresses,

      MIN(IF(NOT is_sender, val, NULL))       AS Min_Value_Received,
      MAX(IF(NOT is_sender, val, NULL))       AS Max_Value_Received,
      AVG(IF(NOT is_sender, val, NULL))       AS Avg_Value_Received,

      MIN(IF(is_sender, val, NULL))           AS Min_Val_Sent,
      MAX(IF(is_sender, val, NULL))           AS Max_Val_Sent,
      AVG(IF(is_sender, val, NULL))           AS Avg_Val_Sent,

      MIN(IF(is_sender AND peer_is_contract, val, NULL))
                                             AS Min_Value_Sent_To_Contract,
      MAX(IF(is_sender AND peer_is_contract, val, NULL))
                                             AS Max_Value_Sent_To_Contract,
      AVG(IF(is_sender AND peer_is_contract, val, NULL))
                                             AS Avg_Value_Sent_To_Contract,

      SUM(IF(is_sender AND peer_is_contract, val, 0))
                                             AS Total_Ether_Sent_Contracts,
      SUM(IF(is_sender, val, 0))              AS Total_Ether_Sent,
      SUM(IF(NOT is_sender, val, 0))          AS Total_Ether_Received,

      COUNT(*)                                AS Total_Transactions_Including_Tnx_to_Create_Contract,

      TIMESTAMP_DIFF(MAX(ts), MIN(ts), MINUTE)
                                             AS Time_Diff_between_first_and_last_Mins,

      AVG(IF(is_sender, gap, NULL))           AS Avg_min_between_sent_tnx,
      AVG(IF(NOT is_sender, gap, NULL))       AS Avg_min_between_received_tnx
    FROM (
      SELECT e.*,
             TIMESTAMP_DIFF(ts,
               LAG(ts) OVER (PARTITION BY addr, is_sender ORDER BY ts),
             MINUTE) AS gap
      FROM eth_tag e
    ) e
    JOIN `celtic-tendril-459507-q8.fraud_lstm.addresses` a
      ON e.addr = a.address
    JOIN win w
      ON DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY a.address, w.window_start
  ),

  -- 3. ERC-20 transfers
  tok AS (
    SELECT LOWER(address) AS token_addr,
           COALESCE(SAFE_CAST(decimals AS INT64),0) AS decs
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
    WHERE DATE(r.block_timestamp) BETWEEN date_from AND date_to
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
      a.address,
      w.window_start,

      COUNTIF(sender = a.address)           AS ERC20_Total_Ether_Sent_Tnx,
      COUNTIF(receiver = a.address)         AS ERC20_Total_Ether_Recv_Tnx,
      SUM(IF(sender = a.address, val, 0))   AS ERC20_Total_Ether_Sent,
      SUM(IF(receiver = a.address, val, 0)) AS ERC20_Total_Ether_Received,
      SUM(IF(sender = a.address AND peer_is_contract, val, 0))
                                             AS ERC20_Total_Ether_Sent_Contract,
      COUNT(DISTINCT IF(sender = a.address, receiver, NULL))
                                             AS ERC20_Uniq_Sent_Addr,
      COUNT(DISTINCT IF(receiver = a.address, sender, NULL))
                                             AS ERC20_Uniq_Rec_Addr,
      COUNT(DISTINCT IF(receiver = a.address AND peer_is_contract, sender, NULL))
                                             AS ERC20_Uniq_Rec_Contract_Addr,
      MIN(IF(receiver = a.address, val, NULL)) AS ERC20_Min_Val_Rec,
      MAX(IF(receiver = a.address, val, NULL)) AS ERC20_Max_Val_Rec,
      AVG(IF(receiver = a.address, val, NULL)) AS ERC20_Avg_Val_Rec,
      MIN(IF(sender = a.address, val, NULL))   AS ERC20_Min_Val_Sent,
      MAX(IF(sender = a.address, val, NULL))   AS ERC20_Max_Val_Sent,
      AVG(IF(sender = a.address, val, NULL))   AS ERC20_Avg_Val_Sent
    FROM erc_tag e
    JOIN `celtic-tendril-459507-q8.fraud_lstm.addresses` a
      ON e.sender = a.address OR e.receiver = a.address
    JOIN win w
      ON DATE(e.ts) BETWEEN w.window_start AND w.window_end
    GROUP BY a.address, w.window_start
  ),

  -- 4. Полные пары (address, window)
  addr_win AS (
    SELECT a.address, w.window_start
    FROM `celtic-tendril-459507-q8.fraud_lstm.addresses` a
    CROSS JOIN win w
  ),

  full_win AS (
    SELECT
      aw.address,
      aw.window_start,

      COALESCE(eth.Sent_tnx, 0)                                 AS Sent_tnx,
      COALESCE(eth.Received_tnx, 0)                             AS Received_tnx,
      COALESCE(eth.Number_of_Created_Contracts, 0)              AS Number_of_Created_Contracts,
      COALESCE(eth.Unique_Sent_To_Addresses, 0)                 AS Unique_Sent_To_Addresses,
      COALESCE(eth.Unique_Received_From_Addresses, 0)           AS Unique_Received_From_Addresses,
      COALESCE(eth.Min_Value_Received, 0.0)                     AS Min_Value_Received,
      COALESCE(eth.Max_Value_Received, 0.0)                     AS Max_Value_Received,
      COALESCE(eth.Avg_Value_Received, 0.0)                     AS Avg_Value_Received,
      COALESCE(eth.Min_Val_Sent, 0.0)                           AS Min_Val_Sent,
      COALESCE(eth.Max_Val_Sent, 0.0)                           AS Max_Val_Sent,
      COALESCE(eth.Avg_Val_Sent, 0.0)                           AS Avg_Val_Sent,
      COALESCE(eth.Min_Value_Sent_To_Contract, 0.0)             AS Min_Value_Sent_To_Contract,
      COALESCE(eth.Max_Value_Sent_To_Contract, 0.0)             AS Max_Value_Sent_To_Contract,
      COALESCE(eth.Avg_Value_Sent_To_Contract, 0.0)             AS Avg_Value_Sent_To_Contract,
      COALESCE(eth.Total_Ether_Sent_Contracts, 0.0)             AS Total_Ether_Sent_Contracts,
      COALESCE(eth.Total_Ether_Sent, 0.0)                       AS Total_Ether_Sent,
      COALESCE(eth.Total_Ether_Received, 0.0)                   AS Total_Ether_Received,
      COALESCE(eth.Total_Transactions_Including_Tnx_to_Create_Contract, 0) AS Total_Transactions_Including_Tnx_to_Create_Contract,
      COALESCE(eth.Time_Diff_between_first_and_last_Mins, 0)    AS Time_Diff_between_first_and_last_Mins,
      COALESCE(eth.Avg_min_between_sent_tnx, 0.0)               AS Avg_min_between_sent_tnx,
      COALESCE(eth.Avg_min_between_received_tnx, 0.0)           AS Avg_min_between_received_tnx,

      COALESCE(erc.ERC20_Total_Ether_Sent_Tnx, 0)               AS ERC20_Total_Ether_Sent_Tnx,
      COALESCE(erc.ERC20_Total_Ether_Recv_Tnx, 0)               AS ERC20_Total_Ether_Recv_Tnx,
      COALESCE(erc.ERC20_Total_Ether_Sent, 0.0)                 AS ERC20_Total_Ether_Sent,
      COALESCE(erc.ERC20_Total_Ether_Received, 0.0)             AS ERC20_Total_Ether_Received,
      COALESCE(erc.ERC20_Total_Ether_Sent_Contract, 0.0)        AS ERC20_Total_Ether_Sent_Contract,
      COALESCE(erc.ERC20_Uniq_Sent_Addr, 0)                     AS ERC20_Uniq_Sent_Addr,
      COALESCE(erc.ERC20_Uniq_Rec_Addr, 0)                      AS ERC20_Uniq_Rec_Addr,
      COALESCE(erc.ERC20_Uniq_Rec_Contract_Addr, 0)             AS ERC20_Uniq_Rec_Contract_Addr,
      COALESCE(erc.ERC20_Min_Val_Rec, 0.0)                      AS ERC20_Min_Val_Rec,
      COALESCE(erc.ERC20_Max_Val_Rec, 0.0)                      AS ERC20_Max_Val_Rec,
      COALESCE(erc.ERC20_Avg_Val_Rec, 0.0)                      AS ERC20_Avg_Val_Rec,
      COALESCE(erc.ERC20_Min_Val_Sent, 0.0)                     AS ERC20_Min_Val_Sent,
      COALESCE(erc.ERC20_Max_Val_Sent, 0.0)                     AS ERC20_Max_Val_Sent,
      COALESCE(erc.ERC20_Avg_Val_Sent, 0.0)                     AS ERC20_Avg_Val_Sent
    FROM addr_win aw
    LEFT JOIN eth_win eth
      ON aw.address = eth.address
     AND aw.window_start = eth.window_start
    LEFT JOIN erc_win erc
      ON aw.address = erc.address
     AND aw.window_start = erc.window_start
  ),

  -- 6. Нумерация и обрезка до max_steps
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
  (max_steps - 1) - seq.idx_desc AS step_idx,
  seq.window_start,

  -- Ether
  seq.Sent_tnx, seq.Received_tnx,
  seq.Number_of_Created_Contracts,
  seq.Unique_Sent_To_Addresses, seq.Unique_Received_From_Addresses,
  seq.Min_Value_Received, seq.Max_Value_Received, seq.Avg_Value_Received,
  seq.Min_Val_Sent, seq.Max_Val_Sent, seq.Avg_Val_Sent,
  seq.Min_Value_Sent_To_Contract, seq.Max_Value_Sent_To_Contract, seq.Avg_Value_Sent_To_Contract,
  seq.Total_Ether_Sent_Contracts, seq.Total_Ether_Sent, seq.Total_Ether_Received,
  seq.Time_Diff_between_first_and_last_Mins,
  seq.Avg_min_between_sent_tnx, seq.Avg_min_between_received_tnx,
  seq.Total_Transactions_Including_Tnx_to_Create_Contract,
  (seq.Total_Ether_Received - seq.Total_Ether_Sent) AS Total_Ether_Balance,

  -- ERC-20
  (seq.ERC20_Total_Ether_Sent_Tnx + seq.ERC20_Total_Ether_Recv_Tnx) AS Total_ERC20_Tnxs,
  seq.ERC20_Total_Ether_Received, seq.ERC20_Total_Ether_Sent,
  seq.ERC20_Total_Ether_Sent_Contract,
  seq.ERC20_Uniq_Sent_Addr, seq.ERC20_Uniq_Rec_Addr, seq.ERC20_Uniq_Rec_Contract_Addr,
  seq.ERC20_Min_Val_Rec, seq.ERC20_Max_Val_Rec, seq.ERC20_Avg_Val_Rec,
  seq.ERC20_Min_Val_Sent, seq.ERC20_Max_Val_Sent, seq.ERC20_Avg_Val_Sent
FROM sequenced seq
JOIN `celtic-tendril-459507-q8.fraud_lstm.addresses` a
  ON seq.address = a.address
WHERE seq.idx_desc < max_steps
ORDER BY a.address, step_idx;
