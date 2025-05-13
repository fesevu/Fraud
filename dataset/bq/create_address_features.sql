
---------------------------------------------------------------------------
-- ПЕРЕМЕННЫЕ ДЛЯ ВСЕХ НИЖЕ ИДУЩИХ STATEMENT-ов
---------------------------------------------------------------------------
DECLARE date_from DATE DEFAULT DATE '2015-07-30';
DECLARE date_to   DATE DEFAULT DATE '2025-04-30';
---------------------------------------------------------------------------

/* 0. адреса с флагами */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.addresses` AS
SELECT
  LOWER(Address)             AS address,
  LOWER(FLAG)                AS flag_text,
  CAST(is_contract AS BOOL)  AS is_contract,
  scam_type,
  CASE
    WHEN LOWER(FLAG) = 'scam'    THEN 1
    WHEN LOWER(FLAG) = 'legit'   THEN 0
    WHEN LOWER(FLAG) = 'suspect' THEN 2
  END                         AS flag_num
FROM `celtic-tendril-459507-q8.fraud_lstm.address_labels`;

/* 1. справочник контрактов */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.all_contracts` AS
SELECT DISTINCT LOWER(address) AS contract_addr
FROM `bigquery-public-data.crypto_ethereum.contracts`;

/* 2. Ether-tx  (укорочено; логика не менялась) */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.tx_eth` AS
WITH base AS (SELECT address FROM `celtic-tendril-459507-q8.fraud_lstm.addresses`)
SELECT
  LOWER(from_address) AS address,
  LOWER(to_address)   AS peer,
  block_timestamp     AS ts,
  CAST(value AS FLOAT64)/1e18 AS eth_value,
  receipt_contract_address IS NOT NULL AS created_contract,
  TRUE  AS is_sender
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
  AND LOWER(from_address) IN (SELECT address FROM base)
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
  AND LOWER(to_address) IN (SELECT address FROM base);

/* 3. Ether-features */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.eth_features` AS
WITH t AS (
  SELECT *,
         peer IN (SELECT contract_addr FROM `celtic-tendril-459507-q8.fraud_lstm.all_contracts`)
           AS peer_is_contract
  FROM `celtic-tendril-459507-q8.fraud_lstm.tx_eth`
),
sent AS (
  SELECT
    address,
    COUNT(*)                                AS Sent_tnx,
    COUNTIF(created_contract)               AS Number_of_Created_Contracts,
    COUNT(DISTINCT peer)                    AS Unique_Sent_To_Addresses,
    MIN(eth_value)                          AS Min_Val_Sent,
    MAX(eth_value)                          AS Max_Val_Sent,
    AVG(eth_value)                          AS Avg_Val_Sent,
    MIN(IF(peer_is_contract, eth_value, NULL))
                                           AS Min_Value_Sent_To_Contract,
    MAX(IF(peer_is_contract, eth_value, NULL))
                                           AS Max_Value_Sent_To_Contract,
    AVG(IF(peer_is_contract, eth_value, NULL))
                                           AS Avg_Value_Sent_To_Contract,
    SUM(IF(peer_is_contract, eth_value, 0)) AS Total_Ether_Sent_Contracts,
    SUM(eth_value)                          AS Total_Ether_Sent
  FROM t WHERE is_sender GROUP BY address
),
recv AS (
  SELECT
    address,
    COUNT(*)                                AS Received_tnx,
    COUNT(DISTINCT peer)                    AS Unique_Received_From_Addresses,
    MIN(eth_value)                          AS Min_Value_Received,
    MAX(eth_value)                          AS Max_Value_Received,
    AVG(eth_value)                          AS Avg_Value_Received,
    SUM(eth_value)                          AS Total_Ether_Received
  FROM t WHERE NOT is_sender GROUP BY address
),
timing AS (
  SELECT
    address,
    TIMESTAMP_DIFF(MAX(ts), MIN(ts), MINUTE)
                                       AS Time_Diff_between_first_and_last_Mins,
    AVG(CASE WHEN is_sender  AND gap IS NOT NULL THEN gap END)
                                       AS Avg_min_between_sent_tnx,
    AVG(CASE WHEN NOT is_sender AND gap IS NOT NULL THEN gap END)
                                       AS Avg_min_between_received_tnx
  FROM (
    SELECT
      address,
      is_sender,
      ts,
      TIMESTAMP_DIFF(ts,
         LAG(ts) OVER (PARTITION BY address,is_sender ORDER BY ts),
         MINUTE) AS gap
    FROM t
  )
  GROUP BY address
),
total AS (
  SELECT
    address,
    COUNT(*) AS Total_Transactions_Including_Tnx_to_Create_Contract
  FROM `celtic-tendril-459507-q8.fraud_lstm.tx_eth`
  GROUP BY address
)
SELECT
  a.address,
  sent.*   EXCEPT(address),
  recv.*   EXCEPT(address),
  timing.* EXCEPT(address),
  total.*  EXCEPT(address),
  (COALESCE(recv.Total_Ether_Received,0.0) -
   COALESCE(sent.Total_Ether_Sent,0.0))     AS Total_Ether_Balance
FROM `celtic-tendril-459507-q8.fraud_lstm.addresses` a
LEFT JOIN sent   USING(address)
LEFT JOIN recv   USING(address)
LEFT JOIN timing USING(address)
LEFT JOIN total  USING(address);

/* 4. ERC-20 transfers — без коррелированных подзапросов, переменные без @ */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.tx_erc20` AS
WITH base AS (
  SELECT address FROM `celtic-tendril-459507-q8.fraud_lstm.addresses`
),
raw AS (
  SELECT
    LOWER(from_address)  AS sender,
    LOWER(to_address)    AS receiver,
    LOWER(token_address) AS token_addr,
    block_timestamp      AS ts,
    CAST(value AS FLOAT64) AS raw_val
  FROM `bigquery-public-data.crypto_ethereum.token_transfers`
  WHERE DATE(block_timestamp) BETWEEN date_from AND date_to
    AND (LOWER(from_address) IN (SELECT address FROM base)
      OR LOWER(to_address)   IN (SELECT address FROM base))
),
tok AS (
  SELECT
    LOWER(address)                          AS token_addr,
    COALESCE(SAFE_CAST(decimals AS INT64),0) AS decs,
    name
  FROM `bigquery-public-data.crypto_ethereum.tokens`
),
joined AS (
  SELECT
    r.sender,
    r.receiver,
    r.ts,
    r.raw_val/POW(10, t.decs)               AS tok_val,
    t.name                                  AS token_name,
    CASE WHEN b.address IS NOT NULL THEN TRUE ELSE FALSE END AS is_sender
  FROM raw AS r
  LEFT JOIN base AS b   ON r.sender = b.address
  LEFT JOIN tok  AS t   ON r.token_addr = t.token_addr
)
SELECT
  CASE WHEN is_sender THEN sender   ELSE receiver END AS address,
  CASE WHEN is_sender THEN receiver ELSE sender   END AS peer,
  is_sender,
  ts,
  tok_val,
  token_name,
  (CASE WHEN is_sender THEN receiver ELSE sender END)
    IN (SELECT contract_addr FROM `celtic-tendril-459507-q8.fraud_lstm.all_contracts`)
    AS peer_is_contract
FROM joined;

/* 5. ERC-20 features */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.erc20_features` AS
WITH b AS (SELECT * FROM `celtic-tendril-459507-q8.fraud_lstm.tx_erc20`),
sent AS (
  SELECT
    address,
    COUNT(*)                    AS ERC20_Total_Ether_Sent_Tnx,
    COUNT(DISTINCT peer)        AS ERC20_Uniq_Sent_Addr,
    MIN(tok_val)                AS ERC20_Min_Val_Sent,
    MAX(tok_val)                AS ERC20_Max_Val_Sent,
    AVG(tok_val)                AS ERC20_Avg_Val_Sent,
    MIN(IF(peer_is_contract,tok_val,NULL))
                               AS ERC20_Min_Val_Sent_Contract,
    MAX(IF(peer_is_contract,tok_val,NULL))
                               AS ERC20_Max_Val_Sent_Contract,
    AVG(IF(peer_is_contract,tok_val,NULL))
                               AS ERC20_Avg_Val_Sent_Contract,
    SUM(tok_val)                AS ERC20_Total_Ether_Sent
  FROM b WHERE is_sender GROUP BY address
),
recv AS (
  SELECT
    address,
    COUNT(*)                    AS ERC20_Total_Ether_Recv_Tnx,
    COUNT(DISTINCT peer)        AS ERC20_Uniq_Rec_Addr,
    MIN(tok_val)                AS ERC20_Min_Val_Rec,
    MAX(tok_val)                AS ERC20_Max_Val_Rec,
    AVG(tok_val)                AS ERC20_Avg_Val_Rec,
    SUM(tok_val)                AS ERC20_Total_Ether_Received,
    COUNT(DISTINCT IF(peer_is_contract, peer, NULL))
                               AS ERC20_Uniq_Rec_Contract_Addr
  FROM b WHERE NOT is_sender GROUP BY address
),
sent_most AS (
  SELECT address,
         ARRAY_AGG(token_name ORDER BY cnt DESC LIMIT 1)[OFFSET(0)]
                               AS ERC20_Most_Sent_Token_Type
  FROM (SELECT address, token_name, COUNT(*) AS cnt
        FROM b WHERE is_sender GROUP BY address, token_name)
  GROUP BY address
),
recv_most AS (
  SELECT address,
         ARRAY_AGG(token_name ORDER BY cnt DESC LIMIT 1)[OFFSET(0)]
                               AS ERC20_Most_Rec_Token_Type
  FROM (SELECT address, token_name, COUNT(*) AS cnt
        FROM b WHERE NOT is_sender GROUP BY address, token_name)
  GROUP BY address
),
uniq_tok_sent AS (
  SELECT address,
         COUNT(DISTINCT token_name) AS ERC20_Uniq_Sent_Token_Name
  FROM b WHERE is_sender GROUP BY address
),
uniq_tok_recv AS (
  SELECT address,
         COUNT(DISTINCT token_name) AS ERC20_Uniq_Rec_Token_Name
  FROM b WHERE NOT is_sender GROUP BY address
),
timing AS (
  SELECT
    address,
    AVG(CASE WHEN is_sender AND gap IS NOT NULL THEN gap END)
                               AS ERC20_Avg_Time_Between_Sent_Tnx,
    AVG(CASE WHEN NOT is_sender AND gap IS NOT NULL THEN gap END)
                               AS ERC20_Avg_Time_Between_Rec_Tnx,
    AVG(CASE WHEN peer_is_contract AND gap IS NOT NULL THEN gap END)
                               AS ERC20_Avg_Time_Between_Contract_Tnx
  FROM (
    SELECT address,is_sender,peer_is_contract,ts,
           TIMESTAMP_DIFF(ts,
             LAG(ts) OVER (PARTITION BY address ORDER BY ts),
             MINUTE) AS gap
    FROM b
  )
  GROUP BY address
)
SELECT
  COALESCE(sent.address, recv.address)                  AS address,
  (COALESCE(ERC20_Total_Ether_Sent_Tnx,CAST(0 AS INT64)) +
   COALESCE(ERC20_Total_Ether_Recv_Tnx,CAST(0 AS INT64)))
                                                       AS Total_ERC20_Tnxs,
  COALESCE(ERC20_Total_Ether_Received,0.0)              AS ERC20_Total_Ether_Received,
  COALESCE(ERC20_Total_Ether_Sent,0.0)                  AS ERC20_Total_Ether_Sent,
  COALESCE(ERC20_Avg_Val_Sent_Contract,0.0)             AS ERC20_Total_Ether_Sent_Contract,
  COALESCE(ERC20_Uniq_Sent_Addr,CAST(0 AS INT64))       AS ERC20_Uniq_Sent_Addr,
  COALESCE(ERC20_Uniq_Rec_Addr,CAST(0 AS INT64))        AS ERC20_Uniq_Rec_Addr,
  COALESCE(ERC20_Uniq_Rec_Contract_Addr,CAST(0 AS INT64))
                                                       AS ERC20_Uniq_Rec_Contract_Addr,
  COALESCE(ERC20_Avg_Time_Between_Sent_Tnx,0.0)         AS ERC20_Avg_Time_Between_Sent_Tnx,
  COALESCE(ERC20_Avg_Time_Between_Rec_Tnx,0.0)          AS ERC20_Avg_Time_Between_Rec_Tnx,
  COALESCE(ERC20_Avg_Time_Between_Contract_Tnx,0.0)     AS ERC20_Avg_Time_Between_Contract_Tnx,
  COALESCE(ERC20_Min_Val_Rec,0.0)                       AS ERC20_Min_Val_Rec,
  COALESCE(ERC20_Max_Val_Rec,0.0)                       AS ERC20_Max_Val_Rec,
  COALESCE(ERC20_Avg_Val_Rec,0.0)                       AS ERC20_Avg_Val_Rec,
  COALESCE(ERC20_Min_Val_Sent,0.0)                      AS ERC20_Min_Val_Sent,
  COALESCE(ERC20_Max_Val_Sent,0.0)                      AS ERC20_Max_Val_Sent,
  COALESCE(ERC20_Avg_Val_Sent,0.0)                      AS ERC20_Avg_Val_Sent,
  COALESCE(ERC20_Min_Val_Sent_Contract,0.0)             AS ERC20_Min_Val_Sent_Contract,
  COALESCE(ERC20_Max_Val_Sent_Contract,0.0)             AS ERC20_Max_Val_Sent_Contract,
  COALESCE(ERC20_Avg_Val_Sent_Contract,0.0)             AS ERC20_Avg_Val_Sent_Contract,
  COALESCE(ERC20_Uniq_Sent_Token_Name,CAST(0 AS INT64)) AS ERC20_Uniq_Sent_Token_Name,
  COALESCE(ERC20_Uniq_Rec_Token_Name,CAST(0 AS INT64))  AS ERC20_Uniq_Rec_Token_Name,
  ERC20_Most_Sent_Token_Type,
  ERC20_Most_Rec_Token_Type
FROM sent
FULL JOIN recv            USING(address)
LEFT JOIN sent_most       USING(address)
LEFT JOIN recv_most       USING(address)
LEFT JOIN uniq_tok_sent   USING(address)
LEFT JOIN uniq_tok_recv   USING(address)
LEFT JOIN timing          USING(address);

/* 6. финальная таблица */
CREATE OR REPLACE TABLE `celtic-tendril-459507-q8.fraud_lstm.address_features` AS
SELECT
  ad.address,
  ad.flag_text AS FLAG,
  ad.flag_num  AS FLAG_NUM,
  eth.* EXCEPT(address),
  erc.* EXCEPT(address)
FROM `celtic-tendril-459507-q8.fraud_lstm.addresses`       ad
LEFT JOIN `celtic-tendril-459507-q8.fraud_lstm.eth_features`  eth USING(address)
LEFT JOIN `celtic-tendril-459507-q8.fraud_lstm.erc20_features` erc USING(address);

