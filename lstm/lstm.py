# %%
# !pip uninstall -y flax orbax-checkpoint jax jaxlib \
#     ml-dtypes tf-keras tensorflow tensorflow-cpu tensorflow-text \
#     tensorflow-decision-forests keras keras-hub \
#     chex optax fastai spacy tensorstore numba \
#     umap-learn pynndescent librosa shap cuml-cu12 cudf-cu12 dask-cuda

%pip install -U threadpoolctl joblib
#%pip install --force-reinstall numpy==1.26.4
%pip install --upgrade scikit-learn
%pip install -U imbalanced-learn umap-learn

%pip install --upgrade --force-reinstall \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    tensorflow

# %%
import importlib, random, os, numpy as np, tensorflow as tf

print("===== VERSIONS =====")
for lib in ("numpy", "pandas", "sklearn", "imblearn",
            "tensorflow", "matplotlib", "seaborn"):
    m = importlib.import_module(lib if lib != "sklearn" else "sklearn")
    print(f"{lib:17s}: {m.__version__}")

print("\n===== TensorFlow devices =====")
print(tf.config.list_physical_devices())   # ← здесь будет только CPU, и это нормально

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"\nRandom seed set to {SEED}")

# %%
# === @title Этап 2 (Kaggle API) — Ethereum Fraud Detection ===
# 0. (если делали раньше, шаги с pip и токеном можно пропустить)

# — 0.1 Установка библиотеки kaggle —
%pip install --quiet kaggle

# # — 0.2 Загрузка kaggle.json (один раз за сессию Colab) —
# #    Если токен уже лежит в ~/.kaggle, эту строку можно пропустить.
# from google.colab import files, auth
# import pathlib, io, os, json, pandas as pd

# if not pathlib.Path("~/.kaggle/kaggle.json").expanduser().exists():
#     print("📂 Загрузите kaggle.json ➜")
#     token_file = files.upload()           # откроется диалог
#     if "kaggle.json" not in token_file:
#         raise ValueError("Нужно загрузить файл под именем kaggle.json")
#     pathlib.Path("~/.kaggle").expanduser().mkdir(exist_ok=True)
#     with open(pathlib.Path("~/.kaggle/kaggle.json").expanduser(), "wb") as f:
#         f.write(token_file["kaggle.json"])
#     !chmod 600 ~/.kaggle/kaggle.json
# else:
#     print("🔑 kaggle.json уже настроен")

import os
import json
import pathlib
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. Убедимся, что у нас есть файл kaggle.json (с токеном Kaggle)
kaggle_json_path = '../kaggle.json'

if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError("Файл 'kaggle.json' не найден. Пожалуйста, скачайте его с вашего аккаунта Kaggle.")

# 2. Конфигурируем API Kaggle с использованием токена
with open(kaggle_json_path) as f:
    kaggle_json = json.load(f)

os.environ['KAGGLE_USERNAME'] = kaggle_json['username']
os.environ['KAGGLE_KEY'] = kaggle_json['key']

# 3. Инициализируем API
api = KaggleApi()
api.authenticate()

# 1. Скачиваем архив датасета
slug = "vagifa/ethereum-frauddetection-dataset"
print(f"\n⬇️ Скачиваю датасет {slug} ...")
!kaggle datasets download -d {slug} -p ./data --force --quiet

# 2. Распаковываем
print("📦 Распаковываю ...")
!unzip -o ./data/*.zip -d ./data > /dev/null

# 3. Открываем transactions.csv
csv_path = "./data/transactions.csv"
if not os.path.exists(csv_path):
    # fallback: берем первый найденный CSV
    import glob
    found = glob.glob("./data/**/*.csv", recursive=True)
    if not found:
        raise RuntimeError("CSV-файлы не найдены в архиве.")
    csv_path = found[0]
print(f"✅ Нашёл CSV: {csv_path}")

df = pd.read_csv(csv_path)

# 4. Отладочная информация
print("\n===== SHAPE =====")
print(df.shape)            # ожидаем (9840, 49)

print("\n===== HEAD (5 строк) =====")
display(df.head())

print("\n===== Классовое соотношение (isFraud) =====")
if "FLAG" in df.columns:
    display((df["FLAG"].value_counts(normalize=True) * 100).round(2).rename("%"))
else:
    print("Столбец 'FLAG' не найден — проверьте структуру датасета.")

# %%
import pandas as pd
df = pd.read_csv('../dataset/data2/lstm_dataset_address.csv')

# 4. Отладочная информация
print("\n===== SHAPE =====")
print(df.shape)

print("\n===== HEAD (5 строк) =====")
display(df.head())

print("\n===== Классовое соотношение (isFraud) =====")
if "FLAG" in df.columns:
    display((df["FLAG"].value_counts(normalize=True) * 100).round(2).rename("%"))
else:
    print("Столбец 'FLAG' не найден — проверьте структуру датасета.")

# %%
import pandas as pd

# Загрузка датасета
df = pd.read_csv('../dataset/data2/lstm_dataset_address.csv')

df['FLAG'] = df['FLAG'].map({'scam': 1, 'legit': 0})
df = df[df['is_contract']==False]

print(df.shape)


# %%
df.drop(columns=["Unnamed: 0", "Index", "FLAG_NUM", "is_contract", "scam_type"], inplace=True, errors="ignore")
df['FLAG'].value_counts(normalize=True)

# %%
# === @title Этап 3: первичный анализ и инвентаризация признаков ===
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

# 0) Копируем исходный df (на всякий случай)
df_eda = df.copy()

# 1) Удаляем служебные колонки, если есть
svc_cols = ["Unnamed: 0", "Index"]
df_eda.drop(columns=[c for c in svc_cols if c in df_eda.columns],
            inplace=True, errors="ignore")

print("==> Shape после удаления служебных колонок:", df_eda.shape)

# 2) Проверяем пропуски
na_counts = df_eda.isna().sum()
na_nonzero = na_counts[na_counts > 0].sort_values(ascending=False)

print("\n===== ТОП-10 столбцов по числу NaN =====")
display(na_nonzero.head(10))

# 3) Корреляционная матрица Пирсона
num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
corr = df_eda[num_cols].corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0,
            vmax=1, vmin=-1, square=True,
            cbar_kws={"shrink": .8}, xticklabels=False, yticklabels=False)
plt.title("Correlation heatmap (numeric features)")
plt.show()

# 4) Признаки с нулевой дисперсией (постоянные значения)
zero_var = df_eda[num_cols].loc[:, df_eda[num_cols].nunique() == 1].columns.tolist()
print(f"\n===== Признаки с нулевой дисперсией (n={len(zero_var)}) =====")
print(zero_var)

# 5) Быстрый взгляд на распределение классов (FLAG)
print("\n===== Распределение FLAG =====")
display(df_eda["FLAG"].value_counts(normalize=True).rename("%").mul(100).round(2))

# %%
# import pandas as pd

# # Группы по классу
# df_flag_1 = df[df['FLAG'] == 1]
# df_flag_0 = df[df['FLAG'] == 0]

# # Определим минимальное количество между классами
# min_class_count = min(len(df_flag_1), len(df_flag_0))

# # Случайная выборка из каждого класса
# df_flag_1_sampled = df_flag_1.sample(n=min_class_count, random_state=42)
# df_flag_0_sampled = df_flag_0.sample(n=int(min_class_count), random_state=42)

# # Объединяем и перемешиваем
# df_balanced = pd.concat([df_flag_1_sampled, df_flag_0_sampled], ignore_index=True)
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # Проверим баланс классов
# print(df_balanced['FLAG'].value_counts())

# # Обновляем переменную df
# df = df_balanced


# %%
# === @title Этап 4: очистка признаков перед обучением ===

import pandas as pd, numpy as np

# 0) Копируем исходный df
df_clean = df.copy()

# 1) Убираем пробелы в заголовках и служебные поля
df_clean.columns = df_clean.columns.str.strip()
df_clean.drop(columns=["Unnamed: 0", "Index"], inplace=True, errors="ignore")

# 2) Удаляем все строковые столбцы (Address, токены и т.п.)
str_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
print("Удаляю строковые колонки:", str_cols)
df_clean = df_clean.drop(columns=str_cols)

# 3) Удаляем признаки с нулевой дисперсией
to_drop = [c for c in zero_var if c in df_clean.columns]
print("Удаляю zero-variance:", to_drop)
df_clean = df_clean.drop(columns=to_drop)

# 4) Заполняем все NaN медианой по столбцу
num_cols = df_clean.columns.tolist()  # теперь все — числовые
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 5) Убираем одну из пары сильнокоррелирующих признаков (|ρ| > 0.9)
corr = df_clean.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
print("Удаляю сильно коррелирующие (>0.9):", high_corr_drop)
df_clean = df_clean.drop(columns=high_corr_drop)

# 6) Финальная проверка
print("\nИтоговая форма df_clean:", df_clean.shape)
print("Осталось NaN всего:", df_clean.isna().sum().sum())

# %%
# === @title Этап 5: разделение и SMOTE ===
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler


# 1) Готовим X и y
SEED = 42
X = df_clean.drop(columns=["FLAG"])
y = df_clean["FLAG"]

# 1) Разбиваем на train+val и test (80/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2) Делим оставшиеся на train и val (75/25 от X_temp → в итоге 60/20/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,    # 0.25 * 0.8 = 0.2
    stratify=y_temp,
    random_state=42
)

# 3) SMOTE только на train
sm = SMOTE(random_state=42)
X_train_res, y_train_res = X_train, y_train

# 4) Масштабирование только по train_res
scaler = MinMaxScaler()
scaler.fit(X_train_res)

X_train_s = scaler.transform(X_train_res)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# 5) reshape для LSTM
n_features    = X_train_s.shape[1]
X_train_lstm  = X_train_s.reshape(-1, 1, n_features)
X_val_lstm    = X_val_s.reshape(-1,   1, n_features)
X_test_lstm   = X_test_s.reshape(-1,  1, n_features)

# 6) Правильно формируем y-объекты для валидации и теста
y_val_lstm  = y_val.to_numpy()   # или y_val.values
y_test_lstm = y_test.to_numpy()  # или y_test.values


# %%
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.decomposition import PCA
# import umap
# import matplotlib.pyplot as plt

# # 1. Получаем X_train_res, y_train_res
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

# # 2. Помечаем, какие точки синтетические
# is_synth = ['real'] * len(X_train) + ['synthetic'] * (len(X_res) - len(X_train))
# df_vis = pd.DataFrame(X_res)
# df_vis['label'] = y_res
# df_vis['type']  = is_synth
# import numpy as np

# X_raw = df_vis.drop(columns=['label','type'])
# cols_inf = [c for c in X_raw.columns if np.isinf(X_raw[c]).any()]
# print("Есть inf в столбцах:", cols_inf)
# X_clean = X_raw.replace([np.inf, -np.inf], np.nan)
# X_clean = X_clean.fillna(X_clean.median())

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_clean)


# # 3. UMAP в 2D
# reducer = umap.UMAP(n_components=2, random_state=42)
# emb = reducer.fit_transform(X_scaled)

# # 4. Рисуем
# plt.figure(figsize=(8,6))
# for t, m in [('real','o'), ('synthetic','x')]:
#     idx = df_vis['type']==t
#     plt.scatter(emb[idx,0], emb[idx,1],
#                 c=df_vis.loc[idx,'label'], marker=m,
#                 alpha=0.6, label=t)
# plt.legend()
# plt.title("UMAP: реальные vs синтетические точки")
# plt.show()


# %%
# from sklearn.neighbors import NearestNeighbors
# import numpy as np

# # X_res — массив всех точек после SMOTE, y_res — их метки
# # mask_synth — булев массив, где True для синтетических
# X_array = X_res  # shape (n_res, n_features)
# y_array = y_res  # shape (n_res,)
# mask_synth = np.array(is_synth) == 'synthetic'

# # Найдём k ближайших соседей по Евклиду
# k = 10
# nn = NearestNeighbors(n_neighbors=k+1).fit(X_array)
# distances, indices = nn.kneighbors(X_array)

# bad = []  # здесь будем собирать %
# for i in np.where(mask_synth)[0]:
#     neigh_idx = indices[i][1:]  # без себя самого
#     # сколько среди соседей — другого класса?
#     frac_other = np.mean(y_array[neigh_idx] != y_array[i])
#     bad.append(frac_other)

# # Посмотрим распределение
# import matplotlib.pyplot as plt
# plt.hist(bad, bins=20)
# plt.xlabel("Доля соседей другого класса")
# plt.ylabel("Число синтетических точек")
# plt.title("Насколько «грязная» синтетика?")
# plt.show()

# # Сколько точек с frac_other > 0.5
# print("Сомнительных точек (>50% чужих соседей):",
#       np.sum(np.array(bad)>0.5), "из", len(bad))


# %%
# === @title Этап 8: построение LSTM-модели ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1) Создаём модель
model = Sequential([
    LSTM(64, input_shape=(1, X_train_lstm.shape[2])),  # timesteps=1, features=n_features
    Dense(1, activation='sigmoid')
])

# 2) Компиляция
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3) Вывод структуры модели
model.summary()

# %%
# === @title Этап 9: обучение LSTM-модели ===
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1) Определяем колбэки
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
rlrp = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
cp_callback = ModelCheckpoint(
    filepath='checkpoints/lstm_best_weights.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

epochs=200
batch_size=32
# рассчитываем вес редкого класса
w0 = 1.0
w1 = len(y_train[y_train==0]) / len(y_train[y_train==1])

history = model.fit(
    X_train_lstm, y_train_res,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_lstm, y_val),
    class_weight={0: w0, 1: w1},
    callbacks=[es, rlrp, cp_callback],
    verbose=2
)


# --- Отладочные графики ---

# 1) Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'],    label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2) Accuracy
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'],    label='train_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 3) Финальные метрики
print(f"Final train_acc: {history.history['accuracy'][-1]:.4f}")
print(f"Final val_acc:   {history.history['val_accuracy'][-1]:.4f}")

# %%
y_pred_proba = model.predict(X_test_lstm, verbose=0)
y_pred       = (y_pred_proba >= 0.5).astype(int).flatten()

# Проверим формы
print("X_test_lstm.shape =", X_test_lstm.shape)
print("y_test_lstm.shape =", y_test_lstm.shape)
print("y_pred.shape       =", y_pred.shape)

# Посчитаем метрики
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

acc  = accuracy_score(y_test_lstm, y_pred)
prec = precision_score(y_test_lstm, y_pred)
rec  = recall_score(y_test_lstm, y_pred)
f1   = f1_score(y_test_lstm, y_pred)
cm   = confusion_matrix(y_test_lstm, y_pred)

print("=== Test Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}\n")
print("=== Confusion Matrix ===")
print(cm)

# %%
# Шаг 1: Настройка callback для сохранения лучших весов
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os

# Создаём папку для чекпойнтов, если её ещё нет
os.makedirs("checkpoints", exist_ok=True)

# Путь для сохранения лучших весов
checkpoint_path = "checkpoints/lstm_best_weights.h5"

# Callback для сохранения только лучших весов на основе 'val_loss'
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',       # Следим за валидационной метрикой
    save_best_only=True,      # Сохраняем только лучшие веса
    mode='min',               # Минимизируем метрику (например, для val_loss)
    verbose=1,                # Печать, когда сохраняется
    save_freq='epoch'        # Сохраняем после каждой эпохи
)

# Шаг 2: Обучение модели с использованием этого колбэка
# рассчитываем вес редкого класса
w0 = 1.0
w1 = len(y_train[y_train==0]) / len(y_train[y_train==1])

history = model.fit(
    X_train_lstm, y_train_res,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_lstm, y_val),
    class_weight={0: w0, 1: w1},
    callbacks=[cp_callback],
    verbose=2
)

# Шаг 3: Сохранение модели целиком и веса
import pickle
from pathlib import Path

# 1) Создаём папку для артефактов
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

# 2) Сохраняем модель целиком (архитектура + веса)
model_path = artifacts_dir / "lstm_eth_fraud_model.h5"
model.save(str(model_path))
print(f"Модель сохранена в {model_path}")

# 3) Сохраняем scaler
scaler_path = artifacts_dir / "scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler сохранён в {scaler_path}")

# 4) Сохраняем список признаков (чтобы при загрузке знать порядок столбцов)
feature_list = X.columns.tolist()  # X – до разбиения, без FLAG
features_path = artifacts_dir / "features_list.pkl"
with open(features_path, "wb") as f:
    pickle.dump(feature_list, f)
print(f"Список признаков сохранён в {features_path}")

# Шаг 4: Извлечение и сохранение эмбеддингов
_ = model.predict(X_train_lstm[:1])

# Создаём модель для получения эмбеддингов (например, из первого LSTM слоя)
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)

# Извлекаем эмбеддинги для тренировочных и тестовых данных
embeddings_train = embedding_model.predict(X_train_lstm)
embeddings_test = embedding_model.predict(X_test_lstm)

# Сохраняем эмбеддинги
np.save("artifacts/embeddings_train.npy", embeddings_train)
np.save("artifacts/embeddings_test.npy", embeddings_test)
print(f"Эмбеддинги сохранены в 'artifacts/embeddings_train.npy' и 'artifacts/embeddings_test.npy'")

# Шаг 5: Проверка воспроизводимости (если необходимо)
# Загружаем модель и другие артефакты
model = tf.keras.models.load_model(str(artifacts_dir / "lstm_eth_fraud_model.h5"))
with open(artifacts_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(artifacts_dir / "features_list.pkl", "rb") as f:
    features_list = pickle.load(f)

# Загружаем эмбеддинги (если нужно)
embeddings_train = np.load("artifacts/embeddings_train.npy")
embeddings_test = np.load("artifacts/embeddings_test.npy")

print("Эмбеддинги и модель успешно загружены для дальнейшего использования.")



