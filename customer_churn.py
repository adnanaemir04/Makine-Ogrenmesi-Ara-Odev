"""
Makine Öğrenmesi Ara Ödev - Müşteri Ayrılma Tahmini

Amaç:
Müşterilerin abonelikten ayrılıp ayrılmayacağını (churn) tahmin etmek
için temel bir sınıflandırma akışı uygulamaktır.

Kullanılan kütüphaneler:
- pandas
- numpy
- scikit-learn
- matplotlib

Çalıştırma:
1. Gerekli paketleri kurun:
   pip install -r requirements.txt
2. customer_data.csv dosyasının bu Python dosyasıyla aynı klasörde
   olduğundan emin olun.
3. Çalıştırın:
   python customer_churn.py

Akış:
Veri okuma -> veri inceleme -> eksik değer doldurma ->
feature engineering -> train/validation/test ayrımı ->
One-Hot Encoding + StandardScaler -> Logistic Regression ve KNN ->
validation karşılaştırması -> seçilen modelin test değerlendirmesi.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

df = pd.read_csv("customer_data.csv")

print("=" * 60)
print("VERİ SETİNİN İLK 5 SATIRI")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("VERİ SETİ BOYUTU")
print("=" * 60)
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")

print("\n" + "=" * 60)
print("VERİ TİPLERİ")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("CHURN DAĞILIMI")
print("=" * 60)
print(df["churn"].value_counts())
print("\nOransal dağılım:")
print(df["churn"].value_counts(normalize=True))

# ---------------------------------------------------------
# 2. EKSİK DEĞER KONTROLÜ
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("EKSİK DEĞERLER - DOLDURMA ÖNCESİ")
print("=" * 60)
print(df.isnull().sum())

numeric_columns = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
]

categorical_columns = [
    "sehir",
    "uyelik_tipi",
]

# Sayısal değişkenlerde medyan, kategorik değişkenlerde mod kullanıyoruz.
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\n" + "=" * 60)
print("EKSİK DEĞERLER - DOLDURMA SONRASI")
print("=" * 60)
print(df.isnull().sum())

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
# Destek talebi olan müşterileri 1, olmayanları 0 yapıyoruz.
df["destek_talebi_var_mi"] = (
    df["destek_talebi_sayisi"] > 0
).astype(int)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING SONRASI")
print("=" * 60)
print(df[[
    "destek_talebi_sayisi",
    "destek_talebi_var_mi"
]].head(10))

# ---------------------------------------------------------
# 4. X ve y
# ---------------------------------------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

# ---------------------------------------------------------
# 5. TRAIN / VALIDATION / TEST
# ---------------------------------------------------------
# Önce %80 train+validation, %20 test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train+validation'ın %75'i train (%60 toplam),
# %25'i validation (%20 toplam)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

print("\n" + "=" * 60)
print("VERİ BÖLÜMLERİ")
print("=" * 60)
print(f"Train      : {X_train.shape[0]} satır")
print(f"Validation : {X_val.shape[0]} satır")
print(f"Test       : {X_test.shape[0]} satır")

# ---------------------------------------------------------
# 6. PREPROCESSING
# ---------------------------------------------------------
numeric_features = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "destek_talebi_var_mi",
]

categorical_features = [
    "sehir",
    "uyelik_tipi",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)

# ---------------------------------------------------------
# 7. MODEL PIPELINE'LARI
# ---------------------------------------------------------
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            KNeighborsClassifier(n_neighbors=5),
        ),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
}

# ---------------------------------------------------------
# 8. VALIDATION KARŞILAŞTIRMASI
# ---------------------------------------------------------
validation_results = []

print("\n" + "=" * 60)
print("VALIDATION SONUÇLARI")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(
        y_val, y_val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_val_pred, zero_division=0
    )

    validation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    })

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

validation_df = pd.DataFrame(validation_results)

# Modeli F1-score'a göre seçiyoruz.
best_model_name = validation_df.loc[
    validation_df["F1-Score"].idxmax(),
    "Model",
]

print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMASI")
print("=" * 60)
print(validation_df.to_string(index=False))

print(f"\nValidation sonucuna göre seçilen model: {best_model_name}")

# ---------------------------------------------------------
# 9. SEÇİLEN MODELİ TRAIN + VALIDATION ÜZERİNDE YENİDEN EĞİTME
# ---------------------------------------------------------
best_model = models[best_model_name]

best_model.fit(X_train_val, y_train_val)

# ---------------------------------------------------------
# 10. TEST DEĞERLENDİRMESİ
# ---------------------------------------------------------
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(
    y_test, y_test_pred, zero_division=0
)
test_recall = recall_score(
    y_test, y_test_pred, zero_division=0
)
test_f1 = f1_score(
    y_test, y_test_pred, zero_division=0
)

print("\n" + "=" * 60)
print("TEST SONUÇLARI")
print("=" * 60)
print(f"Seçilen model: {best_model_name}")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-Score : {test_f1:.4f}")

# ---------------------------------------------------------
# 11. CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Kalır (0)", "Ayrılır (1)"],
)

disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 12. KISA YORUM
# ---------------------------------------------------------
other_model = (
    "KNN"
    if best_model_name == "Logistic Regression"
    else "Logistic Regression"
)

print("\n" + "=" * 60)
print("KISA YORUM")
print("=" * 60)
print(
    f"Validation sonuçlarına göre {best_model_name}, "
    f"{other_model} modelinden daha yüksek F1-score elde ettiği "
    "için seçilmiştir."
)
print(
    f"Seçilen model test setinde {test_accuracy:.4f} accuracy, "
    f"{test_precision:.4f} precision, {test_recall:.4f} recall "
    f"ve {test_f1:.4f} F1-score elde etmiştir."
)
print(
    "KNN mesafe tabanlı olduğu için sayısal değişkenlerin "
    "StandardScaler ile ölçeklenmesi önemlidir. Kategorik "
    "değişkenler ise One-Hot Encoding ile sayısal forma "
    "dönüştürülmüştür."
)
