import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Baca Data
df = pd.read_excel("cuaca_ekstrem.xlsx", header=1)

# 2. Bersihkan data yang null
df = df.dropna(subset=["Kejadian", "Provinsi", "Meninggal", "Hilang", "Terluka", "Rumah Rusak", "Rumah Terendam", "Fasum Rusak"])

# 3. Tambahkan label manual untuk klasifikasi (misal: ringan/sedang/parah)
def tentukan_label(row):
    total = row["Meninggal"] + row["Hilang"] + row["Terluka"] + row["Rumah Rusak"] + row["Rumah Terendam"] + row["Fasum Rusak"]
    if total == 0:
        return "Ringan"
    elif total <= 20:
        return "Sedang"
    else:
        return "Parah"

df["Label"] = df.apply(tentukan_label, axis=1)

# 4. Encode kolom
le_kejadian = LabelEncoder()
le_provinsi = LabelEncoder()
le_label = LabelEncoder()

df["Kejadian_enc"] = le_kejadian.fit_transform(df["Kejadian"])
df["Provinsi_enc"] = le_provinsi.fit_transform(df["Provinsi"])
df["Label_enc"] = le_label.fit_transform(df["Label"])

# 5. Fitur dan target
X = df[["Kejadian_enc", "Provinsi_enc", "Meninggal", "Hilang", "Terluka", "Rumah Rusak", "Rumah Terendam", "Fasum Rusak"]]
y = df["Label_enc"]

# 6. Latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 7. Simpan model dan encoder
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(le_kejadian, "encoder_kejadian.pkl")
joblib.dump(le_provinsi, "encoder_provinsi.pkl")
joblib.dump(le_label, "encoder_label.pkl")

print("Model dan encoder berhasil disimpan.")
