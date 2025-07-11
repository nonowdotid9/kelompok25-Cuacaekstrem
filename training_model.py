import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Baca data Excel
df = pd.read_excel("cuaca_ekstrem.xlsx", header=1)
df.columns = df.columns.str.strip()

# 2. Bersihkan dan isi NaN
for col in ["Meninggal", "Hilang", "Terluka", "Rumah Rusak", "Rumah Terendam", "Fasum Rusak"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 3. Normalisasi nama
df["Provinsi"] = (
    df["Provinsi"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
)
df["Kejadian"] = (
    df["Kejadian"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
)

# 4. Hitung label otomatis
df["Korban"] = df["Meninggal"] + df["Hilang"] + df["Terluka"]
df["Kerusakan"] = df["Rumah Rusak"] + df["Rumah Terendam"] + df["Fasum Rusak"]
df["Skor"] = df["Korban"] + df["Kerusakan"]

def tentukan_label(skor):
    if skor == 0:
        return "Ringan"
    elif skor <= 10:
        return "Sedang"
    else:
        return "Parah"

df["Tingkat_Keparahan"] = df["Skor"].apply(tentukan_label)

# 5. Encode fitur
le_kejadian = LabelEncoder()
le_provinsi = LabelEncoder()
le_label = LabelEncoder()

df["Kejadian_enc"] = le_kejadian.fit_transform(df["Kejadian"])
df["Provinsi_enc"] = le_provinsi.fit_transform(df["Provinsi"])
df["Label"] = le_label.fit_transform(df["Tingkat_Keparahan"])

# 6. Training
X = df[["Kejadian_enc", "Provinsi_enc", "Meninggal", "Hilang", "Terluka", 
        "Rumah Rusak", "Rumah Terendam", "Fasum Rusak"]]
y = df["Label"]

model = RandomForestClassifier()
model.fit(X, y)

# 7. Simpan model & encoder
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(le_kejadian, "encoder_kejadian.pkl")
joblib.dump(le_provinsi, "encoder_provinsi.pkl")
joblib.dump(le_label, "encoder_label.pkl")

print("✅ Model dan encoder berhasil disimpan.")
