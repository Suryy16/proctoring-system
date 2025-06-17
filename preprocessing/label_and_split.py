import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Simpan waktu pertama sebagai acuan awal
first_time = None
# Baca hasil fitur dari video
df = pd.read_csv("data/features_log.csv")

# Tambahkan kolom Label berdasarkan timestamp
def label_from_time(t):
    global first_time
    t = pd.to_datetime(t)
    if first_time is None:
        first_time = t
    elapsed = (t - first_time).total_seconds()  # detik dari awal
    if 0 <= elapsed < 30:
        return 0  # normal
    elif 30 <= elapsed < 45:
        return 1  # cheat left
    elif 45 <= elapsed < 60:
        return 1  # cheat right
    elif 60 <= elapsed < 75:
        return 1  # cheat down
    elif 75 <= elapsed < 90:
        return 0  # blink
    elif 90 <= elapsed < 105:
        return 1  # fake cheat
    elif 105 <= elapsed < 120:
        return 0  # back to normal
    else:
        return -1

df['Label'] = df['timestamp'].apply(label_from_time)

# Hapus baris tak berlabel
df = df[df['Label'] != -1]

# Simpan dataset lengkap
df.to_csv("data/features_labeled.csv", index=False)

# Split ke train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Labeling & splitting done.")
