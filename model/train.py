import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('/mnt/d/sursat/kuliah/Semester 6/PKL SE/proctoring-system/data/train.csv')
#df = df.dropna()
df.drop(columns=['test', 'timestamp'], inplace=True)
X = df.drop(columns=['Label'])
y = df['Label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'model/model_rf.pkl')
print("✅ Model trained and saved.")