import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# Read Dataset

df = pd.read_excel(
r"data/concrete_ai_dataset.csv.xlsx"
)

# ==========================
# MODEL 1 : UPV ONLY
# ==========================

X1 = df[['UPV']]
y = df['STRENGHT']

model1 = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model1.fit(X1,y)

joblib.dump(
    model1,
    "models/upv_model.pkl"
)

# ==========================
# MODEL 2 : SONREB
# ==========================

X2 = df[['UPV','REBOUND']]

model2 = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model2.fit(X2,y)

joblib.dump(
    model2,
    "models/sonreb_model.pkl"
)

# ==========================
# MODEL 3 : ADVANCED AI
# ==========================

X3 = df[['UPV','REBOUND','RCA']]

model3 = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model3.fit(X3,y)

joblib.dump(
    model3,
    "models/advanced_model.pkl"
)

print("All Models Trained Successfully")