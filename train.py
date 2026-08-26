
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

DEPTH_LABELS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

X = np.load('INCOIS_X_surface.npy')
Y = np.load('INCOIS_Y_subsurface.npy')

n_samples, n_depths, n_rows, n_cols = Y.shape
sst = X[:, 0, :, :]
sss = X[:, 1, :, :]
ssh = X[:, 2, :, :]
u_curr = X[:, 3, :, :]
v_curr = X[:, 4, :, :]

records = []
for s in range(n_samples):
    sst_s = sst[s]
    land_mask = sst_s != 0
    valid_rows, valid_cols = np.where(land_mask)

    y_vals = Y[s, :, valid_rows, valid_cols]

    df_s = pd.DataFrame({
        'row': valid_rows,
        'col': valid_cols,
        'sst': sst_s[valid_rows, valid_cols],
        'sss': sss[s][valid_rows, valid_cols],
        'ssh': ssh[s][valid_rows, valid_cols],
        'u_curr': u_curr[s][valid_rows, valid_cols],
        'v_curr': v_curr[s][valid_rows, valid_cols],
    })
    for d in range(n_depths):
        df_s[f'depth_{d}'] = y_vals[:, d]
    records.append(df_s)

df = pd.concat(records, ignore_index=True)
print("Total surface-ocean rows:", len(df))

feature_cols = ['row', 'col', 'sst', 'sss', 'ssh', 'u_curr', 'v_curr']

models = {}
print("\n--- Model performance (per depth, 5 real input variables) ---")
for d in range(n_depths):
    col = f'depth_{d}'
    sub = df[df[col] != 0]
    X_train, X_test, y_train, y_test = train_test_split(
        sub[feature_cols], sub[col], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=30, max_depth=9, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"  {DEPTH_LABELS_M[d]}m: R2={r2:.3f}  RMSE={rmse:.3f}C  (n={len(sub)})")
    models[d] = model

joblib.dump(models, 'models.pkl', compress=3)
print("\nSaved models.pkl (dict of 15 RandomForestRegressor models, keyed by depth index)")
print("Feature order used by every model:", feature_cols)