import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

models = joblib.load('models.pkl')
X_surface = np.load('INCOIS_X_surface.npy')

DEPTH_LABELS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

LAT_ORIGIN = 5.0
LON_ORIGIN = 45.0
RESOLUTION = 0.25

N_SAMPLES, _, N_ROWS, N_COLS = X_surface.shape


def latlon_to_rowcol(lat, lon):
    row = round((lat - LAT_ORIGIN) / RESOLUTION)
    col = round((lon - LON_ORIGIN) / RESOLUTION)
    row = int(np.clip(row, 0, N_ROWS - 1))
    col = int(np.clip(col, 0, N_COLS - 1))
    return row, col


def get_features(row, col, date=None):
    """
    We only have 10 time samples and no confirmed dates yet.
    Average across all 10 samples at this pixel for now.
    Once real dates are confirmed, pick the closest matching sample instead.
    """
    sst_vals = X_surface[:, 0, row, col]
    ocean_mask = sst_vals != 0
    if ocean_mask.sum() == 0:
        return None

    feats = {}
    for i, name in enumerate(['sst', 'sss', 'ssh', 'u_curr', 'v_curr']):
        vals = X_surface[:, i, row, col][ocean_mask]
        feats[name] = float(vals.mean())
    return feats


def predict(lat, lon, date=None):
    row, col = latlon_to_rowcol(lat, lon)
    feats = get_features(row, col, date)

    if feats is None:
        return {"error": "Selected location is on land or has no data."}

    feature_row = [[row, col, feats['sst'], feats['sss'], feats['ssh'], feats['u_curr'], feats['v_curr']]]

    results = []
    for d, depth_m in enumerate(DEPTH_LABELS_M):
        model = models[d]
        temp = float(model.predict(feature_row)[0])
        results.append({"depth_m": depth_m, "temp_c": round(temp, 2)})

    return results


if __name__ == "__main__":
    result = predict(lat=15.0, lon=85.0, date="2024-06-15")  # Bay of Bengal, mid-domain
    for r in result:
        print(r)