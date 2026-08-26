import numpy as np


DEPTHS = [
    0, 5, 10, 20, 30,
    50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000
]


def predict(day_index: int):
    """
    Temporary prediction function.

    This will later be replaced with the real
    trained OceanEmbed model.
    """

    # Temporary dummy output:
    # 15 depth levels × 101 latitude points × 241 longitude points
    prediction = np.zeros((15, 101, 241), dtype=np.float32)

    return prediction


def get_depth_profile(day_index: int, latitude: float, longitude: float):
    """
    Get the 15-depth temperature profile at the nearest
    0.25-degree grid point.
    """

    prediction = predict(day_index)

    lat_index = round((latitude - 5.0) / 0.25)
    lon_index = round((longitude - 45.0) / 0.25)

    if not (0 <= lat_index < 101):
        raise ValueError("Latitude is outside the North Indian Ocean grid.")

    if not (0 <= lon_index < 241):
        raise ValueError("Longitude is outside the North Indian Ocean grid.")

    temperatures = prediction[:, lat_index, lon_index]

    return {
        "latitude": 5.0 + lat_index * 0.25,
        "longitude": 45.0 + lon_index * 0.25,
        "depths": DEPTHS,
        "temperatures": temperatures.tolist()
    }