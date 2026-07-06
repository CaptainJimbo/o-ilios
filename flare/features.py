"""The model's feature contract — importable without the training stack
(no pandas/sklearn/lightgbm), because the live worker's requirements are
deliberately slim.

SHARP keywords present BOTH in SWAN-SF (training) and in JSOC's
hmi.sharp_cea_720s_nrt (live). Order matters: it is the feature order the
model was trained with.
"""

FEATURES = [
    "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
    "MEANPOT", "R_VALUE", "MEANSHR", "SHRGT45", "MEANGAM", "MEANGBT",
    "MEANGBZ", "MEANGBH", "MEANJZH", "MEANJZD", "MEANALP",
]
