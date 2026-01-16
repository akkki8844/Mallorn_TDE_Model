import pandas as pd
import numpy as np
from tqdm import tqdm
import os

FILTERS = ["u", "g", "r", "i", "z", "y"]

def basic_stats(time, flux):
    feats = {}
    feats["n_obs"] = len(flux)
    feats["time_span"] = time.max() - time.min()
    feats["flux_mean"] = flux.mean()
    feats["flux_std"] = flux.std()
    feats["flux_min"] = flux.min()
    feats["flux_max"] = flux.max()
    feats["flux_amp"] = flux.max() - flux.min()

    if len(flux) > 1:
        try:
            feats["flux_slope"] = np.polyfit(time, flux, 1)[0]
        except:
            feats["flux_slope"] = 0.0
    else:
        feats["flux_slope"] = 0.0

    return feats


def shape_features(time, flux):
    feats = {}

    peak_idx = np.argmax(flux)
    t_peak = time[peak_idx]
    f_peak = flux[peak_idx]

    feats["time_to_peak"] = t_peak - time.min()

    pre = flux[:peak_idx + 1]
    post = flux[peak_idx:]

    if len(pre) > 1:
        feats["rise_slope"] = (pre[-1] - pre[0]) / (time[peak_idx] - time[0] + 1e-6)
    else:
        feats["rise_slope"] = 0.0

    if len(post) > 1:
        feats["decay_slope"] = (post[-1] - post[0]) / (time[-1] - time[peak_idx] + 1e-6)
    else:
        feats["decay_slope"] = 0.0

    feats["asymmetry"] = abs(feats["decay_slope"]) / (abs(feats["rise_slope"]) + 1e-6)

    half_max = 0.5 * f_peak
    feats["duration_above_half_max"] = np.sum(flux >= half_max)

    return feats


def extract_object_features(lc):
    feats = {}

    time = lc["Time (MJD)"].values
    flux = lc["Flux"].values

    feats.update(basic_stats(time, flux))
    feats.update(shape_features(time, flux))

    global_peak_time = time[np.argmax(flux)]

    for f in FILTERS:
        sub = lc[lc["Filter"] == f]

        if len(sub) == 0:
            for k in [
                "n_obs", "time_span", "flux_mean", "flux_std",
                "flux_min", "flux_max", "flux_amp", "flux_slope"
            ]:
                feats[f"{k}_{f}"] = 0.0
            feats[f"t_peak_offset_{f}"] = 0.0
            continue

        t = sub["Time (MJD)"].values
        fl = sub["Flux"].values

        stats = basic_stats(t, fl)
        for k, v in stats.items():
            feats[f"{k}_{f}"] = v

        feats[f"t_peak_offset_{f}"] = t[np.argmax(fl)] - global_peak_time

    return feats


def build_features(log_df, mode):
    rows = []

    for split_name, split_group in tqdm(log_df.groupby("split"), desc=f"{mode} splits"):
        lc_path = f"data/raw/{split_name}/{mode}_full_lightcurves.csv"
        lc_df = pd.read_csv(lc_path)

        for _, meta in split_group.iterrows():
            obj_id = meta["object_id"]
            lc = lc_df[lc_df["object_id"] == obj_id]

            if len(lc) == 0:
                continue

            feats = extract_object_features(lc)
            feats["object_id"] = obj_id

            if mode == "train":
                feats["target"] = meta["target"]

            rows.append(feats)

    return pd.DataFrame(rows)


print("Loading logs...")
train_log = pd.read_csv("data/raw/train_log.csv")
test_log = pd.read_csv("data/raw/test_log.csv")

print("Building train features...")
train_features = build_features(train_log, "train")

print("Building test features...")
test_features = build_features(test_log, "test")

os.makedirs("data/processed", exist_ok=True)

train_features.to_csv("data/processed/train_features.csv", index=False)
test_features.to_csv("data/processed/test_features.csv", index=False)

print("DONE")
print("Train:", train_features.shape)
print("Test :", test_features.shape)
