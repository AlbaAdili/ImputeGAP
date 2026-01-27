import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- For computer vision datasets (HMNIST / SPRITES): visualize imputation over time ---
def create_imputation_plot(
    image_shape,
    time_length,
    miss,
    imputed_no_gt,
    imputed,
    gt,
    sample_id,
    figsize=(15, 5),
    cmap="gray",
):
    """
    Shows a 4-row grid across time steps:
      Row 1: Missing input
      Row 2: Raw imputation (no GT reinsertion)
      Row 3: Final imputed (observed values preserved)
      Row 4: Ground truth
    """
    # avoid mutating original array
    miss_vis = np.array(miss, copy=True)
    miss_vis = np.nan_to_num(miss_vis, nan=0.0)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=time_length,
        figsize=figsize,
        layout="constrained",
        sharey=True,
    )

    def _show(ax, frame_flat):
        frame = frame_flat.reshape(*image_shape)
        ax.imshow(np.clip(frame, 0, 1), cmap=cmap)
        ax.axis("off")

    for t in range(time_length):
        _show(axes[0, t], miss_vis[sample_id, t])
        _show(axes[1, t], imputed_no_gt[sample_id, t])
        _show(axes[2, t], imputed[sample_id, t])
        _show(axes[3, t], gt[sample_id, t])

    axes[0, 0].set_ylabel("Miss")
    fig.suptitle("Miss -> Imputed (no GT) -> Imputed | GT")
    plt.show()


# --- Mask stats: mask shape (S, V, T), where 1 means missing ---
def compute_mask_statistics(mask):
    """
    Returns:
      - (img_avg, img_std): average missing ratio per image/frame across all samples & timesteps
      - (sample_avg, sample_std): average missing ratio per sample (sequence)
      - tot_miss_ratio: global missing ratio
      - (series_avg_mean, series_std_mean): summary of per-series missing counts
    """
    mask = np.asarray(mask)
    s, v, t = mask.shape

    # per-image (flatten S*V images, missing ratio within each image is sum over features / t)
    miss_per_img = mask.sum(axis=2).reshape(-1) / t
    img_avg, img_std = float(miss_per_img.mean()), float(miss_per_img.std())

    # per-sample (missing ratio within a sample is sum over V,T / (V*T))
    miss_per_sample = mask.sum(axis=(1, 2)) / (v * t)
    sample_avg, sample_std = float(miss_per_sample.mean()), float(miss_per_sample.std())

    # global missing ratio
    tot_miss_ratio = float(mask.sum() / (s * v * t))

    # per-series summary: transpose to (S, T, V), then sum over V
    m_tv = mask.transpose(0, 2, 1)  # (S, T, V)
    miss_per_series = m_tv.sum(axis=2)  # (S, T)
    series_avg = miss_per_series.mean(axis=1)  # (S,)
    series_std = miss_per_series.std(axis=1)   # (S,)

    return (
        (img_avg, img_std),
        (sample_avg, sample_std),
        tot_miss_ratio,
        (float(series_avg.mean()), float(series_std.mean())),
    )


def compute_mask_statistics_v2(mask, values=None, target_value=None):
    """
    mask: binary array-like (1 = missing), shape (S, V, T)
    values: original data array, same shape as mask
    target_value: value to condition missingness on (e.g., 1 for white pixels)
    """
    mask = np.asarray(mask)
    s, v, t = mask.shape

    # Make sure logical operations are safe
    mask_bool = mask.astype(bool)

    conditioned = (values is not None) and (target_value is not None)
    if conditioned:
        values = np.asarray(values)
        value_mask = (values == target_value)
        eff = mask_bool & value_mask
        denom_total = int(value_mask.sum())
    else:
        eff = mask_bool
        value_mask = None
        denom_total = s * v * t

    # ---- Per-image missingness ----
    # numerator: sum over features (T)
    num_img = eff.sum(axis=2).reshape(-1)

    if conditioned:
        den_img = value_mask.sum(axis=2).reshape(-1)
    else:
        den_img = np.full((s * v,), t, dtype=np.int64)

    miss_per_img = np.divide(num_img, den_img, out=np.zeros_like(num_img, dtype=float), where=den_img > 0)
    img_avg, img_std = float(miss_per_img.mean()), float(miss_per_img.std())

    # ---- Per-sample missingness ----
    num_sample = eff.sum(axis=(1, 2))
    if conditioned:
        den_sample = value_mask.sum(axis=(1, 2))
    else:
        den_sample = np.full((s,), v * t, dtype=np.int64)

    miss_per_sample = np.divide(
        num_sample,
        den_sample,
        out=np.zeros_like(num_sample, dtype=float),
        where=den_sample > 0,
    )
    sample_avg, sample_std = float(miss_per_sample.mean()), float(miss_per_sample.std())

    # ---- Total missingness ----
    tot_miss_ratio = float(eff.sum() / denom_total) if denom_total > 0 else 0.0

    return (img_avg, img_std), (sample_avg, sample_std), tot_miss_ratio


def plot_training_results(path):
    """
    Reads a training_curve.tsv file and plots:
      - Train/Val total loss: NLL + beta*KL
      - Train/Val NLL
      - Train/Val KL
    """
    curve_path = os.path.join(path, "training_curve.tsv")
    df = pd.read_csv(curve_path, sep="\t")

    # only points where validation exists
    df_val = df[df["val_loss"].notna()]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")

    # Total loss
    axes[0].plot(df.index, df["train_loss"], label="Train")
    axes[0].plot(df_val.index, df_val["val_loss"], label="Validation")
    axes[0].set_title("Train and validation loss")
    axes[0].set_ylabel("Loss: NLL + beta*KL")
    axes[0].set_xlabel("Training steps")
    axes[0].legend(loc="upper right")

    # NLL
    axes[1].plot(df.index, df["train_nll"], label="Train")
    axes[1].plot(df_val.index, df_val["val_nll"], label="Validation")
    axes[1].set_title("Train and validation NLL")
    axes[1].set_ylabel("NLL")
    axes[1].set_xlabel("Training steps")
    axes[1].legend(loc="upper right")

    # KL
    axes[2].plot(df.index, df["train_kl"], label="Train")
    axes[2].plot(df_val.index, df_val["val_kl"], label="Validation")
    axes[2].set_title("Train and validation KL")
    axes[2].set_ylabel("KL")
    axes[2].set_xlabel("Training steps")
    axes[2].legend(loc="upper right")

    plt.show()
