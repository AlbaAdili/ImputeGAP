import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from imputegap.wrapper.AlgoPython.GPVAE.runnerGPVAE import gpvae_recovery


def _candidate_config_paths() -> list[Path]:
    base = Path(__file__).resolve().parents[1] / "wrapper" / "AlgoPython" / "GPVAE"
    return [
        base / "config_gpvae_hmnist.yaml",
        base / "config_gpvae_physionet.yaml",
        base / "config_gpvae_sprites.yaml",
    ]


def _auto_select_config(incomp_data) -> str:
    """
    Auto-select a config by matching:
      - nbr_features == incomp_data.shape[0]
      - incomp_data.shape[1] divisible by seq_length
    """
    if incomp_data is None or getattr(incomp_data, "shape", None) is None:
        raise ValueError("incomp_data must be a numpy-like matrix with .shape")

    n_features = int(incomp_data.shape[0])
    width = int(incomp_data.shape[1])

    for cfg_path in _candidate_config_paths():
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg_features = int(cfg.get("nbr_features", -1))
        cfg_seq_len = int(cfg.get("seq_length", -1))

        if cfg_features == n_features and cfg_seq_len > 0 and (width % cfg_seq_len == 0):
            return str(cfg_path)

    raise ValueError(
        "Could not auto-select a GP-VAE config. "
        "Please pass config_yaml_path explicitly or pass folder_checkpoint_path containing a .yaml."
    )


def gp_vae(
    incomp_data,
    config_yaml_path: Optional[str] = None,
    folder_checkpoint_path: Optional[str] = None,
    epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    inference_batch_size: Optional[int] = None,
    beta: Optional[float] = None,
    learning_rate: Optional[float] = None,
    sigma: Optional[float] = None,
    length_scale: Optional[float] = None,
    kernel_scales: Optional[int] = None,
    ground_truth=None,
    y_val=None,
    return_no_gt_imputation: bool = False,
    verbose: bool = True,
    logs: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform imputation using the GP-VAE algorithm.

    Returns
    -------
    (recov_data, extras)
        recov_data: numpy.ndarray imputed matrix
        extras: dict with optional keys:
            - 'recov_no_gt'
            - 'evaluation_metrics'
    """

    # If config not provided:
    # - if checkpoint is provided, runner may find a YAML inside the folder
    # - else auto-pick based on matrix shape
    if config_yaml_path is None:
        if folder_checkpoint_path is not None:
            config_yaml_path = ""  # runner will search for yaml inside the checkpoint folder
        else:
            config_yaml_path = _auto_select_config(incomp_data)
            if verbose:
                print(f"(GPVAE) Auto-selected config: {config_yaml_path}")

    start_time = time.time()

    recov_data, extras = gpvae_recovery(
        incomp_data,
        config_yaml_path,
        folder_checkpoint_path=folder_checkpoint_path,
        epoch=epoch,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        beta=beta,
        learning_rate=learning_rate,
        sigma=sigma,
        length_scale=length_scale,
        kernel_scales=kernel_scales,
        ground_truth=ground_truth,
        y_val=y_val,
        return_no_gt_imputation=return_no_gt_imputation,
        verbose=verbose,
    )

    if logs and verbose:
        end_time = time.time()
        print(f"\n> logs: imputation gpvae - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data, extras
