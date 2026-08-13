# AI Model Repository for RAL-GWC Project

A fork/extension of [MILES-CREDIT](https://miles-credit.readthedocs.io/en/latest/) (`credit` package,
version 2025.3.0) used for the RAL GWC work.

## Installation
* Run `create_derecho_env.sh` — creates the `credit-gwc3` conda env (Python 3.11), installs the
  Derecho MPI-enabled PyTorch wheels, then `pip install -e .`.
* Alternative: MILES-CREDIT [documentation](https://miles-credit.readthedocs.io/en/latest/)

---

## Repository Layout

### Top level

| Path | Purpose |
| --- | --- |
| `credit/` | Installable packages. |
| `applications/` | Training and prediction drivers that use YAML configs. |
| `GWC_MESO/` | CONUS404 emulation project: data preprocessing notebooks and run configs. |
| `GWC_MICRO/` | FastEddy LES emulation project: preprocessing, verification. |
| `create_derecho_env.sh` | Environment bootstrap script for NCAR Derecho. |

### `credit/` — core library

**Subpackages**

| Path | Purpose |
| --- | --- |
| `models/` | Network architectures plus `base_model.py`, `checkpoint.py`, `reset.py`. |
| `datasets/` | PyTorch datasets/dataloaders per task and data source: ERA5 single-step and multi-step, WRF, LES, downscaling, diagnostics, real-time prediction, and `load_dataset_and_dataloader.py`. |
| `trainers/` | Training loops, one per task/model family (`trainerERA5*`, `trainerWRF*`, `trainerLES`, `trainerDscale`, `trainerDiag`, `trainerCorrDiff`, `trainer404`), on top of `base_trainer.py`. |
| `transforms/` | Normalization / variable-packing pipelines per domain (`transforms_global`, `_wrf`, `_les`, `_dscale`, `_diag`, `_quantile`). |
| `losses/` | Loss functions: weighted/latitude-weighted losses, spectral, CRPS variants (`kcrps`, `almost_fair_crps`), logcosh, MSLE, power, xtanh/xsigmoid, and task-specific LES/diagnostic losses. |
| `ensemble/` | Ensemble generation and scoring: `gaussian_noise.py`, `bred_vector.py`, `crps.py`. |
| `verification/` | Deterministic (`standard.py`) and ensemble (`ensemble.py`) verification metrics. |
| `metadata/` | Static reference data: ERA5/CESM level info NetCDFs, variable metadata YAMLs, model-level index tables. |

**Key modules**

| File | Purpose |
| --- | --- |
| `parser.py` | YAML config parsing/validation — the entry point for all experiment configuration. |
| `data.py` | Low-level Zarr/NetCDF loading helpers and variable handling. |
| `xr_sampler.py` | xarray-based sampling of forecast windows from stored datasets. |
| `forecast.py` | Forecast time-stepping / rollout scheduling. |
| `output.py` | Writing model predictions to NetCDF/Zarr with proper metadata. |
| `loss.py`, `metrics.py` | Top-level loss and metric selection from config. |
| `scheduler.py`, `mixed_precision.py`, `distributed.py`, `seed.py` | LR schedules, AMP/FSDP setup, DDP/FSDP distributed helpers, reproducibility. |
| `pbs.py` | PBS job-script generation and submission for Derecho. |
| `physics_core.py`, `physics_constants.py` | Physics-based constraints (mass/energy/water conservation) and constants. |
| `grid.py`, `interp.py`, `regrid.py`, `boundary_padding.py` | Grid definitions, interpolation/regridding, boundary padding for limited-area domains. |

### `applications/` — run scripts

| File | Purpose |
| --- | --- |
| `train.py` | Generic (ERA5/global) training driver. |
| `WRF_train.py`, `WRF_train_multi.py`, `WRF_train_subset.py` | CONUS404/WRF training: single-step, multi-step rollout, and subset-domain variants. |
| `WRF_pred_future.py`, `WRF_pred_future_subset.py`, `WRF_pred_teacher.py`, `WRF_pred_metrics.py` | WRF inference: free-running forecasts, teacher-forced runs, and scoring. |
| `LES_train.py`, `LES_pred.py` | FastEddy LES emulator training and inference. |
| `dscale_train.py`, `dscale_pred_metrics.py` | Downscaling model training and verification. |
| `diag_train.py`, `diag_pred_metrics.py` | Diagnostic-variable model training and verification. |
| `corrdiff_train.py`, `corrdiff_pred.py` | CorrDiff (diffusion-based downscaling) training and inference. |
| `scaler.py` | Compute mean/std (and other) scaling files from a dataset. |
| `model_summary.py` | Print model architecture/parameter counts from a config. |
| `config_visualization_example.yml` | Example config for the visualization tools. |

### `GWC_MESO/` — CONUS404 mesoscale emulation

Domain suffixes: `FULL` (full CONUS), `GP` (Great Plains), `PNW` (Pacific Northwest), `SW` (Southwest).

| Path | Purpose |
| --- | --- |
| `data_preprocessing_{FULL,GP,PNW,SW}/` | Per-domain notebooks that build the training data: raw CONUS404 gathering (`DATA00`), static fields (`DATA01`), 1-hourly packing to Zarr (`DATA02`), ERA5 interpolation (`DATA03`), downscaling data prep (`DATA04`); `OPT00`–`OPT02` create mean/std, residual coefficient, and residual-norm files; `QC*`/`PLOT*` check and plot the domain; `qsub_*` notebooks generate PBS jobs; `scripts/` holds the generated job scripts; `data_config_*.yml` define variables and paths. |
| `ERA5_1h_{FULL,GP,SW}/` | ERA5 boundary/forcing preprocessing at 1-hourly (and 3-hourly/8 km) resolution for each domain. |
| `GDAS_6h_GP/` | GDAS 6-hourly analysis gathering and preprocessing, used for real-data initialization. |
| `example/CONUS_FULL/`, `example/CONUS_GP/` | Ready-to-run experiment configs and launch scripts: single-step (`model_single`), multi-step rollouts by lead time (`model_multi_NN`), hyperparameter tuning (`model_tune`/`model_opt`), inference (`model_pred*`), and long climate runs driven by CESM members, GDAS, or varying boundary-update frequency (`model_clim_*`). |

### `GWC_MICRO/` — FastEddy LES emulation

| Path | Purpose |
| --- | --- |
| `data_preprocessing/` | FastEddy data exploration, static-field prep, packing, downsampling for boundaries, and mean/std, residual-coefficient, and variable-weight generation; `scripts/` holds generated batch jobs. |
| `dev_foundation_model/` | Development notebooks: foundation-model prototyping, dataset/transform/trainer development (`DEV00`–`DEV06`), inference (with and without boundaries), precipitation loss, climate runs, and FuXi checks. |
| `libs/` | Standalone analysis utilities used by the notebooks: verification (`verif_utils`, `score_utils`, `seeps_utils`), pressure-level handling (`plevel_utils`), interpolation, physics, solar, graph, and preprocessing helpers. |
| `example/` | Model configs (`model.yml`, `model_single.yml`, `model_predict.yml`) and matching launch shell scripts. |
| `visualization/` | Notebooks for training-log plots and downscaled surface-field figures. |
