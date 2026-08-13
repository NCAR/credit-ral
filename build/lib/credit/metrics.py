import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from credit.loss import latitude_weights


class LatWeightedMetrics:
    def __init__(self, conf, training_mode=True):
        self.conf = conf
        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None
        if training_mode:
            self.ensemble_size = conf["trainer"].get("ensemble_size", 1)  # default value of 1 if not set
        else:
            self.ensemble_size = conf["predict"].get("ensemble_size", 1)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # forecast_datetime is passed for interface consistency but not used here

        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.0
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            # calculate ensemble mean, if ensemble_size=1, does nothing
            if self.ensemble_size > 1:
                pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:])  # b, ensemble, c, t, lat, lon
                std_dev = torch.std(pred, dim=1) * (self.ensemble_size + 1) / (self.ensemble_size - 1)  # std dev of ensemble
                pred = pred.mean(dim=1)

            error = pred - y
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = torch.sqrt(torch.sum(w_var * w_lat * pred_prime**2) * torch.sum(w_var * w_lat * y_prime**2)) + epsilon

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))))
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()
                # mean of std across all batches
                if self.ensemble_size > 1:
                    loss_dict[f"std_{var}"] = torch.mean(torch.sqrt(torch.mean(std_dev[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))))

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])
        if self.ensemble_size > 1:
            loss_dict["std"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "std_" in k])

        return loss_dict

class CorrDiffMetrics:
    def __init__(self, conf, training_mode=True):
        self.conf = conf

        # ---- Build variable name list matching model output channel order ---- #
        atmos_vars   = conf["data"].get("variables", [])
        surface_vars = conf["data"].get("surface_variables", [])
        diag_vars    = conf["data"].get("diagnostic_variables", [])
        levels       = conf["data"].get("levels", 1)

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += list(surface_vars)
        self.vars += list(diag_vars)
        self.n_vars = len(self.vars)

        # ---- Identify which channel is precipitation (if any) ---- #
        # Used to enable precip-specific metrics.  Match by substring; adjust
        # the keywords if your variable names differ.
        precip_keys = ("PWAT", "PRCP", "RAIN", "PREC", "Q_tot")
        self.precip_channels = [
            i for i, name in enumerate(self.vars)
            if any(k in name.upper() for k in precip_keys)
        ]

        # Thresholds in *normalized* units (post-scaler).  Override via YAML
        # if you want physical thresholds; see `precip_thresholds` below.
        self.precip_thresholds = conf.get("metrics", {}).get(
            "precip_thresholds", [0.1, 0.5, 1.0]
        )
        self.fss_window = conf.get("metrics", {}).get("fss_window", 9)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _rmse(err):
        # err: (B, T, H, W) for one channel
        return torch.sqrt((err ** 2).mean())

    @staticmethod
    def _mae(err):
        return err.abs().mean()

    @staticmethod
    def _bias(err):
        return err.mean()

    @staticmethod
    def _corr(pred, y):
        """Pearson correlation across all spatial/temporal points."""
        p = pred.flatten() - pred.mean()
        t = y.flatten() - y.mean()
        denom = (p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()).clamp_min(1e-7)
        return (p * t).sum() / denom

    # --------------------------------------------------------------------- #
    @staticmethod
    def _contingency(pred, y, thr):
        """Returns hits, misses, false alarms, correct negatives (scalars)."""
        p = (pred >= thr)
        t = (y    >= thr)
        hits     = ( p &  t).sum().float()
        misses   = (~p &  t).sum().float()
        falarms  = ( p & ~t).sum().float()
        corneg   = (~p & ~t).sum().float()
        return hits, misses, falarms, corneg

    def _csi(self, pred, y, thr):
        """Critical Success Index = hits / (hits + misses + false alarms)."""
        h, m, f, _ = self._contingency(pred, y, thr)
        return h / (h + m + f).clamp_min(1.0)

    def _freq_bias(self, pred, y, thr):
        """Frequency bias = (hits + false alarms) / (hits + misses).
        =1 perfect, >1 over-forecasts area, <1 under-forecasts."""
        h, m, _f, _ = self._contingency(pred, y, thr)
        return (h + _f) / (h + m).clamp_min(1.0)

    def _fss(self, pred, y, thr, window):
        """Fractions Skill Score at a given threshold and neighborhood size.

        FSS = 1 - MSE(P, T) / MSE_ref, where P and T are fractional coverage
        fields obtained by box-averaging the binary exceedance masks, and
        MSE_ref = mean(P^2) + mean(T^2).  Window is in grid cells; for 8 km
        data, window=9 corresponds to ~72 km neighborhoods.
        """
        # pred, y: (B, T, H, W)
        p_bin = (pred >= thr).float()
        t_bin = (y    >= thr).float()
        kernel = torch.ones((1, 1, window, window),
                            device=pred.device, dtype=pred.dtype) / (window * window)

        # Box-average per (B*T, 1, H, W).
        B, T, H, W = p_bin.shape
        p_frac = F.conv2d(p_bin.reshape(B * T, 1, H, W), kernel, padding=window // 2)
        t_frac = F.conv2d(t_bin.reshape(B * T, 1, H, W), kernel, padding=window // 2)

        mse    = ((p_frac - t_frac) ** 2).mean()
        ref    = (p_frac ** 2).mean() + (t_frac ** 2).mean()
        return 1.0 - mse / ref.clamp_min(1e-7)

    # --------------------------------------------------------------------- #
    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # pred, y: (B, C, T, H, W)
        if transform is not None:
            pred = transform(pred)
            y    = transform(y)

        # Bounds check so the error becomes informative if it ever happens
        # again, instead of a cryptic IndexError mid-loop.
        assert pred.shape[1] == self.n_vars, (
            f"Channel count mismatch: pred has {pred.shape[1]} channels, "
            f"metrics expects {self.n_vars}.  vars = {self.vars}"
        )

        out = {}
        with torch.no_grad():
            for i, var in enumerate(self.vars):
                p_i = pred[:, i].float()
                y_i = y[:, i].float()
                err = p_i - y_i

                out[f"rmse_{var}"] = self._rmse(err)
                out[f"mae_{var}"]  = self._mae(err)
                out[f"bias_{var}"] = self._bias(err)
                out[f"corr_{var}"] = self._corr(p_i, y_i)

            # Precipitation-specific metrics, only for identified precip channels.
            for i in self.precip_channels:
                var = self.vars[i]
                p_i = pred[:, i].float()
                y_i = y[:, i].float()

                for thr in self.precip_thresholds:
                    out[f"csi_{var}_thr{thr}"]      = self._csi(p_i, y_i, thr)
                    out[f"fbias_{var}_thr{thr}"]    = self._freq_bias(p_i, y_i, thr)
                    out[f"fss{self.fss_window}_{var}_thr{thr}"] = \
                        self._fss(p_i, y_i, thr, self.fss_window)

            # Scalar aggregates (averaged across variables, for tqdm bar).
            out["rmse"] = np.mean([v.cpu().item() for k, v in out.items() if k.startswith("rmse_")])
            out["mae"]  = np.mean([v.cpu().item() for k, v in out.items() if k.startswith("mae_")])
            out["bias"] = np.mean([v.cpu().item() for k, v in out.items() if k.startswith("bias_")])
            out["corr"] = np.mean([v.cpu().item() for k, v in out.items() if k.startswith("corr_")])

            # Aliases so your trainer's print line still works without changes.
            out["acc"] = out["corr"]   # 'acc' in the tqdm string

        return out


class DiagMetrics:
    def __init__(self, conf):
        self.conf = conf
        self.vars = conf["data"]["diagnostic_variables"]

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        self.w_var = None

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # forecast_datetime is passed for interface consistency but not used here

        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.0
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = pred - y
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = torch.sqrt(torch.sum(w_var * w_lat * pred_prime**2) * torch.sum(w_var * w_lat * y_prime**2)) + epsilon

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))))
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])

        return loss_dict


class LatWeightedMetricsClimatology:
    def __init__(self, conf, climatology=None):
        self.conf = conf
        self.climatology = climatology  # xarray Dataset with climatology data

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]

        self.vars += surface_vars
        self.vars += diag_vars
        self.acc_vars = surface_vars + diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None

    def get_climatology(self, forecast_datetime, variable):
        """Extract the climatology for the given forecast datetime and variable."""
        if isinstance(forecast_datetime, datetime):
            pass
        elif isinstance(forecast_datetime, int):
            forecast_datetime = datetime.utcfromtimestamp(forecast_datetime)  # Assumes integer datetime
        dayofyear = forecast_datetime.timetuple().tm_yday
        hour = forecast_datetime.hour

        # Extract climatology slice from xarray dataset
        climatology_slice = self.climatology[variable].sel(dayofyear=dayofyear, hour=hour, method="nearest")
        # Convert to PyTorch tensor
        return torch.tensor(climatology_slice.values, dtype=torch.float32)

    def __call__(self, pred, y, extras=None, transform=None, forecast_datetime=None):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights to device
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.0
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        loss_dict = {}
        with torch.no_grad():
            anomaly_scores = False
            if self.climatology and forecast_datetime:
                loss_dict = self.acc(
                    loss_dict,
                    pred,
                    y,
                    extras,
                    transform,
                    forecast_datetime,
                    w_var,
                    w_lat,
                )
                anomaly_scores = True

            # Compute RMSE, MSE, MAE for all vars
            error = pred - y
            for i, var in enumerate(self.vars):
                loss_dict[f"rmse_{var}"] = self.rmse(error[:, i], w_lat, w_var)
                loss_dict[f"mse_{var}"] = self.mse(error[:, i], w_lat, w_var)
                loss_dict[f"mae_{var}"] = self.mae(error[:, i], w_lat, w_var)
                if extras is not None:
                    for k, v in extras.items():
                        loss_dict[f"{k}_{var}"] = (v[:, i] * w_lat * w_var).mean()

            # Compute average metrics
            if anomaly_scores:
                loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
            loss_dict["rmse"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "rmse_" in k])
            loss_dict["mse"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "mse_" in k])
            loss_dict["mae"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "mae_" in k])

        return loss_dict

    def acc(self, loss_dict, pred, y, extras, transform, forecast_datetime, w_var, w_lat):
        # Compute ACC for acc_vars using anomalies
        anomalies_pred = []
        anomalies_y = []
        acc_pred = pred
        acc_y = y

        # Get the list of variables from the climatology file
        clim_vars = list(self.climatology.data_vars)

        # Ensure self.acc_vars is in the same order as clim_vars
        ordered_acc_vars = [var for var in clim_vars if var in self.vars]

        # Reorder acc_pred and acc_y to match ordered_acc_vars
        indices = [self.acc_vars.index(var) for var in ordered_acc_vars]
        acc_pred = acc_pred[:, indices]
        acc_y = acc_y[:, indices]

        # Compute anomalies
        for i, var in enumerate(ordered_acc_vars):
            clim = self.get_climatology(forecast_datetime, var).to(dtype=pred.dtype, device=pred.device).unsqueeze(0)
            anomalies_pred.append(acc_pred[:, i] - clim)
            anomalies_y.append(acc_y[:, i] - clim)

        anomalies_pred = torch.stack(anomalies_pred, dim=1)
        anomalies_y = torch.stack(anomalies_y, dim=1)

        for i, var in enumerate(self.acc_vars):
            pred_prime = anomalies_pred[:, i] - torch.mean(anomalies_pred[:, i])
            y_prime = anomalies_y[:, i] - torch.mean(anomalies_y[:, i])

            # Offset the denominator incase its zero.
            denominator = torch.sqrt(torch.sum(w_var * w_lat * pred_prime**2) * torch.sum(w_var * w_lat * y_prime**2))
            denominator = torch.maximum(denominator, torch.tensor(1e-8, device=denominator.device))
            loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
        return loss_dict

    def rmse(self, error, w_lat, w_var):
        return torch.mean(torch.sqrt(torch.mean(error**2 * w_lat * w_var, dim=(-2, -1))))

    def mse(self, error, w_lat, w_var):
        return (error**2 * w_lat * w_var).mean()

    def mae(self, error, w_lat, w_var):
        return (torch.abs(error) * w_lat * w_var).mean()


class LatWeightedMetricsEnsemble:
    """
    metrics for rollout_ens_batcher. will output full xarrays of rmse, std etc
    """

    def __init__(self, conf, training_mode=True):
        self.conf = conf
        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None
        if training_mode:
            self.ensemble_size = conf["trainer"].get("ensemble_size", 1)  # default value of 1 if not set
        else:
            self.ensemble_size = conf["predict"].get("ensemble_size", 1)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # pred is of shape (1, ensemble_size, c, t, lat, lon)
        # we are interested in gridcell-wise: ens mean, rmse, spread
        # TODO: spectrum
        # forecast_datetime is passed for interface consistency but not used here

        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.0
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:])  # b, ensemble, c, t, lat, lon
            # std dev of ensemble for each gridcell/variable
            loss_dict["ens_std"] = torch.std(pred, dim=1) * (self.ensemble_size + 1) / (self.ensemble_size - 1)

            # compute ensemble mean
            pred = pred.mean(dim=1)  # b, c, t, lat, lon
            loss_dict["ens_mean"] = pred
            loss_dict["ens_rmse"] = torch.sqrt((pred - y) ** 2)

        return loss_dict


if __name__ == "__main__":
    import yaml
    import logging
    import xarray as xr
    from credit.parser import credit_main_parser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Open an example config
    with open("../config/example-v2025.2.0.yml") as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)

    # Climatology file
    climatology_data = xr.open_dataset(conf["predict"]["climatology"])

    # Make some fake data

    true = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)
    pred = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)

    logger.info("Computing metrics. ACC without a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetrics(conf=conf)

    # Compute metrics
    results = metrics(pred, true)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Comptue metrics, and ACC correctly.

    logger.info("Computing metrics. ACC with a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetricsClimatology(conf=conf, climatology=climatology_data)

    # Define a forecast datetime (should align with the climatology dataset)
    forecast_datetime = datetime(2024, 6, 15, 12)  # Example forecast datetime

    # Compute metrics
    results = metrics(pred, true, forecast_datetime=forecast_datetime)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")
