import torch
import numpy as np
import xarray as xr


class LatWeightedMetrics:

    def __init__(self, conf):
        lat_file = conf['loss']['latitude_weights']
        atmos_vars = conf['data']['variables']
        self.vars = atmos_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(lat_file)["latitude"].values
            w_lat = np.cos(np.deg2rad(lat))
            w_lat = w_lat / w_lat.mean()
            self.w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)

        self.w_var = None
        if conf["loss"]["use_variable_weights"]:
            var_weights = [value if isinstance(value, list) else [value] for value in conf["loss"]["variable_weights"].values()]
            var_weights = [item for sublist in var_weights for item in sublist]
            self.w_var = torch.from_numpy(var_weights).unsqueeze(0).unsqueeze(-1)

    def __call__(self, pred, y, clim=None, transform=None):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = (pred - y)
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = torch.sqrt(
                    torch.sum(w_var * w_lat * pred_prime**2) * torch.sum(w_var * w_lat * y_prime**2)
                ) + epsilon

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1)))
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])

        return loss_dict


def anomaly_correlation_coefficient(pred, true):

    pred = pred.float()
    true = true.float()

    B, C, H, W = pred.size()

    # Flatten the spatial dimensions
    pred_flat = pred.view(B, C, -1)
    true_flat = true.view(B, C, -1)

    # Mean over spatial dimensions
    pred_mean = torch.mean(pred_flat, dim=-1, keepdim=True)
    true_mean = torch.mean(true_flat, dim=-1, keepdim=True)

    # Anomaly calculation
    pred_anomaly = pred_flat - pred_mean
    true_anomaly = true_flat - true_mean

    # Covariance matrix
    covariance_matrix = torch.bmm(pred_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Variance terms
    pred_var = torch.bmm(pred_anomaly, pred_anomaly.transpose(1, 2)) / (H * W - 1)
    true_var = torch.bmm(true_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Anomaly Correlation Coefficient
    acc_numerator = torch.einsum('bii->b', covariance_matrix).sum()
    acc_denominator = torch.sqrt(torch.einsum('bii->b', pred_var).sum() * torch.einsum('bii->b', true_var).sum())

    # Avoid division by zero
    epsilon = 1e-8
    acc = acc_numerator / (acc_denominator + epsilon)

    return acc.item()
