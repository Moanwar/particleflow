from typing import Optional

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from mlpf.logger import _logger


def sliced_wasserstein_loss(y_pred, y_true, num_projections=200):
    theta = torch.randn(num_projections, y_true.shape[-1]).to(device=y_true.device)
    theta = theta / torch.sqrt(torch.sum(theta**2, dim=1, keepdims=True))
    A = torch.matmul(y_true, torch.transpose(theta, -1, -2))
    B = torch.matmul(y_pred, torch.transpose(theta, -1, -2))
    A_sorted = torch.sort(A, dim=-2).values
    B_sorted = torch.sort(B, dim=-2).values
    ret = torch.sqrt(torch.sum(torch.pow(A_sorted - B_sorted, 2), dim=[-1, -2]))
    return ret


def mlpf_loss(y, ypred, batch):
    loss = {}

    pid_class_weights = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 9.0, 9.0],
        # none  ch   nhad  gam  ele   mu  (log-scaled inverse-freq, measured on ttbar+zll)
        dtype=torch.float32,
        device=ypred["cls_id_onehot"].device
    )
    loss_obj_id = FocalLoss(alpha=pid_class_weights, gamma=2.0, reduction="none")

    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), dim=-1)
    nelem = torch.sum(batch.mask)
    npart = torch.sum(y["cls_id"] != 0)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle

    ypred["cls_binary"] = ypred["cls_binary"].permute((0, 2, 1))
    ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))

    loss_binary_classification = 10.0 * torch.nn.functional.cross_entropy(
        ypred["cls_binary"],
        (y["cls_id"] != 0).long(),
        reduction="none"
    )
    is_track_bin  = (batch.X[:, :, 0] == 1) | (batch.X[:, :, 0] == 4)  # typ=1 track + typ=4 GSF track
    is_had_ts_bin = (batch.X[:, :, 0] == 3)
    binary_boost = torch.ones_like(loss_binary_classification)
    binary_boost = torch.where(
        is_track_bin & (y["cls_id"] != 0),
        torch.full_like(binary_boost, 6.0),
        binary_boost
    )
    binary_boost = torch.where(
        is_had_ts_bin & (y["cls_id"] != 0),
        torch.full_like(binary_boost, 4.0),
        binary_boost
    )
    loss_binary_classification = loss_binary_classification * binary_boost

    loss_pid_classification = loss_obj_id(ypred["cls_id_onehot"], y["cls_id"]).reshape(y["cls_id"].shape)
    loss_pid_classification[y["cls_id"] == 0] *= 0

    loss_regression_pt = torch.nn.functional.mse_loss(ypred["pt"], y["pt"], reduction="none")
    loss_regression_eta = 1e-2 * torch.nn.functional.mse_loss(ypred["eta"], y["eta"], reduction="none")
    loss_regression_sin_phi = 1e-2 * torch.nn.functional.mse_loss(ypred["sin_phi"], y["sin_phi"], reduction="none")
    loss_regression_cos_phi = 1e-2 * torch.nn.functional.mse_loss(ypred["cos_phi"], y["cos_phi"], reduction="none")
    loss_regression_energy = torch.nn.functional.mse_loss(ypred["energy"], y["energy"], reduction="none")

    loss_regression_pt[y["cls_id"] == 0] *= 0
    loss_regression_eta[y["cls_id"] == 0] *= 0
    loss_regression_sin_phi[y["cls_id"] == 0] *= 0
    loss_regression_cos_phi[y["cls_id"] == 0] *= 0
    loss_regression_energy[y["cls_id"] == 0] *= 0

    # Energy-dependent boost for HAD_TS true n.had
    is_had_ts    = (batch.X[:, :, 0] == 3)
    is_true_nhad = (y["cls_id"] == 2)
    elem_e       = batch.X[:, :, 5]

    energy_boost = torch.ones_like(elem_e)
    energy_boost = torch.where(elem_e < 10,   torch.full_like(elem_e, 3.0),  energy_boost)
    energy_boost = torch.where(elem_e >= 10,  torch.full_like(elem_e, 8.0),  energy_boost)
    energy_boost = torch.where(elem_e >= 20,  torch.full_like(elem_e, 4.0),  energy_boost)
    energy_boost = torch.where(elem_e >= 50,  torch.full_like(elem_e, 5.0),  energy_boost)
    energy_boost = torch.where(elem_e >= 100, torch.full_like(elem_e, 2.0),  energy_boost)

    had_boost = torch.where(
        is_had_ts & is_true_nhad,
        energy_boost,
        torch.ones_like(loss_binary_classification)
    )
    loss_binary_classification = loss_binary_classification * had_boost

    # For HAD_TS none: zero regression loss
    is_had_ts_none = is_had_ts & (y["cls_id"] == 0)
    loss_regression_pt[is_had_ts_none]     *= 0
    loss_regression_energy[is_had_ts_none] *= 0

    loss_binary_classification[batch.mask == 0] *= 0
    loss_pid_classification[batch.mask == 0] *= 0
    loss_regression_pt[batch.mask == 0] *= 0
    loss_regression_eta[batch.mask == 0] *= 0
    loss_regression_sin_phi[batch.mask == 0] *= 0
    loss_regression_cos_phi[batch.mask == 0] *= 0
    loss_regression_energy[batch.mask == 0] *= 0

    sqrt_target_pt = torch.sqrt(torch.exp(y["pt"]) * batch.X[:, :, 1])
    loss_regression_pt *= sqrt_target_pt
    loss_regression_energy *= sqrt_target_pt

    loss["Regression_pt"]      = loss_regression_pt.sum() / npart
    loss["Regression_eta"]     = loss_regression_eta.sum() / npart
    loss["Regression_sin_phi"] = loss_regression_sin_phi.sum() / npart
    loss["Regression_cos_phi"] = loss_regression_cos_phi.sum() / npart
    loss["Regression_energy"]  = loss_regression_energy.sum() / npart

    loss["Classification_binary"] = loss_binary_classification.sum() / nelem
    loss["Classification"]        = loss_pid_classification.sum() / nelem

    loss["Total"] = (
        loss["Classification_binary"]
        + loss["Classification"]
        + loss["Regression_pt"]
        + loss["Regression_eta"]
        + loss["Regression_sin_phi"]
        + loss["Regression_cos_phi"]
        + loss["Regression_energy"]
    )
    loss_opt = loss["Total"]
    if torch.isnan(loss_opt):
        _logger.error(ypred)
        _logger.error(sqrt_target_pt)
        _logger.error(loss)
        raise Exception("Loss became NaN")

    for k in loss.keys():
        loss[k] = loss[k].detach()

    return loss_opt, loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0.0, reduction: str = "mean", ignore_index: int = -100):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)
        log_pt = torch.gather(log_p, 1, y.unsqueeze(dim=-1)).squeeze(dim=-1)
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
