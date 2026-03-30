#!/usr/bin/env python3
"""
MLPF TICL Validation Plotting Script
Usage: python3 plot_validation.py <experiment_dir>
Example: python3 plot_validation.py experiments/MLPF_ticl_test_my_training_20260324_152541_233559
"""
import sys
import os
import numpy as np
import awkward
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/cms/data/store/user/moanwar/mlpf_data/experiments/')

CLASS_NAMES = ["none", "ch.had", "n.had", "gamma", "ele", "mu"]
ELEM_NAMES  = {1: "Track", 2: "EM Trackster", 4: "HAD Trackster"}


def plot_loss_curve(exp_dir, plots_dir):
    steps, train_losses, valid_losses = [], [], []
    log_file = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_file):
        print("No train.log found, skipping loss curve")
        return
    with open(log_file) as f:
        for line in f:
            if "VALIDATION" in line:
                try:
                    parts      = line.split("|")
                    step       = int(parts[1].split("=")[1].split("/")[0].strip())
                    train_loss = float(parts[2].split("=")[1].strip())
                    valid_loss = float(parts[3].split("=")[1].strip())
                    steps.append(step)
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                except:
                    pass
    if not steps:
        print("No validation points found in log")
        return

    print(f"Loss curve: {len(steps)} validation points")
    print(f"  Best valid loss: {min(valid_losses):.4f} at step {steps[valid_losses.index(min(valid_losses))]}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_losses, "b-o", lw=2, markersize=5, label="Train Loss")
    ax.plot(steps, valid_losses, "r-o", lw=2, markersize=5, label="Valid Loss")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
    ax.set_title("MLPF TICL Training Loss")
    ax.legend(); ax.grid(alpha=0.3)
    ax.axvline(steps[valid_losses.index(min(valid_losses))], color="g",
               ls="--", alpha=0.5, label="Best valid")
    out = os.path.join(plots_dir, "loss_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def load_predictions(exp_dir, step=None):
    """Load parquet predictions - uses latest step if not specified."""
    preds_dirs = sorted(Path(exp_dir).glob("preds_step_*"))
    if not preds_dirs:
        print("No prediction directories found!")
        return None, None
    if step:
        pred_path = Path(exp_dir) / f"preds_step_{step}" / "cms_pf_ticl_nopu"
    else:
        pred_path = preds_dirs[-1] / "cms_pf_ticl_nopu"
        step = preds_dirs[-1].name.replace("preds_step_", "")

    print(f"Loading predictions from step {step}...")
    from mlpf.plotting.plot_utils import load_eval_data
    parquet_files = str(pred_path / "*.parquet")
    yvals, X, _ = load_eval_data(parquet_files, -1)
    print(f"  Loaded {len(yvals['target_pt'])} events")
    return yvals, X, step


def plot_particle_cls(yvals, plots_dir, step):
    fig, ax = plt.subplots(figsize=(9, 6))
    b = np.arange(-0.5, 6.5, 1)
    for key, label, color in [("target_cls_id","Target","blue"),
                               ("pred_cls_id","MLPF","red"),
                               ("cand_cls_id","PF","green")]:
        cls = awkward.to_numpy(awkward.flatten(yvals[key]))
        ax.hist(cls, bins=b, histtype="step", lw=2, label=label, color=color)
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Particle Class ID (step {step})")
    ax.set_ylabel("Particles / bin")
    out = os.path.join(plots_dir, f"particle_cls_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_particle_pt(yvals, plots_dir, step):
    fig, ax = plt.subplots(figsize=(9, 6))
    b = np.logspace(-1, 3, 100)
    for key, label, color in [("target_pt","Target","blue"),
                               ("pred_pt","MLPF","red"),
                               ("cand_pt","PF","green")]:
        pt  = awkward.to_numpy(awkward.flatten(yvals[key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("pt","cls_id")]))
        pt  = pt[cls != 0]
        if len(pt): ax.hist(pt, bins=b, histtype="step", lw=2, label=f"{label} ({len(pt)})", color=color)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pT (GeV)"); ax.set_ylabel("Particles / bin")
    ax.set_title(f"Particle pT (step {step})")
    ax.legend(); ax.grid(alpha=0.3)
    out = os.path.join(plots_dir, f"particle_pt_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_particle_eta(yvals, plots_dir, step):
    fig, ax = plt.subplots(figsize=(9, 6))
    b = np.linspace(-4, 4, 100)
    for key, label, color in [("target_eta","Target","blue"),
                               ("pred_eta","MLPF","red"),
                               ("cand_eta","PF","green")]:
        eta = awkward.to_numpy(awkward.flatten(yvals[key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("eta","cls_id")]))
        eta = eta[cls != 0]
        if len(eta): ax.hist(eta, bins=b, histtype="step", lw=2, label=label, color=color)
    ax.set_xlabel("eta"); ax.set_ylabel("Particles / bin")
    ax.set_title(f"Particle eta (step {step})")
    ax.legend(); ax.grid(alpha=0.3)
    out = os.path.join(plots_dir, f"particle_eta_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_per_pid(yvals, plots_dir, step):
    """Plot pT and eta for each particle type separately."""
    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue  # skip "none"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # pT
        b = np.logspace(-1, 3, 80)
        for key, label, color in [("target_pt","Target","blue"),
                                   ("pred_pt","MLPF","red"),
                                   ("cand_pt","PF","green")]:
            pt  = awkward.to_numpy(awkward.flatten(yvals[key]))
            cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("pt","cls_id")]))
            pt  = pt[cls == pid_idx]
            if len(pt): ax1.hist(pt, bins=b, histtype="step", lw=2, label=f"{label} ({len(pt)})", color=color)
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("pT (GeV)"); ax1.set_ylabel("Counts")
        ax1.set_title(f"{pid_name} pT"); ax1.legend(); ax1.grid(alpha=0.3)

        # eta
        b = np.linspace(-4, 4, 80)
        for key, label, color in [("target_eta","Target","blue"),
                                   ("pred_eta","MLPF","red"),
                                   ("cand_eta","PF","green")]:
            eta = awkward.to_numpy(awkward.flatten(yvals[key]))
            cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("eta","cls_id")]))
            eta = eta[cls == pid_idx]
            if len(eta): ax2.hist(eta, bins=b, histtype="step", lw=2, label=label, color=color)
        ax2.set_xlabel("eta"); ax2.set_ylabel("Counts")
        ax2.set_title(f"{pid_name} eta"); ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle(f"Particle: {pid_name} (step {step})", fontsize=13)
        out = os.path.join(plots_dir, f"particle_{pid_name}_step{step}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {out}")


def plot_efficiency_purity(yvals, plots_dir, step):
    """Plot efficiency and fake rate per particle class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue
        ax = axes[pid_idx]
        b  = np.logspace(-1, 3, 40)
        bc = 0.5 * (b[:-1] + b[1:])

        tgt_pt  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
        tgt_cls = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
        prd_pt  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
        prd_cls = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

        # efficiency: fraction of target particles predicted correctly
        tgt_msk  = tgt_cls == pid_idx
        prd_msk  = prd_cls == pid_idx
        n_target = np.histogram(tgt_pt[tgt_msk], bins=b)[0]
        n_pred   = np.histogram(prd_pt[prd_msk], bins=b)[0]

        eff = np.where(n_target > 0, n_pred / np.maximum(n_target, 1), 0)
        ax.plot(bc, eff, "r-o", lw=2, markersize=3, label="pred/target ratio")
        ax.axhline(1.0, color="k", ls="--", alpha=0.5)
        ax.set_xscale("log"); ax.set_ylim(0, 2)
        ax.set_xlabel("pT (GeV)"); ax.set_ylabel("N_pred / N_target")
        ax.set_title(pid_name); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Pred/Target ratio per particle type (step {step})", fontsize=13)
    out = os.path.join(plots_dir, f"efficiency_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_energy_response(yvals, plots_dir, step):
    """Plot energy response pred/target."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    b = np.linspace(0, 3, 100)

    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue
        ax = axes[pid_idx]
        tgt_e   = awkward.to_numpy(awkward.flatten(yvals["target_energy"]))
        tgt_cls = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
        prd_e   = awkward.to_numpy(awkward.flatten(yvals["pred_energy"]))
        prd_cls = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))
        cnd_e   = awkward.to_numpy(awkward.flatten(yvals["cand_energy"]))
        cnd_cls = awkward.to_numpy(awkward.flatten(yvals["cand_cls_id"]))

        msk_t = (tgt_cls == pid_idx) & (tgt_e > 0)
        msk_p = (prd_cls == pid_idx) & (prd_e > 0)
        msk_c = (cnd_cls == pid_idx) & (cnd_e > 0)

        # ratio only where both exist — use matched pairs via same index
        msk = msk_t & msk_p
        if msk.sum() > 10:
            ratio_p = prd_e[msk] / tgt_e[msk]
            ratio_p = ratio_p[np.isfinite(ratio_p) & (ratio_p < 10)]
            ax.hist(ratio_p, bins=b, histtype="step", lw=2, label=f"MLPF/Target ({len(ratio_p)})", color="red")
        msk = msk_t & msk_c
        if msk.sum() > 10:
            ratio_c = cnd_e[msk] / tgt_e[msk]
            ratio_c = ratio_c[np.isfinite(ratio_c) & (ratio_c < 10)]
            ax.hist(ratio_c, bins=b, histtype="step", lw=2, label=f"PF/Target ({len(ratio_c)})", color="green")

        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("E_pred / E_target"); ax.set_ylabel("Counts")
        ax.set_title(pid_name); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Energy response per particle (step {step})", fontsize=13)
    out = os.path.join(plots_dir, f"energy_response_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else         "/cms/data/store/user/moanwar/mlpf_data/experiments/"
    step = sys.argv[2] if len(sys.argv) > 2 else None

    plots_dir = os.path.join(exp_dir, "validation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nExperiment: {exp_dir}")
    print(f"Plots dir:  {plots_dir}\n")

    # 1. Loss curve
    print("1. Plotting loss curve...")
    plot_loss_curve(exp_dir, plots_dir)

    # 2. Load predictions
    print("\n2. Loading predictions...")
    yvals, X, step = load_predictions(exp_dir, step)
    if yvals is None:
        return

    # 3. Particle class distribution
    print("\n3. Particle class distribution...")
    plot_particle_cls(yvals, plots_dir, step)

    # 4. Overall pT and eta
    print("\n4. Overall pT and eta...")
    plot_particle_pt(yvals, plots_dir, step)
    plot_particle_eta(yvals, plots_dir, step)

    # 5. Per-PID plots
    print("\n5. Per-PID pT and eta...")
    plot_per_pid(yvals, plots_dir, step)

    # 6. Efficiency/purity
    print("\n6. Pred/target ratio per PID...")
    plot_efficiency_purity(yvals, plots_dir, step)

    # 7. Energy response
    print("\n7. Energy response per PID...")
    plot_energy_response(yvals, plots_dir, step)

    print(f"\n✅ All plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
