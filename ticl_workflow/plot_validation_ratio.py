#!/usr/bin/env python3
"""
MLPF TICL Validation Plotting Script - Enhanced with Ratio Panels
Usage: python3 plot_validation.py <experiment_dir> [step] [pt_cut]

  pt_cut  : minimum pT applied to ALL plots (default: 0.0, i.e. no cut)
  Example : python3 plot_validation.py /path/to/exp 5000 1.0
"""
import sys
import os
sys.path.insert(0, '/home/moanwar/mlpf/particleflow')
import numpy as np
import awkward
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/cms/data/store/user/moanwar/mlpf_data/experiments_heptv2_v1/')

CLASS_NAMES = ["none", "ch.had", "n.had", "gamma", "ele", "mu"]
ELEM_NAMES  = {1: "Track", 2: "EM Trackster", 4: "HAD Trackster"}

COLORS = {
    'target':      '#1f77b4',
    'mlpf':        '#d62728',
    'pf':          '#2ca02c',
    'ratio':       '#ff7f0e',
    'ratio_hline': '#000000',
}

# ── Global pT cut ─────────────────────────────────────────────────────────────
# Changed via command-line (3rd argument) or by editing this default.
PT_CUT: float = 0.0   # GeV — overridden in main()


def _apply_pt_cut(pt_array, *other_arrays):
    """
    Return boolean mask for pt_array > PT_CUT, then apply it to every array
    in other_arrays as well.  Usage:
        mask = _apply_pt_cut(pt, cls, eta)
        pt_sel, cls_sel, eta_sel = pt[mask], cls[mask], eta[mask]
    """
    mask = pt_array > PT_CUT
    return (arr[mask] for arr in (pt_array, *other_arrays))


def _cut_label():
    """Small annotation string for plot titles / labels."""
    if PT_CUT > 0:
        return f"$p_T > {PT_CUT:.4g}$ GeV"
    return "no $p_T$ cut"


# ── Loss curve ────────────────────────────────────────────────────────────────
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
                except Exception:
                    pass
    if not steps:
        print("No validation points found in log")
        return

    print(f"Loss curve: {len(steps)} validation points")
    print(f"  Best valid loss: {min(valid_losses):.4f} at step "
          f"{steps[valid_losses.index(min(valid_losses))]}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_losses, "b-o", lw=2, markersize=5, label="Train Loss")
    ax.plot(steps, valid_losses, "r-o", lw=2, markersize=5, label="Valid Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("MLPF TICL Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(steps[valid_losses.index(min(valid_losses))],
               color="g", ls="--", alpha=0.5, label="Best valid")
    out = os.path.join(plots_dir, "loss_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Load predictions ──────────────────────────────────────────────────────────
def load_predictions(exp_dir, step=None):
    preds_dirs = sorted(Path(exp_dir).glob("preds_step_*"))
    if not preds_dirs:
        print("No prediction directories found!")
        return None, None, None
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


# ── Particle class distribution ───────────────────────────────────────────────
def plot_particle_cls(yvals, plots_dir, step):
    """Class distribution — include 'none' category while applying pT cut to others."""
    fig, ax = plt.subplots(figsize=(9, 6))
    b = np.arange(-0.5, 6.5, 1)

    for pt_key, cls_key, label, color in [
        ("target_pt", "target_cls_id", "Target", COLORS['target']),
        ("pred_pt",   "pred_cls_id",   "MLPF",   COLORS['mlpf']),
        ("cand_pt",   "cand_cls_id",   "PF",     COLORS['pf']),
    ]:
        pt  = awkward.to_numpy(awkward.flatten(yvals[pt_key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[cls_key]))
        
        # Apply pT cut but KEEP "none" (class 0) regardless of pT
        if PT_CUT > 0:
            # Keep particles that pass pT cut OR are "none" class
            mask = (pt > PT_CUT) | (cls == 0)
            cls_filtered = cls[mask]
        else:
            cls_filtered = cls
            
        ax.hist(cls_filtered, bins=b, histtype="step", lw=2, 
                label=f"{label} ({len(cls_filtered)})", color=color)

    ax.set_xticks(range(6))
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Particle Class ID (step {step},  {_cut_label()})")
    ax.set_ylabel("Particles / bin")
    out = os.path.join(plots_dir, f"particle_cls_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    
# ── Helper: safe ratio ────────────────────────────────────────────────────────
def _safe_ratio(num, den):
    num = num.astype(float)
    den = den.astype(float)
    ratio = np.zeros_like(num)
    mask  = den > 0
    ratio[mask] = num[mask] / den[mask]
    err = np.zeros_like(ratio)
    mask_err = mask & (num > 0)
    err[mask_err] = ratio[mask_err] * np.sqrt(
        1.0/num[mask_err] + 1.0/den[mask_err])
    return ratio, err


# ── pT with ratio ─────────────────────────────────────────────────────────────
def plot_particle_pt_with_ratio(yvals, plots_dir, step):
    fig = plt.figure(figsize=(10, 8))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1])

    b = np.logspace(-1, 3, 100)
    bin_centers = np.sqrt(b[:-1] * b[1:])

    for pt_key, cls_key, label, color in [
        ("target_pt", "target_cls_id", "Target", COLORS['target']),
        ("pred_pt",   "pred_cls_id",   "MLPF",   COLORS['mlpf']),
        ("cand_pt",   "cand_cls_id",   "PF",     COLORS['pf']),
    ]:
        pt  = awkward.to_numpy(awkward.flatten(yvals[pt_key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[cls_key]))
        pt_cut, cls_cut = _apply_pt_cut(pt, cls)
        pt_sel = pt_cut[cls_cut != 0]
        ax_main.hist(pt_sel, bins=b, histtype="step", lw=2,
                     label=f"{label} ({len(pt_sel)})", color=color)

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Particles / bin")
    ax_main.set_title(f"Particle $p_T$ (step {step},  {_cut_label()})")
    ax_main.legend(loc='upper right')
    ax_main.grid(alpha=0.3, which='both')
    ax_main.tick_params(labelbottom=False)

    # Ratio MLPF / Target
    tgt_pt  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_cls = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_cls = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt,  tgt_cls = _apply_pt_cut(tgt_pt,  tgt_cls)
    prd_pt,  prd_cls = _apply_pt_cut(prd_pt,  prd_cls)

    th, _ = np.histogram(tgt_pt[tgt_cls != 0], bins=b)
    mh, _ = np.histogram(prd_pt[prd_cls != 0], bins=b)
    ratio, err = _safe_ratio(mh, th)

    ax_ratio.step(bin_centers, ratio, where='mid', color=COLORS['ratio'], lw=2,
                  label='MLPF/Target')
    ax_ratio.fill_between(bin_centers, ratio - err, ratio + err,
                          alpha=0.3, color=COLORS['ratio'])
    ax_ratio.axhline(1.0, color=COLORS['ratio_hline'], ls='--', lw=1, alpha=0.7)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("$p_T$ (GeV)")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(alpha=0.3, which='both')
    ax_ratio.legend(loc='upper right', fontsize=8)

    out = os.path.join(plots_dir, f"particle_pt_with_ratio_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── η with ratio ──────────────────────────────────────────────────────────────
def plot_particle_eta_with_ratio(yvals, plots_dir, step):
    fig = plt.figure(figsize=(10, 8))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1])

    b = np.linspace(-4, 4, 100)
    bin_centers = (b[:-1] + b[1:]) / 2

    for pt_key, eta_key, cls_key, label, color in [
        ("target_pt", "target_eta", "target_cls_id", "Target", COLORS['target']),
        ("pred_pt",   "pred_eta",   "pred_cls_id",   "MLPF",   COLORS['mlpf']),
        ("cand_pt",   "cand_eta",   "cand_cls_id",   "PF",     COLORS['pf']),
    ]:
        pt  = awkward.to_numpy(awkward.flatten(yvals[pt_key]))
        eta = awkward.to_numpy(awkward.flatten(yvals[eta_key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[cls_key]))
        pt_cut, eta_cut, cls_cut = _apply_pt_cut(pt, eta, cls)
        eta_sel = eta_cut[cls_cut != 0]
        ax_main.hist(eta_sel, bins=b, histtype="step", lw=2,
                     label=f"{label} ({len(eta_sel)})", color=color)

    ax_main.set_ylabel("Particles / bin")
    ax_main.set_title(f"Particle $\\eta$ (step {step},  {_cut_label()})")
    ax_main.legend(loc='upper right')
    ax_main.grid(alpha=0.3)
    ax_main.tick_params(labelbottom=False)

    # Ratio
    tgt_pt  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_eta = awkward.to_numpy(awkward.flatten(yvals["target_eta"]))
    tgt_cls = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_eta = awkward.to_numpy(awkward.flatten(yvals["pred_eta"]))
    prd_cls = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt, tgt_eta, tgt_cls = _apply_pt_cut(tgt_pt, tgt_eta, tgt_cls)
    prd_pt, prd_eta, prd_cls = _apply_pt_cut(prd_pt, prd_eta, prd_cls)

    th, _ = np.histogram(tgt_eta[tgt_cls != 0], bins=b)
    mh, _ = np.histogram(prd_eta[prd_cls != 0], bins=b)
    ratio, err = _safe_ratio(mh, th)

    ax_ratio.step(bin_centers, ratio, where='mid', color=COLORS['ratio'], lw=2,
                  label='MLPF/Target')
    ax_ratio.fill_between(bin_centers, ratio - err, ratio + err,
                          alpha=0.3, color=COLORS['ratio'])
    ax_ratio.axhline(1.0, color=COLORS['ratio_hline'], ls='--', lw=1, alpha=0.7)
    ax_ratio.set_xlabel("$\\eta$")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(alpha=0.3)
    ax_ratio.legend(loc='upper right', fontsize=8)

    out = os.path.join(plots_dir, f"particle_eta_with_ratio_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Stacked pT ────────────────────────────────────────────────────────────────
def plot_stacked_pt(yvals, plots_dir, step):
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    b = np.logspace(-1, 3, 80)
    bin_centers = np.sqrt(b[:-1] * b[1:])
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    mlpf_stacks, target_stacks = [], []

    # Pre-cut arrays
    tgt_pt_all  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt_all  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt_all, tgt_cls_all = _apply_pt_cut(tgt_pt_all, tgt_cls_all)
    prd_pt_all, prd_cls_all = _apply_pt_cut(prd_pt_all, prd_cls_all)

    for pid_idx in range(1, 6):
        mh, _ = np.histogram(prd_pt_all[prd_cls_all == pid_idx], bins=b)
        th, _ = np.histogram(tgt_pt_all[tgt_cls_all == pid_idx], bins=b)
        mlpf_stacks.append(mh)
        target_stacks.append(th)

    bottom_m = np.zeros(len(bin_centers))
    bottom_t = np.zeros(len(bin_centers))
    for i, (mh, th, pid_name) in enumerate(
            zip(mlpf_stacks, target_stacks, CLASS_NAMES[1:])):
        ax_main.bar(bin_centers, mh, width=np.diff(b), bottom=bottom_m,
                    label=f"MLPF {pid_name}", color=colors[i],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_main.bar(bin_centers, th, width=np.diff(b), bottom=bottom_t,
                    label=f"Target {pid_name}", color=colors[i],
                    alpha=0.3, edgecolor='none')
        bottom_m += mh
        bottom_t += th

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Particles / bin")
    ax_main.set_title(
        f"Stacked $p_T$: MLPF vs Target (step {step},  {_cut_label()})")
    ax_main.legend(loc='upper right', ncol=2, fontsize=9)
    ax_main.grid(alpha=0.3, which='both')
    ax_main.tick_params(labelbottom=False)

    total_m = np.sum(mlpf_stacks, axis=0)
    total_t = np.sum(target_stacks, axis=0)
    ratio, err = _safe_ratio(total_m, total_t)

    ax_ratio.step(bin_centers, ratio, where='mid', color='black', lw=2,
                  label='Total MLPF/Target')
    ax_ratio.fill_between(bin_centers, ratio - err, ratio + err,
                          alpha=0.3, color='gray')
    ax_ratio.axhline(1.0, color='red', ls='--', lw=1, alpha=0.7)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("$p_T$ (GeV)")
    ax_ratio.set_ylabel("Total Ratio")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(alpha=0.3, which='both')
    ax_ratio.legend(loc='upper right', fontsize=9)

    out = os.path.join(plots_dir, f"stacked_pt_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Stacked η ─────────────────────────────────────────────────────────────────
def plot_stacked_eta(yvals, plots_dir, step):
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    b = np.linspace(-4, 4, 80)
    bin_centers = (b[:-1] + b[1:]) / 2
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    mlpf_stacks, target_stacks = [], []

    tgt_pt_all  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_eta_all = awkward.to_numpy(awkward.flatten(yvals["target_eta"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt_all  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_eta_all = awkward.to_numpy(awkward.flatten(yvals["pred_eta"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt_all, tgt_eta_all, tgt_cls_all = _apply_pt_cut(
        tgt_pt_all, tgt_eta_all, tgt_cls_all)
    prd_pt_all, prd_eta_all, prd_cls_all = _apply_pt_cut(
        prd_pt_all, prd_eta_all, prd_cls_all)

    for pid_idx in range(1, 6):
        mh, _ = np.histogram(prd_eta_all[prd_cls_all == pid_idx], bins=b)
        th, _ = np.histogram(tgt_eta_all[tgt_cls_all == pid_idx], bins=b)
        mlpf_stacks.append(mh)
        target_stacks.append(th)

    bottom_m = np.zeros(len(bin_centers))
    bottom_t = np.zeros(len(bin_centers))
    for i, (mh, th, pid_name) in enumerate(
            zip(mlpf_stacks, target_stacks, CLASS_NAMES[1:])):
        ax_main.bar(bin_centers, mh, width=np.diff(b)[0], bottom=bottom_m,
                    label=f"MLPF {pid_name}", color=colors[i],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_main.bar(bin_centers, th, width=np.diff(b)[0], bottom=bottom_t,
                    label=f"Target {pid_name}", color=colors[i],
                    alpha=0.3, edgecolor='none')
        bottom_m += mh
        bottom_t += th

    ax_main.set_ylabel("Particles / bin")
    ax_main.set_title(
        f"Stacked $\\eta$: MLPF vs Target (step {step},  {_cut_label()})")
    ax_main.legend(loc='upper right', ncol=2, fontsize=9)
    ax_main.grid(alpha=0.3)
    ax_main.tick_params(labelbottom=False)

    total_m = np.sum(mlpf_stacks, axis=0)
    total_t = np.sum(target_stacks, axis=0)
    ratio, err = _safe_ratio(total_m, total_t)

    ax_ratio.step(bin_centers, ratio, where='mid', color='black', lw=2,
                  label='Total MLPF/Target')
    ax_ratio.fill_between(bin_centers, ratio - err, ratio + err,
                          alpha=0.3, color='gray')
    ax_ratio.axhline(1.0, color='red', ls='--', lw=1, alpha=0.7)
    ax_ratio.set_xlabel("$\\eta$")
    ax_ratio.set_ylabel("Total Ratio")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.grid(alpha=0.3)
    ax_ratio.legend(loc='upper right', fontsize=9)

    out = os.path.join(plots_dir, f"stacked_eta_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Stacked efficiency ────────────────────────────────────────────────────────
def plot_stacked_efficiency(yvals, plots_dir, step):
    fig, ax = plt.subplots(figsize=(12, 8))
    b = np.logspace(-1, 3, 50)
    bin_centers = np.sqrt(b[:-1] * b[1:])
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    tgt_pt_all  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt_all  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt_all, tgt_cls_all = _apply_pt_cut(tgt_pt_all, tgt_cls_all)
    prd_pt_all, prd_cls_all = _apply_pt_cut(prd_pt_all, prd_cls_all)

    for pid_idx, pid_name, color in zip(range(1, 6), CLASS_NAMES[1:], colors):
        th, _ = np.histogram(tgt_pt_all[tgt_cls_all == pid_idx], bins=b)
        mh, _ = np.histogram(prd_pt_all[prd_cls_all == pid_idx], bins=b)
        eff = np.where(th > 0, mh / np.maximum(th, 1).astype(float), 0)
        ax.plot(bin_centers, eff, 'o-', lw=2, markersize=4,
                label=pid_name, color=color)

    ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0, 2)
    ax.set_xlabel("$p_T$ (GeV)")
    ax.set_ylabel("MLPF / Target")
    ax.set_title(
        f"Efficiency per Particle Type (step {step},  {_cut_label()})")
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, which='both')

    out = os.path.join(plots_dir, f"stacked_efficiency_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Per-PID pT and η ──────────────────────────────────────────────────────────
def plot_per_pid(yvals, plots_dir, step):
    tgt_pt_all  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_eta_all = awkward.to_numpy(awkward.flatten(yvals["target_eta"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt_all  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_eta_all = awkward.to_numpy(awkward.flatten(yvals["pred_eta"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    # Apply pT cut once
    tgt_pt_all, tgt_eta_all, tgt_cls_all = _apply_pt_cut(
        tgt_pt_all, tgt_eta_all, tgt_cls_all)
    prd_pt_all, prd_eta_all, prd_cls_all = _apply_pt_cut(
        prd_pt_all, prd_eta_all, prd_cls_all)

    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue

        tgt_pt_sel  = tgt_pt_all[tgt_cls_all  == pid_idx]
        prd_pt_sel  = prd_pt_all[prd_cls_all  == pid_idx]
        tgt_eta_sel = tgt_eta_all[tgt_cls_all == pid_idx]
        prd_eta_sel = prd_eta_all[prd_cls_all == pid_idx]

        # ── pT ──────────────────────────────────────────────────────────
        b_pt = np.logspace(-1, 3, 80)
        bc_pt = np.sqrt(b_pt[:-1] * b_pt[1:])

        fig_pt = plt.figure(figsize=(12, 8))
        gs_pt  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main  = plt.subplot(gs_pt[0])
        ax_ratio = plt.subplot(gs_pt[1])

        th, _ = np.histogram(tgt_pt_sel, bins=b_pt)
        mh, _ = np.histogram(prd_pt_sel, bins=b_pt)

        ax_main.hist(tgt_pt_sel, bins=b_pt, histtype="step", lw=2,
                     label=f"Target ({len(tgt_pt_sel)})", color=COLORS['target'])
        ax_main.hist(prd_pt_sel, bins=b_pt, histtype="step", lw=2,
                     label=f"MLPF ({len(prd_pt_sel)})", color=COLORS['mlpf'])
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_ylabel("Counts")
        ax_main.set_title(
            f"{pid_name}  $p_T$ (step {step},  {_cut_label()})")
        ax_main.legend()
        ax_main.grid(alpha=0.3)
        ax_main.tick_params(labelbottom=False)

        ratio, err = _safe_ratio(mh, th)
        ax_ratio.step(bc_pt, ratio, where='mid', color=COLORS['ratio'], lw=2)
        ax_ratio.fill_between(bc_pt, ratio - err, ratio + err,
                              alpha=0.3, color=COLORS['ratio'])
        ax_ratio.axhline(1.0, color=COLORS['ratio_hline'], ls='--', lw=1, alpha=0.7)
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("$p_T$ (GeV)")
        ax_ratio.set_ylabel("MLPF/Target")
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(alpha=0.3)

        out_pt = os.path.join(plots_dir,
                              f"particle_{pid_name}_pt_step{step}.png")
        fig_pt.savefig(out_pt, dpi=150, bbox_inches="tight")
        plt.close(fig_pt)

        # ── η ───────────────────────────────────────────────────────────
        b_eta = np.linspace(-4, 4, 80)
        bc_eta = (b_eta[:-1] + b_eta[1:]) / 2

        fig_eta = plt.figure(figsize=(12, 8))
        gs_eta  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main  = plt.subplot(gs_eta[0])
        ax_ratio = plt.subplot(gs_eta[1])

        th_e, _ = np.histogram(tgt_eta_sel, bins=b_eta)
        mh_e, _ = np.histogram(prd_eta_sel, bins=b_eta)

        ax_main.hist(tgt_eta_sel, bins=b_eta, histtype="step", lw=2,
                     label=f"Target ({len(tgt_eta_sel)})", color=COLORS['target'])
        ax_main.hist(prd_eta_sel, bins=b_eta, histtype="step", lw=2,
                     label=f"MLPF ({len(prd_eta_sel)})", color=COLORS['mlpf'])
        ax_main.set_ylabel("Counts")
        ax_main.set_title(
            f"{pid_name}  $\\eta$ (step {step},  {_cut_label()})")
        ax_main.legend()
        ax_main.grid(alpha=0.3)
        ax_main.tick_params(labelbottom=False)

        ratio_e, err_e = _safe_ratio(mh_e, th_e)
        ax_ratio.step(bc_eta, ratio_e, where='mid', color=COLORS['ratio'], lw=2)
        ax_ratio.fill_between(bc_eta, ratio_e - err_e, ratio_e + err_e,
                              alpha=0.3, color=COLORS['ratio'])
        ax_ratio.axhline(1.0, color=COLORS['ratio_hline'], ls='--', lw=1, alpha=0.7)
        ax_ratio.set_xlabel("$\\eta$")
        ax_ratio.set_ylabel("MLPF/Target")
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(alpha=0.3)

        out_eta = os.path.join(plots_dir,
                               f"particle_{pid_name}_eta_step{step}.png")
        fig_eta.savefig(out_eta, dpi=150, bbox_inches="tight")
        plt.close(fig_eta)

        print(f"  Saved: {out_pt}, {out_eta}")


# ── Efficiency / purity ───────────────────────────────────────────────────────
def plot_efficiency_purity(yvals, plots_dir, step):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    b  = np.logspace(-1, 3, 40)
    bc = np.sqrt(b[:-1] * b[1:])

    tgt_pt_all  = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_pt_all  = awkward.to_numpy(awkward.flatten(yvals["pred_pt"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))

    tgt_pt_all, tgt_cls_all = _apply_pt_cut(tgt_pt_all, tgt_cls_all)
    prd_pt_all, prd_cls_all = _apply_pt_cut(prd_pt_all, prd_cls_all)

    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue
        ax = axes[pid_idx]
        th, _ = np.histogram(tgt_pt_all[tgt_cls_all == pid_idx], bins=b)
        mh, _ = np.histogram(prd_pt_all[prd_cls_all == pid_idx], bins=b)
        eff = np.where(th > 0, mh / np.maximum(th, 1).astype(float), 0)
        ax.plot(bc, eff, "r-o", lw=2, markersize=3, label="pred/target ratio")
        ax.axhline(1.0, color="k", ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_ylim(0, 2)
        ax.set_xlabel("$p_T$ (GeV)")
        ax.set_ylabel("N_pred / N_target")
        ax.set_title(pid_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pred/Target ratio per particle type (step {step},  {_cut_label()})",
        fontsize=13)
    out = os.path.join(plots_dir, f"efficiency_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Energy response ───────────────────────────────────────────────────────────
def plot_energy_response(yvals, plots_dir, step):
    """Energy response — using original arrays with pT cut for selection."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    b = np.linspace(0, 3, 100)

    # Get all arrays (keep original lengths)
    tgt_pt_all = awkward.to_numpy(awkward.flatten(yvals["target_pt"]))
    tgt_e_all  = awkward.to_numpy(awkward.flatten(yvals["target_energy"]))
    tgt_cls_all = awkward.to_numpy(awkward.flatten(yvals["target_cls_id"]))
    prd_e_all  = awkward.to_numpy(awkward.flatten(yvals["pred_energy"]))
    prd_cls_all = awkward.to_numpy(awkward.flatten(yvals["pred_cls_id"]))
    cnd_e_all  = awkward.to_numpy(awkward.flatten(yvals["cand_energy"]))
    cnd_cls_all = awkward.to_numpy(awkward.flatten(yvals["cand_cls_id"]))

    # Apply pT cut only for selection masks
    tgt_pt_mask = tgt_pt_all > PT_CUT
    # Note: pred_pt and cand_pt are not available in the original first code
    # So we'll use the target pT cut for all

    for pid_idx, pid_name in enumerate(CLASS_NAMES):
        if pid_idx == 0:
            continue
        ax = axes[pid_idx]

        # Use target pT cut for all (like first code)
        msk_t = (tgt_cls_all == pid_idx) & (tgt_e_all > 0) & tgt_pt_mask
        msk_p = (prd_cls_all == pid_idx) & (prd_e_all > 0) & tgt_pt_mask
        msk_c = (cnd_cls_all == pid_idx) & (cnd_e_all > 0) & tgt_pt_mask

        # Now all masks are same length because they use the same base arrays
        msk = msk_t & msk_p
        if msk.sum() > 10:
            ratio_p = prd_e_all[msk] / tgt_e_all[msk]
            ratio_p = ratio_p[np.isfinite(ratio_p) & (ratio_p < 10)]
            ax.hist(ratio_p, bins=b, histtype="step", lw=2,
                    label=f"MLPF/Target ({len(ratio_p)})", color=COLORS['mlpf'])

        msk = msk_t & msk_c
        if msk.sum() > 10:
            ratio_c = cnd_e_all[msk] / tgt_e_all[msk]
            ratio_c = ratio_c[np.isfinite(ratio_c) & (ratio_c < 10)]
            ax.hist(ratio_c, bins=b, histtype="step", lw=2,
                    label=f"PF/Target ({len(ratio_c)})", color=COLORS['pf'])

        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("$E_{pred} / E_{target}$")
        ax.set_ylabel("Counts")
        ax.set_title(pid_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Energy response per particle (step {step},  {_cut_label()})",
        fontsize=13)
    out = os.path.join(plots_dir, f"energy_response_step{step}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    

# ── Legacy aliases ────────────────────────────────────────────────────────────
def plot_particle_pt(yvals, plots_dir, step):
    plot_particle_pt_with_ratio(yvals, plots_dir, step)

def plot_particle_eta(yvals, plots_dir, step):
    plot_particle_eta_with_ratio(yvals, plots_dir, step)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global PT_CUT

    exp_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/cms/data/store/user/moanwar/mlpf_data/experiments_ttbar_ptcut_v2/"
    step    = sys.argv[2] if len(sys.argv) > 2 else None
    PT_CUT  = float(sys.argv[3]) if len(sys.argv) > 3 else PT_CUT

    plots_dir = os.path.join(exp_dir, "validation_plots_v1")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nExperiment : {exp_dir}")
    print(f"Plots dir  : {plots_dir}")
    print(f"pT cut     : {_cut_label()}\n")

    print("1. Plotting loss curve...")
    plot_loss_curve(exp_dir, plots_dir)

    print("\n2. Loading predictions...")
    yvals, X, step = load_predictions(exp_dir, step)
    if yvals is None:
        return

    print("\n3. Particle class distribution...")
    plot_particle_cls(yvals, plots_dir, step)

    print("\n4. Overall pT and eta with ratio panels...")
    plot_particle_pt_with_ratio(yvals, plots_dir, step)
    plot_particle_eta_with_ratio(yvals, plots_dir, step)

    print("\n5. Stacked pT and eta plots...")
    plot_stacked_pt(yvals, plots_dir, step)
    plot_stacked_eta(yvals, plots_dir, step)
    plot_stacked_efficiency(yvals, plots_dir, step)

    print("\n6. Per-PID pT and eta with ratio panels...")
    plot_per_pid(yvals, plots_dir, step)

    print("\n7. Pred/target ratio per PID...")
    plot_efficiency_purity(yvals, plots_dir, step)

    print("\n8. Energy response per PID...")
    plot_energy_response(yvals, plots_dir, step)

    print(f"\n✅ All plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
