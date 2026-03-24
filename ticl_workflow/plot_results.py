#!/usr/bin/env python3
"""
Plot MLPF results from parquet predictions.
Usage: python3 plot_results.py <experiment_dir> [step]
"""
import sys, os
import numpy as np
import awkward
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow')
from mlpf.plotting.plot_utils import load_eval_data

def make_plots(exp_dir, step=10, sample="cms_pf_ticl_nopu"):
    pred_path  = Path(f"{exp_dir}/preds_step_{step}/{sample}/")
    plots_path = Path(f"{exp_dir}/plots_step_{step}/{sample}/")
    plots_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {pred_path}...")
    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), -1)
    print(f"Loaded {len(yvals['target_pt'])} events")

    class_names = ["none","ch.had","n.had","gamma","ele","mu"]

    # 1. Jet pT
    fig, ax = plt.subplots(figsize=(8,6))
    b = np.logspace(0, 3, 100)
    for key, label in [("jets_target_pt","Target"),("jets_cand_pt","PF"),
                       ("jets_pred_pt","MLPF"),("jets_gen_pt","Truth")]:
        pt = awkward.to_numpy(awkward.flatten(yvals[key]))
        pt = pt[pt > 0]
        if len(pt): ax.hist(pt, bins=b, histtype="step", lw=2, label=f"{label} ({len(pt)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Jet pT (GeV)"); ax.set_ylabel("Jets / bin")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Jet pT")
    fig.savefig(str(plots_path/"jet_pt.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  jet_pt.png")

    # 2. Jet ratio
    fig, ax = plt.subplots(figsize=(8,6))
    b = np.linspace(0, 3, 100)
    for key, label in [("jet_ratio_gen_to_target_pt","gen/target"),
                       ("jet_ratio_gen_to_cand_pt","gen/PF"),
                       ("jet_ratio_gen_to_pred_pt","gen/MLPF")]:
        r = yvals[key]
        if len(r) > 0:
            ax.hist(r, bins=b, histtype="step", lw=2, label=f"{label} ({len(r)})")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("Jet pT ratio"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Jet pT ratio")
    fig.savefig(str(plots_path/"jet_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  jet_ratio.png")

    # 3. Particle pT
    fig, ax = plt.subplots(figsize=(8,6))
    b = np.logspace(-1, 3, 100)
    for key, label in [("target_pt","Target"),("pred_pt","MLPF"),("cand_pt","PF")]:
        pt = awkward.to_numpy(awkward.flatten(yvals[key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("pt","cls_id")]))
        pt = pt[cls != 0]
        if len(pt): ax.hist(pt, bins=b, histtype="step", lw=2, label=f"{label} ({len(pt)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Particle pT (GeV)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Particle pT")
    fig.savefig(str(plots_path/"particle_pt.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  particle_pt.png")

    # 4. Particle eta
    fig, ax = plt.subplots(figsize=(8,6))
    b = np.linspace(-5, 5, 100)
    for key, label in [("target_eta","Target"),("pred_eta","MLPF"),("cand_eta","PF")]:
        eta = awkward.to_numpy(awkward.flatten(yvals[key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("eta","cls_id")]))
        eta = eta[cls != 0]
        if len(eta): ax.hist(eta, bins=b, histtype="step", lw=2, label=f"{label} ({len(eta)})")
    ax.set_xlabel("Particle eta"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Particle eta")
    fig.savefig(str(plots_path/"particle_eta.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  particle_eta.png")

    # 5. Particle class ID
    fig, ax = plt.subplots(figsize=(8,6))
    for key, label in [("target_cls_id","Target"),("pred_cls_id","MLPF"),("cand_cls_id","PF")]:
        cls = awkward.to_numpy(awkward.flatten(yvals[key]))
        ax.hist(cls, bins=np.arange(-0.5,6.5,1), histtype="step", lw=2, label=label)
    ax.set_xticks(range(6)); ax.set_xticklabels(class_names, rotation=45)
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Particle class ID")
    fig.savefig(str(plots_path/"particle_cls.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  particle_cls.png")

    # 6. Num elements
    fig, ax = plt.subplots(figsize=(8,6))
    n = awkward.to_numpy(awkward.sum(X[:,:,0] != 0, axis=-1))
    ax.hist(n, bins=50, histtype="step", lw=2)
    ax.set_xlabel("Elements / event"); ax.grid(alpha=0.3)
    ax.set_title("Number of input elements")
    fig.savefig(str(plots_path/"num_elements.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  num_elements.png")

    # 7. Particle energy
    fig, ax = plt.subplots(figsize=(8,6))
    b = np.logspace(-1, 3, 100)
    for key, label in [("target_energy","Target"),("pred_energy","MLPF"),("cand_energy","PF")]:
        e = awkward.to_numpy(awkward.flatten(yvals[key]))
        cls = awkward.to_numpy(awkward.flatten(yvals[key.replace("energy","cls_id")]))
        e = e[cls != 0]
        if len(e): ax.hist(e, bins=b, histtype="step", lw=2, label=f"{label} ({len(e)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Particle energy (GeV)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Particle energy")
    fig.savefig(str(plots_path/"particle_energy.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  particle_energy.png")

    print(f"\nAll plots saved to {plots_path}")

if __name__ == "__main__":
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/MLPF_ticl_test_my_training_20260322_230318_792799"
    step    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    make_plots(exp_dir, step)
