import uproot
import numpy as np
import networkx as nx
import argparse
import os
import math
import pickle
from tqdm import tqdm
from collections import defaultdict
import fastjet
import traceback
import awkward
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

"""
cd /afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/mlpf/data/cms/

not use the SuperClusteringDNN :

python3 postprocessing_ticl.py \
    --input prtg_mix_0pu_oneEtaSide_1M.txt \
    --output /eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/pikl_files_v2/ticl_graph_data_prt.pkl \
    --events-per-pkl 20000

using the SuperClusteringDNN :

python3 postprocessing_ticl.py \
    --input prtg_mix_0pu_oneEtaSide_1M.txt \
    --output /eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/pikl_files_v2/ticl_graph_data_prt.pkl \
    --events-per-pkl 20000 \
    --use-superclustering

quick test :
 python3 postprocessing_ticl.py \
    --input prtg_mix_0pu_oneEtaSide_1M.txt \
    --output tmp/test_no_sc2.pkl \
    --num-events 100 --max-files 5

"""
# python3 postprocessing_ticl.py --input 211_0pu.txt --output ticl_graph_data_pion_0pu.pkl

# Prevent threading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Feature definitions
#elem_branches = [
#    "typ", "pt", "eta", "phi", "energy", "layer", "charge",
#    "px", "py", "pz",
#    "sigma_x", "sigma_y", "sigma_z", "deltap", "sigmadeltap",
#    "num_hits", "cluster_flags", "corr_energy", "corr_energy_err",
#    "vx", "vy", "vz", "pterror", "etaerror", "phierror", "lambd", "lambdaerror",
#    "theta", "thetaerror", "time", "timeerror", "etaerror1", "etaerror2",
#]

elem_branches = [
    "typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz",
    "em_energy", "bary_z", "nhits",
    "min_dR_track",    # min dR to nearest good track at HGCAL
    "near_track_pt",   # pT of nearest track (0 if none)
    "shower_depth",    # energy-weighted shower depth Σ(|z|×E)/Σ(E)
    "sum_pt_dR10",     # NEW: sum of track pT within dR<0.10
]

particle_feature_order = [
    "pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
    "ispu", "simulatorStatus", "cp_to_track", "cp_to_cluster", "jet_idx"
]

#particle_feature_order = [
#    "pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
#    "ispu", "generatorStatus", "simulatorStatus", "cp_to_track",
#    "cp_to_cluster", "jet_idx"
#]

# Pre-build lookup sets/dicts once at module level
_NEUTRAL_PIDS   = frozenset([130, 22, 310])
_CHARGED_PIDS   = frozenset([11, 13, 211, 321])
_NEUTRINO_PIDS  = frozenset([12, 14, 16])


def get_charge(pid):
    abs_pid = abs(pid)
    if pid in _NEUTRAL_PIDS:
        return 0.0
    if abs_pid in _CHARGED_PIDS:
        return -math.copysign(1.0, pid)
    return 0.0


def is_charged_particle(pid):
    return abs(pid) in _CHARGED_PIDS


# Jet / MET
def _make_pseudojets(pts, etas, phis, energies):
    """Vectorised PseudoJet construction."""
    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    safe_eta = np.where(np.abs(etas) < 10, etas, 0.0)
    pz = pts * np.sinh(safe_eta)
    return [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(energies[i]))
            for i in range(len(pts))]


_JET_DEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


def compute_truth_jets(g, pt_min=3.0):
    gen_nodes = [n for n in g.nodes if n[0] == "gen"]
    if not gen_nodes:
        return np.array([])
    
    stable = [n for n in gen_nodes
              if g.nodes[n].get("status", 0) == 1
              and g.nodes[n].get("num_daughters", 0) == 0
              and g.nodes[n].get("pid", 0) not in _NEUTRINO_PIDS]

    if not stable:
        return np.array([])

    attrs    = g.nodes
    pts      = np.array([attrs[n]["pt"]     for n in stable], dtype=np.float64)
    etas     = np.array([attrs[n]["eta"]    for n in stable], dtype=np.float64)
    phis     = np.array([attrs[n]["phi"]    for n in stable], dtype=np.float64)
    energies = np.array([attrs[n]["energy"] for n in stable], dtype=np.float64)

    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)
        return np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets]) if jets else np.array([])
    except Exception as e:
        print(f"Warning: truth jet clustering failed: {e}")
        return np.array([])


def compute_target_jets(ytarget, pt_min=3.0):
    valid = ytarget["pid"] != 0
    if not np.any(valid):
        return np.zeros((0, 4), dtype=np.float32)

    pts      = ytarget["pt"][valid].astype(np.float64)
    etas     = ytarget["eta"][valid].astype(np.float64)
    phis     = np.arctan2(ytarget["sin_phi"][valid], ytarget["cos_phi"][valid]).astype(np.float64)
    energies = ytarget["energy"][valid].astype(np.float64)

    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)
        return np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets], dtype=np.float32) \
               if jets else np.zeros((0, 4), dtype=np.float32)
    except Exception as e:
        print(f"Warning: target jet clustering failed: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def compute_genmet(g):
    gen_nodes = [n for n in g.nodes if n[0] == "gen"]
    attrs = g.nodes
    neutrinos = [n for n in gen_nodes
                 if attrs[n].get("status", 0) == 1
                 and attrs[n].get("pid", 0) in _NEUTRINO_PIDS
                 and attrs[n].get("num_daughters", 0) == 0]
    if not neutrinos:
        return np.array([0.0, 0.0])

    pts  = np.array([attrs[n]["pt"]  for n in neutrinos])
    phis = np.array([attrs[n]["phi"] for n in neutrinos])
    sum_px = np.sum(pts * np.cos(phis))
    sum_py = np.sum(pts * np.sin(phis))
    return np.array([math.hypot(sum_px, sum_py), math.atan2(sum_py, sum_px)])

def compute_genmetv(g):
    """
    Compute generator-level MET as the vector sum of neutrino pT.
    """
    gen_nodes = [n for n in g.nodes if n[0] == "gen"]
    attrs = g.nodes
    neutrino_nodes = [n for n in gen_nodes if attrs[n].get("pid", 0) in _NEUTRINO_PIDS]
    if neutrino_nodes:
        print(f"Neutrino statuses: {[(n, attrs[n].get('status',0), attrs[n].get('pt',0)) for n in neutrino_nodes[:6]]}")

    # Select stable neutrinos these carry away the missing energy
    neutrinos = [n for n in gen_nodes
                 if attrs[n].get("status", 0) == 1
                 and attrs[n].get("pid", 0) in _NEUTRINO_PIDS]
    
    if not neutrinos:
        return np.array([0.0, 0.0])

    pts  = np.array([attrs[n]["pt"]  for n in neutrinos])
    phis = np.array([attrs[n]["phi"] for n in neutrinos])
    sum_px = np.sum(pts * np.cos(phis))
    sum_py = np.sum(pts * np.sin(phis))
    return np.array([math.hypot(sum_px, sum_py), math.atan2(sum_py, sum_px)])


def assign_jet_indices(ytarget, targetjets, pt_min=3.0):
    ytarget_copy = ytarget.copy()
    ytarget_copy["jet_idx"] = -1

    valid = ytarget["pid"] != 0
    if not np.any(valid) or len(targetjets) == 0:
        return ytarget_copy

    pts          = ytarget["pt"][valid].astype(np.float64)
    etas         = ytarget["eta"][valid].astype(np.float64)
    phis         = np.arctan2(ytarget["sin_phi"][valid], ytarget["cos_phi"][valid]).astype(np.float64)
    energies     = ytarget["energy"][valid].astype(np.float64)
    valid_indices = np.where(valid)[0]

    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)

        if hasattr(cluster, 'constituent_index'):
            indices = cluster.constituent_index(ptmin=pt_min)
        else:
            indices = []
            for jet in jets:
                consts     = jet.constituents()
                jet_indices = [i for i, p in enumerate(pjs) if any(
                    abs(p.pt() - c.pt()) < 1e-6 for c in consts)]
                indices.append(jet_indices)

        for jet_idx, const_indices in enumerate(indices):
            for ci in const_indices:
                if ci < len(valid_indices):
                    ytarget_copy["jet_idx"][valid_indices[ci]] = jet_idx
    except Exception as e:
        print(f"Warning: jet index assignment failed: {e}")

    return ytarget_copy


# Event reading
def read_event(trees, iev):
    ev = {}
    sl = dict(entry_start=iev, entry_stop=iev + 1, library="np")

    a = trees['tkst'].arrays(["raw_energy", "barycenter_eta", "barycenter_phi", "raw_pt",
                              "raw_em_energy", "barycenter_z",
                              "vertices_z", "vertices_energy"], **sl)
    ev["ts_energy"]    = a["raw_energy"][0];    ev["ts_pt"]       = a["raw_pt"][0]
    ev["ts_eta"]       = a["barycenter_eta"][0]; ev["ts_phi"]      = a["barycenter_phi"][0]
    ev["ts_z"]         = a["barycenter_z"][0];  ev["ts_em_energy"]= a["raw_em_energy"][0]
    ev["ts_vertices_z"]= a["vertices_z"][0];    ev["ts_vertices_e"]= a["vertices_energy"][0]

    a = trees['tkstEG'].arrays(["raw_energy", "barycenter_eta", "barycenter_phi", "raw_pt", "raw_em_energy","barycenter_z"], **sl)
    ev["tsEG_energy"] = a["raw_energy"][0];  ev["tsEG_pt"]  = a["raw_pt"][0]
    ev["tsEG_eta"]    = a["barycenter_eta"][0]; ev["tsEG_phi"] = a["barycenter_phi"][0]
    ev["tsEG_z"]    = a["barycenter_z"][0]; ev["tsEG_em_energy"] = a["raw_em_energy"][0]

    a = trees['simtkst'].arrays(
        ["regressed_energy", "barycenter_eta", "barycenter_phi", "pdgID", "trackIdx", "CPidx"], **sl)
    ev["simtkst_energy"]   = a["regressed_energy"][0]
    ev["simtkst_eta"]      = a["barycenter_eta"][0]
    ev["simtkst_phi"]      = a["barycenter_phi"][0]
    ev["simtkst_pdgid"]    = a["pdgID"][0]
    ev["simtkst_trackIdx"] = a["trackIdx"][0]
    ev["simtkst_CPidx"]    = a["CPidx"][0]

    a = trees['cand'].arrays(
        ["candidate_pt", "candidate_eta", "candidate_phi", "candidate_energy",
         "trackstersLinks_in_candidate", "candidate_pdgId", "track_in_candidate"], **sl)
    ev["tcan_pt"]         = a["candidate_pt"][0];     ev["tcan_eta"]    = a["candidate_eta"][0]
    ev["tcan_phi"]        = a["candidate_phi"][0];    ev["tcan_energy"] = a["candidate_energy"][0]
    ev["trkst_indcies"]   = a["trackstersLinks_in_candidate"][0]
    ev["trks_indcies"]    = a["track_in_candidate"][0]
    ev["candidate_pdgId"] = a["candidate_pdgId"][0]

    a = trees['simcan'].arrays(
        ["simTICLCandidate_pdgId", "simTICLCandidate_pt", "simTICLCandidate_eta",
         "simTICLCandidate_phi", "simTICLCandidate_regressed_energy",
         "simTICLCandidate_raw_energy",
         "simTICLCandidate_tracks_in_candidate", "simTICLCandidate_simTracksterCPIndex",
         "simTICLCandidate_ispu"], **sl)
    ev["simcan_pdgid"]               = a["simTICLCandidate_pdgId"][0]
    ev["simcan_pt"]                  = a["simTICLCandidate_pt"][0]
    ev["simcan_eta"]                 = a["simTICLCandidate_eta"][0]
    ev["simcan_phi"]                 = a["simTICLCandidate_phi"][0]
    ev["simcan_reg_energy"]          = a["simTICLCandidate_regressed_energy"][0]
    ev["simcan_raw_energy"]          = a["simTICLCandidate_raw_energy"][0]
    ev["simcan_trkId"]               = a["simTICLCandidate_tracks_in_candidate"][0]
    ev["simcan_simTracksterCPIndex"] = a["simTICLCandidate_simTracksterCPIndex"][0]
    ev["simcan_ispu"]                = a["simTICLCandidate_ispu"][0]

    a = trees['genpar'].arrays(
        ["GenPart_status", "GenPart_genPartIdxMother", "GenPart_eta",
         "GenPart_phi", "GenPart_pdgId", "GenPart_mass", "GenPart_pt", "GenPart_energy"], **sl)
    ev["genpar_pdgid"]            = a["GenPart_pdgId"][0]
    ev["genpar_mass"]             = a["GenPart_mass"][0]
    ev["genpar_eta"]              = a["GenPart_eta"][0]
    ev["genpar_phi"]              = a["GenPart_phi"][0]
    ev["genpar_status"]           = a["GenPart_status"][0]
    ev["genpar_genPartIdxMother"] = a["GenPart_genPartIdxMother"][0]
    ev["genpar_pt"]               = a["GenPart_pt"][0]
    ev["genpar_energy"]           = a["GenPart_energy"][0]

    a = trees['track'].arrays(
        ["track_pt", "track_p", "track_eta", "track_hgcal_phi", "track_hgcal_eta",
         "track_charge", "track_id", "track_missing_outer_hits", 
         "track_quality", "track_nhits"], **sl)
    ev["track_pt"]       = a["track_pt"][0];   ev["track_p"]       = a["track_p"][0]
    ev["track_eta"]      = a["track_eta"][0];  ev["track_phi"]     = a["track_hgcal_phi"][0]
    ev["track_hgcal_eta"]= a["track_hgcal_eta"][0]
    ev["track_hgcal_phi"]= a["track_hgcal_phi"][0]
    ev["track_charge"]   = a["track_charge"][0]; ev["track_id"]    = a["track_id"][0]
    ev["track_hits"]     = a["track_missing_outer_hits"][0]
    ev["track_quality"]  = a["track_quality"][0]
    ev["track_nhits"]    = a["track_nhits"][0]
    
    a = trees['assoc'].arrays(
        ["ticlTracksterLinks_recoToSim_CP_score", "ticlTracksterLinks_simToReco_CP_score",
         "ticlTracksterLinks_simToReco_CP_sharedE", "ticlTracksterLinks_recoToSim_CP",
         "ticlTracksterLinks_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"], **sl)
    ev["ticlTracksterLinks_recoToSim_CP_score"]   = a["ticlTracksterLinks_recoToSim_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_score"]   = a["ticlTracksterLinks_simToReco_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_sharedE"] = a["ticlTracksterLinks_simToReco_CP_sharedE"][0]
    ev["ticlTracksterLinks_recoToSim_CP"]         = a["ticlTracksterLinks_recoToSim_CP"][0]
    ev["ticlTracksterLinks_simToReco_CP"]         = a["ticlTracksterLinks_simToReco_CP"][0]
    ev["ticlTracksterLinksDNN_recoToSim_CP_score"]   = a["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP_score"]   = a["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"] = a["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE"][0]
    ev["ticlTracksterLinksDNN_recoToSim_CP"]         = a["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP"]         = a["ticlTracksterLinksSuperclusteringDNN_simToReco_CP"][0]

    return ev


def read_all_events(trees, start_event=0, num_events=-1):
    """
    Read ALL events at once per file — much faster than per-event reading.
    Returns a list of event dicts, same format as read_event().
    """
    total = trees['tkst'].num_entries
    if num_events == -1:
        num_events = total - start_event
    else:
        num_events = min(num_events, total - start_event)
    
    sl = dict(entry_start=start_event, 
              entry_stop=start_event + num_events, 
              library="np")

    # Read ALL events in one shot per tree
    a_tkst = trees['tkst'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", 
         "raw_pt", "raw_em_energy", "barycenter_z",
         "vertices_z", "vertices_energy"], **sl)
    
    a_tkstEG = trees['tkstEG'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", 
         "raw_pt", "raw_em_energy", "barycenter_z"], **sl)
    
    a_simtkst = trees['simtkst'].arrays(
        ["regressed_energy", "barycenter_eta", "barycenter_phi", 
         "pdgID", "trackIdx", "CPidx"], **sl)
    
    a_cand = trees['cand'].arrays(
        ["candidate_pt", "candidate_eta", "candidate_phi", 
         "candidate_energy", "trackstersLinks_in_candidate", 
         "candidate_pdgId", "track_in_candidate"], **sl)
    
    a_simcan = trees['simcan'].arrays(
        ["simTICLCandidate_pdgId", "simTICLCandidate_pt",
         "simTICLCandidate_eta", "simTICLCandidate_phi",
         "simTICLCandidate_regressed_energy", "simTICLCandidate_raw_energy",
         "simTICLCandidate_tracks_in_candidate",
         "simTICLCandidate_simTracksterCPIndex",
         "simTICLCandidate_ispu"], **sl)
    
    a_genpar = trees['genpar'].arrays(
        ["GenPart_status", "GenPart_genPartIdxMother", "GenPart_eta",
         "GenPart_phi", "GenPart_pdgId", "GenPart_mass", 
         "GenPart_pt", "GenPart_energy"], **sl)
    
    a_track = trees['track'].arrays(
        ["track_pt", "track_p", "track_eta", "track_hgcal_phi", "track_hgcal_eta",
         "track_charge", "track_id", "track_missing_outer_hits",
         "track_quality", "track_nhits"], **sl)
    a_assoc = trees['assoc'].arrays(
        ["ticlTracksterLinks_recoToSim_CP_score",
         "ticlTracksterLinks_simToReco_CP_score",
         "ticlTracksterLinks_simToReco_CP_sharedE",
         "ticlTracksterLinks_recoToSim_CP",
         "ticlTracksterLinks_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"], **sl)

    # Build per-event dicts
    events = []
    for i in range(num_events):
        ev = {}
        ev["ts_energy"]      = a_tkst["raw_energy"][i]
        ev["ts_pt"]          = a_tkst["raw_pt"][i]
        ev["ts_eta"]         = a_tkst["barycenter_eta"][i]
        ev["ts_phi"]         = a_tkst["barycenter_phi"][i]
        ev["ts_z"]           = a_tkst["barycenter_z"][i]
        ev["ts_vertices_z"]  = a_tkst["vertices_z"][i]
        ev["ts_vertices_e"]  = a_tkst["vertices_energy"][i]
        ev["ts_em_energy"]   = a_tkst["raw_em_energy"][i]
        ev["ts_vertices_z"]  = a_tkst["vertices_z"][i]
        ev["ts_vertices_e"]  = a_tkst["vertices_energy"][i]

        ev["tsEG_energy"]    = a_tkstEG["raw_energy"][i]
        ev["tsEG_pt"]        = a_tkstEG["raw_pt"][i]
        ev["tsEG_eta"]       = a_tkstEG["barycenter_eta"][i]
        ev["tsEG_phi"]       = a_tkstEG["barycenter_phi"][i]
        ev["tsEG_z"]         = a_tkstEG["barycenter_z"][i]
        ev["tsEG_em_energy"] = a_tkstEG["raw_em_energy"][i]

        ev["simtkst_energy"]   = a_simtkst["regressed_energy"][i]
        ev["simtkst_eta"]      = a_simtkst["barycenter_eta"][i]
        ev["simtkst_phi"]      = a_simtkst["barycenter_phi"][i]
        ev["simtkst_pdgid"]    = a_simtkst["pdgID"][i]
        ev["simtkst_trackIdx"] = a_simtkst["trackIdx"][i]
        ev["simtkst_CPidx"]    = a_simtkst["CPidx"][i]

        ev["tcan_pt"]         = a_cand["candidate_pt"][i]
        ev["tcan_eta"]        = a_cand["candidate_eta"][i]
        ev["tcan_phi"]        = a_cand["candidate_phi"][i]
        ev["tcan_energy"]     = a_cand["candidate_energy"][i]
        ev["trkst_indcies"]   = a_cand["trackstersLinks_in_candidate"][i]
        ev["trks_indcies"]    = a_cand["track_in_candidate"][i]
        ev["candidate_pdgId"] = a_cand["candidate_pdgId"][i]

        ev["simcan_pdgid"]               = a_simcan["simTICLCandidate_pdgId"][i]
        ev["simcan_pt"]                  = a_simcan["simTICLCandidate_pt"][i]
        ev["simcan_eta"]                 = a_simcan["simTICLCandidate_eta"][i]
        ev["simcan_phi"]                 = a_simcan["simTICLCandidate_phi"][i]
        ev["simcan_reg_energy"]          = a_simcan["simTICLCandidate_regressed_energy"][i]
        ev["simcan_raw_energy"]          = a_simcan["simTICLCandidate_raw_energy"][i]
        ev["simcan_trkId"]               = a_simcan["simTICLCandidate_tracks_in_candidate"][i]
        ev["simcan_simTracksterCPIndex"] = a_simcan["simTICLCandidate_simTracksterCPIndex"][i]
        ev["simcan_ispu"]                = a_simcan["simTICLCandidate_ispu"][i]

        ev["genpar_pdgid"]            = a_genpar["GenPart_pdgId"][i]
        ev["genpar_mass"]             = a_genpar["GenPart_mass"][i]
        ev["genpar_eta"]              = a_genpar["GenPart_eta"][i]
        ev["genpar_phi"]              = a_genpar["GenPart_phi"][i]
        ev["genpar_status"]           = a_genpar["GenPart_status"][i]
        ev["genpar_genPartIdxMother"] = a_genpar["GenPart_genPartIdxMother"][i]
        ev["genpar_pt"]               = a_genpar["GenPart_pt"][i]
        ev["genpar_energy"]           = a_genpar["GenPart_energy"][i]

        ev["track_pt"]      = a_track["track_pt"][i]
        ev["track_p"]       = a_track["track_p"][i]
        ev["track_eta"]     = a_track["track_eta"][i]
        ev["track_phi"]     = a_track["track_hgcal_phi"][i]
        ev["track_charge"]  = a_track["track_charge"][i]
        ev["track_id"]      = a_track["track_id"][i]
        ev["track_hits"]    = a_track["track_missing_outer_hits"][i]
        ev["track_quality"] = a_track["track_quality"][i]
        ev["track_nhits"]   = a_track["track_nhits"][i]
        ev["track_hgcal_eta"] = a_track["track_hgcal_eta"][i]
        ev["track_hgcal_phi"] = a_track["track_hgcal_phi"][i]

        ev["ticlTracksterLinks_recoToSim_CP_score"]   = a_assoc["ticlTracksterLinks_recoToSim_CP_score"][i]
        ev["ticlTracksterLinks_simToReco_CP_score"]   = a_assoc["ticlTracksterLinks_simToReco_CP_score"][i]
        ev["ticlTracksterLinks_simToReco_CP_sharedE"] = a_assoc["ticlTracksterLinks_simToReco_CP_sharedE"][i]
        ev["ticlTracksterLinks_recoToSim_CP"]         = a_assoc["ticlTracksterLinks_recoToSim_CP"][i]
        ev["ticlTracksterLinks_simToReco_CP"]         = a_assoc["ticlTracksterLinks_simToReco_CP"][i]
        ev["ticlTracksterLinksDNN_recoToSim_CP_score"]   = a_assoc["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"][i]
        ev["ticlTracksterLinksDNN_simToReco_CP_score"]   = a_assoc["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score"][i]
        ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"] = a_assoc["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE"][i]
        ev["ticlTracksterLinksDNN_recoToSim_CP"]         = a_assoc["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP"][i]
        ev["ticlTracksterLinksDNN_simToReco_CP"]         = a_assoc["ticlTracksterLinksSuperclusteringDNN_simToReco_CP"][i]

        events.append(ev)
    return events, num_events


# Connection collectors
def collect_hadronic_connections(ev):
    connections = []
    r2s_scores  = ev["ticlTracksterLinks_recoToSim_CP_score"]
    s2r_scores  = ev["ticlTracksterLinks_simToReco_CP_score"]
    s2r_sharedE = ev["ticlTracksterLinks_simToReco_CP_sharedE"]
    s2r_index   = ev["ticlTracksterLinks_simToReco_CP"]

    simcan_energy = ev["simcan_reg_energy"]
    simcan_pdgid  = ev["simcan_pdgid"]
    simcan_eta    = ev["simcan_eta"]
    ts_energy     = ev["ts_energy"]
    '''
    for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
        if len(shared_arr) == 0:
            continue

        cp_energy = simcan_energy[sim_idx]
        cp_pid    = simcan_pdgid[sim_idx]

        if abs(cp_pid) in [11, 22]:
            continue

        cp_eta = simcan_eta[sim_idx]

        best_candidate = None
        best_s_score = float("inf")
        best_r_score = float("inf")

        for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
            if shared_energy <= 0:
                continue

            r_score = (
                r2s_scores[trackster_idx][0]
                if trackster_idx < len(r2s_scores) and len(r2s_scores[trackster_idx]) > 0
                else 1.0
            )

            s_score = (
                s2r_scores[sim_idx][idx2]
                if sim_idx < len(s2r_scores) and idx2 < len(s2r_scores[sim_idx])
                else 1.0
            )
            
            # Keep cuts ONLY for charged
            #if cp_pid in [211, 321]:  # charged hadrons
            #    if r_score > 0.6 or s_score > 0.9:
            #        continue

            if (s_score < best_s_score) or (
                    s_score == best_s_score and r_score < best_r_score
            ):
                best_s_score = s_score
                best_r_score = r_score
                best_candidate = (trackster_idx, shared_energy)

        # Keep only best match
        if best_candidate is not None:
            trackster_idx, shared_energy = best_candidate

            connections.append({
                'cp_idx': sim_idx,
                'cp_pid': cp_pid,
                'cp_energy': cp_energy,
                'cp_eta': cp_eta,
                'element_idx': trackster_idx,
                'element_type': 4,
                'shared_energy': ts_energy[trackster_idx],
                'element_energy': ts_energy[trackster_idx],
                'is_charged': is_charged_particle(cp_pid)
            })
            
    return connections
    '''
    for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
        if len(shared_arr) == 0:
            continue
        cp_energy = simcan_energy[sim_idx]
        cp_pid    = simcan_pdgid[sim_idx]
        if abs(cp_pid) == 11 or abs(cp_pid) == 22:
            continue
        cp_eta    = simcan_eta[sim_idx]
        for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
            if shared_energy <= 0:
                continue
            r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores) and len(r2s_scores[trackster_idx]) > 0 else 1.0
            s_score = s2r_scores[sim_idx][idx2]    if sim_idx < len(s2r_scores) and idx2 < len(s2r_scores[sim_idx]) else 1.0
            if cp_pid in [211, 321]:  # charged hadrons                                                                        
                if r_score > 0.6 or s_score > 0.9:
                    continue
            elif cp_pid in [130, 310]:  # neutral hadrons                                                                      
                if r_score > 0.6 or s_score > 0.9:
                    continue
            connections.append({
                'cp_idx': sim_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                'cp_eta': cp_eta, 'element_idx': trackster_idx, 'element_type': 4,
                'shared_energy': ts_energy[trackster_idx], 'element_energy': ts_energy[trackster_idx],
                'is_charged': is_charged_particle(cp_pid)
            })
    return connections


def collect_em_connections(ev, use_superclustering=False):
    """
    Collect EM trackster connections.
    
    use_superclustering=True:  use SuperclusteringDNN (better EM quality)
                               but creates duplication with ticlTracksterLinks
    use_superclustering=False: use ticlTracksterLinks for EM as well
                               (type 2 assigned via CP matching with EM particles)
                               no duplication, consistent with HAD tracksters
    """
    connections = []
    n_ts     = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])

    if use_superclustering:
        # Use SuperclusteringDNN tracksters (tsEG)
        r2s_scores  = ev["ticlTracksterLinksDNN_recoToSim_CP_score"]
        s2r_scores  = ev["ticlTracksterLinksDNN_simToReco_CP_score"]
        s2r_sharedE = ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"]
        s2r_index   = ev["ticlTracksterLinksDNN_simToReco_CP"]

        if s2r_sharedE is None or len(s2r_sharedE) == 0:
            return connections

        simcan_pdgid      = ev["simcan_pdgid"]
        simcan_raw_energy = ev["simcan_raw_energy"]
        simcan_reg_energy = ev["simcan_reg_energy"]
        simcan_eta        = ev["simcan_eta"]
        tsEG_energy       = ev["tsEG_energy"]

        for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
            if len(shared_arr) == 0:
                continue
            cp_pid  = simcan_pdgid[sim_idx]
            abs_pid = abs(cp_pid)
            if abs_pid not in (11, 22):
                continue
            cp_eta    = simcan_eta[sim_idx]
            cp_energy = simcan_raw_energy[sim_idx] if abs_pid == 11                         else simcan_reg_energy[sim_idx]

            for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
                if shared_energy <= 0:
                    continue
                r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores)                           and len(r2s_scores[trackster_idx]) > 0 else 1.0
                s_score = s2r_scores[sim_idx][idx2] if sim_idx < len(s2r_scores)                           and idx2 < len(s2r_scores[sim_idx]) else 1.0
                if r_score > 0.6 or s_score > 0.9:
                    continue
                connections.append({
                    'cp_idx':        sim_idx,
                    'cp_pid':        cp_pid,
                    'cp_energy':     cp_energy,
                    'cp_eta':        cp_eta,
                    'element_idx':   n_ts + n_tracks + trackster_idx,
                    'element_type':  2,
                    'shared_energy': tsEG_energy[trackster_idx],
                    'element_energy':tsEG_energy[trackster_idx],
                    'is_charged':    is_charged_particle(cp_pid)
                })

    else:
        # Use ticlTracksterLinks for EM as well (no duplication)
        # Type 2 assigned via CP matching — same tracksters as HAD but
        # those matched to EM CPs (ele/gamma) get type 2
        r2s_scores  = ev["ticlTracksterLinks_recoToSim_CP_score"]
        s2r_scores  = ev["ticlTracksterLinks_simToReco_CP_score"]
        s2r_sharedE = ev["ticlTracksterLinks_simToReco_CP_sharedE"]
        s2r_index   = ev["ticlTracksterLinks_simToReco_CP"]

        simcan_pdgid      = ev["simcan_pdgid"]
        simcan_raw_energy = ev["simcan_raw_energy"]
        simcan_reg_energy = ev["simcan_reg_energy"]
        simcan_eta        = ev["simcan_eta"]
        ts_energy         = ev["ts_energy"]

        for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
            if len(shared_arr) == 0:
                continue
            cp_pid  = simcan_pdgid[sim_idx]
            abs_pid = abs(cp_pid)
            # Only EM particles (ele/gamma) → type 2
            if abs_pid not in (11, 22):
                continue
            cp_eta    = simcan_eta[sim_idx]
            cp_energy = simcan_raw_energy[sim_idx] if abs_pid == 11                         else simcan_reg_energy[sim_idx]

            for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
                if shared_energy <= 0:
                    continue
                r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores)                           and len(r2s_scores[trackster_idx]) > 0 else 1.0
                s_score = s2r_scores[sim_idx][idx2] if sim_idx < len(s2r_scores)                           and idx2 < len(s2r_scores[sim_idx]) else 1.0
                if r_score > 0.6 or s_score > 0.9:
                    continue
                # element_idx is within ticlTracksterLinks (n_ts space)
                # type 2 because matched to EM CP
                connections.append({
                    'cp_idx':        sim_idx,
                    'cp_pid':        cp_pid,
                    'cp_energy':     cp_energy,
                    'cp_eta':        cp_eta,
                    'element_idx':   trackster_idx,  # in ts space, NOT tsEG
                    'element_type':  2,              # type 2!
                    'shared_energy': ts_energy[trackster_idx],
                    'element_energy':ts_energy[trackster_idx],
                    'is_charged':    is_charged_particle(cp_pid)
                })

    return connections


def collect_track_connections(ev):
    connections = []
    n_ts = len(ev["ts_energy"])
    track_id_to_idx = {tid: i for i, tid in enumerate(ev["track_id"])}

    simcan_trkId      = ev["simcan_trkId"]
    simcan_pdgid      = ev["simcan_pdgid"]
    simcan_reg_energy = ev["simcan_reg_energy"]
    simcan_raw_energy = ev["simcan_raw_energy"]
    simcan_eta        = ev["simcan_eta"]
    track_p           = ev["track_p"]

    for cp_idx, track_indices in enumerate(simcan_trkId):
        if len(track_indices) == 0:
            continue
        cp_pid      = simcan_pdgid[cp_idx]
        cp_energy = simcan_raw_energy[cp_idx] if abs(cp_pid) == 11 else simcan_reg_energy[cp_idx]
        cp_eta      = simcan_eta[cp_idx]
        element_type = 1  # all tracks are type 1        
        for track_id in track_indices:
            track_idx = track_id_to_idx.get(track_id)
            if track_idx is None:
                continue
            tp = track_p[track_idx]
            connections.append({
                'cp_idx': cp_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                'cp_eta': cp_eta, 'element_idx': n_ts + track_idx, 'element_type': element_type,
                'shared_energy': tp, 'element_energy': tp,
                'is_charged': is_charged_particle(cp_pid),
            })
    return connections


def split_caloparticles(connections, ev):
    cp_groups = defaultdict(list)
    for conn in connections:
        cp_groups[conn['cp_idx']].append(conn)

    split_cps  = []
    new_cp_idx = len(ev["simcan_raw_energy"])

    simcan_ispu  = ev["simcan_ispu"]
    simcan_eta   = ev["simcan_eta"]
    simcan_phi   = ev["simcan_phi"]
    simcan_pdgid = ev["simcan_pdgid"]
    n_ts         = len(ev["ts_energy"])
    n_ts_eg      = len(ev["tsEG_energy"])
    n_tracks     = len(ev["track_pt"])
    track_eta    = ev["track_eta"]
    ts_eta       = ev["ts_eta"]
    tsEG_eta     = ev["tsEG_eta"]
            
    for cp_idx, conns in cp_groups.items():
        if not conns:
            continue

        # ispu comes from the simcan array directly — same for all connections of this CP
        cp_ispu = float(simcan_ispu[cp_idx]) if cp_idx < len(simcan_ispu) else 0.0

        # Use first connection only to determine pid/charge for routing decisions
        # All other properties (cp_energy, cp_eta, cp_pid) come per-element from each conn
        abs_pid    = abs(conns[0]['cp_pid'])
        is_charged = conns[0]['is_charged']

        # Validate each connection by checking eta-sign consistency
        # elem_eta must have same sign as cp_eta (from the connection itself)
        valid_elements = []
        for conn in conns:
            elem_idx  = conn['element_idx']
            elem_type = conn['element_type']
            cp_eta    = conn['cp_eta']   # per-connection cp_eta
            elem_eta  = None
            if elem_type == 1:
                ti = elem_idx - n_ts
                if 0 <= ti < len(track_eta):
                    elem_eta = track_eta[ti]
            elif elem_type == 4:
                if 0 <= elem_idx < n_ts:
                    elem_eta = ts_eta[elem_idx]
            elif elem_type in (2, 3):
                # Check both ts space (no-superclustering) and tsEG space
                if elem_idx < n_ts:
                    # type 2 from ticlTracksterLinks (no-superclustering mode)
                    elem_eta = ts_eta[elem_idx]
                else:
                    ti = elem_idx - (n_ts + n_tracks)
                    if 0 <= ti < n_ts_eg:
                        elem_eta = tsEG_eta[ti]
            # type 5 no longer used
            if elem_eta is not None and cp_eta * elem_eta > 0:
                valid_elements.append(conn)

        if not valid_elements:
            continue

        tracks         = [e for e in valid_elements if e['element_type'] == 1]
        had_tracksters = [e for e in valid_elements if e['element_type'] == 4]
        em_tracksters  = [e for e in valid_elements if e['element_type'] == 2]

        if abs_pid in (11, 22, 13):
            # electrons and photons: use EM tracksters + tracks (type 1 for all)
            # muons: use tracks only

            if abs_pid == 11:
                if not tracks and not em_tracksters:
                    continue
                total_shared_tracks = sum(t['shared_energy'] for t in tracks) or 1.0
                total_shared_em     = sum(e['shared_energy'] for e in em_tracksters) or 1.0
    
                for t in tracks:
                    sf = t['shared_energy'] / total_shared_tracks
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx':      new_cp_idx,   # same idx for all elements of this electron
                        'element_idx':  t['element_idx'],
                        'element_type': 1,
                        'cp_energy':    t['cp_energy'] * sf,
                        'cp_pid':       t['cp_pid'],
                        'cp_eta':       t['cp_eta'],
                        'weight':       1.0,
                        'is_winner':    True,
                        'ispu':         cp_ispu,
                    })
                for em in em_tracksters:
                    sf = em['shared_energy'] / total_shared_em
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx':      new_cp_idx,   # same for all elements of this electron
                        'element_idx':  em['element_idx'],
                        'element_type': 2,
                        'cp_energy':    em['cp_energy'] * sf,
                        'cp_pid':       em['cp_pid'],
                        'cp_eta':       em['cp_eta'],
                        'weight':       1.0,
                        'is_winner':    True,
                        'ispu':         cp_ispu,
                    })

                new_cp_idx += 1
    
            elif abs_pid == 22:
                # photon: EM tracksters only
                if not em_tracksters:
                    continue
                total_shared = sum(e['shared_energy'] for e in em_tracksters) or 1.0
                for em in em_tracksters:
                    sf = em['shared_energy'] / total_shared
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx':      new_cp_idx,
                        'element_idx':  em['element_idx'],
                        'element_type': 2,
                        'cp_energy':    em['cp_energy'] * sf,
                        'cp_pid':       em['cp_pid'],
                        'cp_eta':       em['cp_eta'],
                        'weight':       1.0,
                        'is_winner':    True,
                        'ispu':         cp_ispu,
                    })
                    new_cp_idx += 1
        
            elif abs_pid == 13:  # muon - use tracks only, type 1
                if not tracks:
                    continue
                total_shared = sum(m['shared_energy'] for m in tracks) or 1.0
                for m in tracks:
                    sf = m['shared_energy'] / total_shared
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx':      new_cp_idx,
                        'element_idx':  m['element_idx'],
                        'element_type': 1,
                        'cp_energy':    m['cp_energy'] * sf,
                        'cp_pid':       m['cp_pid'],
                        'cp_eta':       m['cp_eta'],
                        'weight':       1.0,
                        'is_winner':    True,
                        'ispu':         cp_ispu,
                    })
                    new_cp_idx += 1

        elif is_charged and len(tracks) == 1:
            t = tracks[0]
            split_cps.append({
                'original_idx': cp_idx,
                'new_idx':      new_cp_idx,
                'element_idx':  t['element_idx'],
                'element_type': 1,
                'cp_energy':    t['cp_energy'],   
                'cp_pid':       t['cp_pid'],
                'cp_eta':       t['cp_eta'],
                'cp_fraction':  1.0,
                'weight':       1,
                'is_winner':    True,
                'ispu':         cp_ispu,
            })
            new_cp_idx += 1
            for had in had_tracksters:
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx':      new_cp_idx,
                    'element_idx':  had['element_idx'],
                    'element_type': 4,
                    'cp_energy':    0.0,            
                    'cp_pid':       had['cp_pid'],
                    'cp_eta':       had['cp_eta'],
                    'weight':       0.0,
                    'is_winner':    False,
                    'ispu':         cp_ispu,
                })
                new_cp_idx += 1

        elif is_charged and len(tracks) > 1:
            total_shared = sum(t['shared_energy'] for t in tracks) or 1.0
            for track in tracks:
                sf = track['shared_energy'] / total_shared
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx':      new_cp_idx,
                    'element_idx':  track['element_idx'],
                    'element_type': 1,
                    'cp_energy':    track['cp_energy'] * sf,   # from THIS track connection
                    'cp_pid':       track['cp_pid'],
                    'cp_eta':       track['cp_eta'],
                    'weight':       1,
                    'is_winner':    True,
                    'ispu':         cp_ispu,
                })
                new_cp_idx += 1
            for had in had_tracksters:
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx':      new_cp_idx,
                    'element_idx':  had['element_idx'],
                    'element_type': 4,
                    'cp_energy':    0.0,            # dummy , intentionally zero
                    'cp_pid':       had['cp_pid'],
                    'cp_eta':       had['cp_eta'],
                    'weight':       0.0,
                    'is_winner':    False,
                    'ispu':         cp_ispu,
                })
                new_cp_idx += 1

        else:  # neutral hadrons / charged hadrons without tracks matched 
            total_shared = sum(e['shared_energy'] for e in had_tracksters) or 1.0
            for elem in had_tracksters:
                sf = elem['shared_energy'] / total_shared
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx':      new_cp_idx,
                    'element_idx':  elem['element_idx'],
                    'element_type': elem['element_type'],
                    'cp_energy':    elem['cp_energy'] * sf,   # from THIS connection
                    'cp_pid':       elem['cp_pid'],
                    'cp_eta':       elem['cp_eta'],
                    'weight':       1,
                    'is_winner':    True,
                    'ispu':         cp_ispu,
                })
                new_cp_idx += 1

    return split_cps


def compute_trackster_features(ev):
    """
    Compute new RECO-level features for tracksters:
    1. min_dR to nearest good track at HGCAL entrance
    2. nearest track pT
    3. shower depth = Σ(|z|×E) / Σ(E) from layer clusters
    All features are RECO-level — no truth leakage.
    """
    # Good quality tracks only (PU-robust)
    trk_pt   = ev["track_pt"]
    trk_qual = ev["track_quality"]
    good     = (trk_pt >= 1) & (trk_qual >= 1)
    
    tk_eta = ev["track_hgcal_eta"][good]
    tk_phi = ev["track_hgcal_phi"][good]
    tk_pt  = trk_pt[good]

    def get_min_dR(ts_eta, ts_phi):
        if len(tk_eta) == 0:
            return 99.0, 0.0
        deta = tk_eta - ts_eta
        dphi = np.arctan2(np.sin(tk_phi - ts_phi),
                          np.cos(tk_phi - ts_phi))
        dR   = np.sqrt(deta**2 + dphi**2)
        idx  = np.argmin(dR)
        return float(dR[idx]), float(tk_pt[idx])

    n_ts = len(ev["ts_energy"])
    ts_min_dR  = np.zeros(n_ts)
    ts_near_pt = np.zeros(n_ts)
    ts_depth   = np.zeros(n_ts)

    for i in range(n_ts):
        # min dR to track
        ts_min_dR[i], ts_near_pt[i] = get_min_dR(
            float(ev["ts_eta"][i]), float(ev["ts_phi"][i]))
        
        # shower depth from layer clusters
        try:
            vz_raw = ev["ts_vertices_z"][i]
            ve_raw = ev["ts_vertices_e"][i]
            # Handle awkward/STLVector jagged arrays
            import awkward as ak
            vz = np.abs(np.array(ak.to_numpy(ak.flatten(ak.Array([vz_raw])))))
            ve = np.array(ak.to_numpy(ak.flatten(ak.Array([ve_raw]))))
            if len(vz) > 0 and ve.sum() > 0:
                ts_depth[i] = float(np.sum(vz * ve) / np.sum(ve))
            else:
                ts_depth[i] = float(np.abs(ev["ts_z"][i]))
        except Exception:
            ts_depth[i] = float(np.abs(ev["ts_z"][i]))

    # Feature 4: sum of track pT within dR < 0.10
    # n.had: ~0 (no tracks nearby)
    # none-chHAD: high (charged hadron track inside)
    ts_sum_pt10 = np.zeros(n_ts)
    for i in range(n_ts):
        ts_eta_i = float(ev["ts_eta"][i])
        ts_phi_i = float(ev["ts_phi"][i])
        if len(tk_eta) > 0:
            deta = tk_eta - ts_eta_i
            dphi = np.arctan2(np.sin(tk_phi - ts_phi_i),
                              np.cos(tk_phi - ts_phi_i))
            dR   = np.sqrt(deta**2 + dphi**2)
            ts_sum_pt10[i] = float(np.sum(tk_pt[dR < 0.10]))

    return ts_min_dR, ts_near_pt, ts_depth, ts_sum_pt10


def make_graph(ev, iev, use_superclustering=False):
    g = nx.DiGraph()

    n_ts     = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    n_ts_eg  = len(ev["tsEG_energy"])

    track_id_to_idx        = {tid: i for i, tid in enumerate(ev["track_id"])}
    # No truth-based type assignment for tracks - all tracks are type 1
    # The model learns to distinguish muons/electrons/hadrons from kinematics

    # Compute new RECO features for tracksters
    ts_min_dR, ts_near_pt, ts_depth, ts_sum_pt10 = compute_trackster_features(ev)

    ts_eta = ev["ts_eta"]; ts_phi = ev["ts_phi"]
    ts_pt  = ev["ts_pt"];  ts_en  = ev["ts_energy"]
    _theta = 2.0 * np.arctan(np.exp(-ts_eta.astype(np.float64)))
    _px    = ts_pt * np.cos(ts_phi)
    _py    = ts_pt * np.sin(ts_phi)
    _pz    = ts_en * np.cos(_theta)

    g.add_nodes_from(
        (("elem", i), dict(
            typ=4, pt=float(ts_pt[i]), energy=float(ts_en[i]),
            eta=float(ts_eta[i]), phi=float(ts_phi[i]), charge=0.0,
            px=float(_px[i]), py=float(_py[i]), pz=float(_pz[i]),
            em_energy=float(ev["ts_em_energy"][i]),
            bary_z=float(ev["ts_z"][i]),
            nhits=0.0,
            min_dR_track=float(ts_min_dR[i]),      # min dR to nearest track
            near_track_pt=float(ts_near_pt[i]),    # pT of nearest track
            shower_depth=float(ts_depth[i]),        # energy-weighted depth
            sum_pt_dR10=float(ts_sum_pt10[i]),     # sum track pT in dR<0.10
            isolation=float(ts_min_dR[i] * float(ts_en[i])),  # NEW: min_dR × energy
        ))
        for i in range(n_ts)
    )

    trk_pt  = ev["track_pt"];  trk_p   = ev["track_p"]
    trk_eta = ev["track_eta"]; trk_phi = ev["track_phi"]
    trk_chg = ev["track_charge"]; trk_id = ev["track_id"]
    trk_quality = ev["track_quality"]
    safe_eta = np.where(np.abs(trk_eta) < 10, trk_eta, 0.0)
    _pz_trk  = trk_pt * np.sinh(safe_eta.astype(np.float64))

    track_nodes = []
    for itrk in range(n_tracks):
        tid = trk_id[itrk]
        if trk_pt[itrk] < 1 or trk_quality[itrk] < 1:
            continue
        elem_type = 1  # all tracks are type 1 - model learns to distinguish
        track_nodes.append((
            ("elem", n_ts + itrk),
            dict(
                typ=elem_type,
                pt=float(trk_pt[itrk]), energy=float(trk_p[itrk]),
                eta=float(trk_eta[itrk]), phi=float(trk_phi[itrk]),
                layer=0, charge=float(trk_chg[itrk]),
                px=float(trk_pt[itrk] * math.cos(trk_phi[itrk])),
                py=float(trk_pt[itrk] * math.sin(trk_phi[itrk])),
                pz=float(_pz_trk[itrk]),
                em_energy=0.0,
                bary_z=0.0,
                nhits=float(ev["track_nhits"][itrk]),
            )
        ))
    g.add_nodes_from(track_nodes)

    eg_eta = ev["tsEG_eta"]; eg_phi = ev["tsEG_phi"]
    eg_pt  = ev["tsEG_pt"];  eg_en  = ev["tsEG_energy"]
    _theta_eg = 2.0 * np.arctan(np.exp(-eg_eta.astype(np.float64)))
    _px_eg    = eg_pt * np.cos(eg_phi)
    _py_eg    = eg_pt * np.sin(eg_phi)
    _pz_eg    = eg_en * np.cos(_theta_eg)

    if use_superclustering:
        g.add_nodes_from(
            (("elem", n_ts + n_tracks + i), dict(
                typ=2, pt=float(eg_pt[i]), energy=float(eg_en[i]),
                eta=float(eg_eta[i]), phi=float(eg_phi[i]), charge=0.0,
                px=float(_px_eg[i]), py=float(_py_eg[i]), pz=float(_pz_eg[i]),
                em_energy=float(ev["tsEG_em_energy"][i]),
                bary_z=float(ev["tsEG_z"][i]),
                nhits=0.0,
            ))
            for i in range(n_ts_eg)
        )


    em_connections  = collect_em_connections(ev, use_superclustering=use_superclustering)
    all_connections = (collect_hadronic_connections(ev) +
                       em_connections +
                       collect_track_connections(ev))
    split_cps = split_caloparticles(all_connections, ev)

    # Relabel ts nodes matched to EM CPs from type 4 to type 2
    # This ensures gamma/ele targets go to type 2 elements
    if not use_superclustering:
        em_ts_indices = set(conn["element_idx"] for conn in em_connections)
        for ts_idx in em_ts_indices:
            node_id = ("elem", ts_idx)
            if node_id in g.nodes:
                g.nodes[node_id]["typ"] = 2

    cp_totals = defaultdict(lambda: {'track': 0.0, 'cluster': 0.0})
    for sc in split_cps:
        key = 'track' if sc['element_type'] == 1 else 'cluster'
        cp_totals[sc['original_idx']][key] += sc['weight']

    simcan_eta = ev["simcan_eta"]; simcan_phi = ev["simcan_phi"]
    simcan_pid = ev["simcan_pdgid"]

    cp_nodes = []
    cp_edges = []

    seen_cp_idx = {}
    for sc in split_cps:
        original_idx = sc['original_idx']
        if original_idx >= len(ev["simcan_raw_energy"]):
            continue
        cp_idx    = sc['new_idx']
        node_id   = ("cp", cp_idx)

        if cp_idx not in seen_cp_idx:
            # first entry for this cp_idx sets the node attributes
            # for electrons this will be the track entry (appended first)
            cp_eta    = float(sc['cp_eta'])
            cp_phi    = float(simcan_phi[original_idx])
            cp_pid_v  = int(sc['cp_pid'])
            cp_energy = float(sc['cp_energy'])
            theta     = 2.0 * math.atan(math.exp(-cp_eta))
            cp_pt     = cp_energy * math.sin(theta)
            track_total   = cp_totals[original_idx]['track']
            cluster_total = cp_totals[original_idx]['cluster']
            
            cp_nodes.append((node_id, dict(
                pid=abs(int(cp_pid_v)), pt=cp_pt, eta=cp_eta,
                sin_phi=math.sin(cp_phi), cos_phi=math.cos(cp_phi),
                energy=cp_energy, ispu=float(sc['ispu']),
                generatorStatus=0, simulatorStatus=1,
                cp_to_track=float(track_total), cp_to_cluster=float(cluster_total),
                jet_idx=-1,
                px=cp_pt * math.cos(cp_phi), py=cp_pt * math.sin(cp_phi),
                pz=cp_energy * math.cos(theta),
                charge=get_charge(cp_pid_v), original_cp_idx=original_idx
            )))
            seen_cp_idx[cp_idx] = True

        # always add edge for every entry regardless
        elem_node = ("elem", sc['element_idx'])
        cp_edges.append((node_id, elem_node, sc['element_type'], sc['weight'], sc['is_winner']))

    g.add_nodes_from(cp_nodes)
        
    for (node_id, elem_node, element_type, weight, is_winner) in cp_edges:
        if elem_node not in g.nodes:
            continue
        g.add_edge(node_id, elem_node, weight=weight, is_winner=is_winner)
    
    n_tcan     = len(ev["tcan_pt"])
    cand_nodes = []
    for ipf in range(n_tcan):
        pid = ev["candidate_pdgId"][ipf] if ipf < len(ev["candidate_pdgId"]) else 211
        cand_nodes.append((
            ("pfcand", ipf),
            dict(pid=abs(int(pid)), charge=float(get_charge(pid)),
                 pt=float(ev["tcan_pt"][ipf]), eta=float(ev["tcan_eta"][ipf]),
                 sin_phi=math.sin(ev["tcan_phi"][ipf]),
                 cos_phi=math.cos(ev["tcan_phi"][ipf]),
                 energy=float(ev["tcan_energy"][ipf]),
                 ispu=0.0, generatorStatus=0, simulatorStatus=0,
                 cp_to_track=0.0, cp_to_cluster=0.0, jet_idx=-1)
        ))
    g.add_nodes_from(cand_nodes)

    cand_edges = []
    for cand_idx in range(n_tcan):
        cand_node    = ("pfcand", cand_idx)
        ts_data      = ev["trkst_indcies"][cand_idx]
        ts_indices   = ts_data if hasattr(ts_data, '__len__') else ([ts_data] if ts_data != -1 else [])
        trk_data     = ev["trks_indcies"][cand_idx]
        trk_idx_list = trk_data if hasattr(trk_data, '__len__') else ([trk_data] if trk_data != -1 else [])

        for ts_idx in ts_indices:
            if 0 <= ts_idx < n_ts:
                en = ("elem", ts_idx)
                if en in g.nodes:
                    cand_edges.append((en, cand_node, 1.0))
        for trk_idx in trk_idx_list:
            if 0 <= trk_idx < n_tracks:
                en = ("elem", n_ts + trk_idx)
                if en in g.nodes:
                    cand_edges.append((en, cand_node, 1.0))

    g.add_edges_from((u, v, {"weight": w}) for u, v, w in cand_edges)

    if "genpar_pdgid" in ev and len(ev["genpar_pdgid"]) > 0:
        n_gen  = len(ev["genpar_pdgid"])
        pts    = ev["genpar_pt"];   etas   = ev["genpar_eta"]
        phis   = ev["genpar_phi"];  ens    = ev["genpar_energy"]
        masses = ev["genpar_mass"]; stats  = ev["genpar_status"]
        mothers= ev["genpar_genPartIdxMother"]

        safe_eta_gen = np.where(np.abs(etas) < 10, etas, 0.0)
        px_gen = pts * np.cos(phis)
        py_gen = pts * np.sin(phis)
        pz_gen = pts * np.sinh(safe_eta_gen.astype(np.float64))

        gen_nodes_list = [
            (("gen", i), dict(
                pid=abs(int(ev["genpar_pdgid"][i])),
                pt=float(pts[i]), eta=float(etas[i]), phi=float(phis[i]),
                energy=float(ens[i]), mass=float(masses[i]),
                status=int(stats[i]),
                px=float(px_gen[i]), py=float(py_gen[i]), pz=float(pz_gen[i]),
                num_daughters=0, is_stable=(int(stats[i]) == 1)
            ))
            for i in range(n_gen)
        ]
        g.add_nodes_from(gen_nodes_list)

        gen_edges     = []
        num_daughters = defaultdict(int)
        for igen in range(n_gen):
            m = int(mothers[igen])
            if 0 <= m < n_gen:
                gen_edges.append((("gen", m), ("gen", igen)))
                num_daughters[m] += 1
        g.add_edges_from(gen_edges)
        for m, nd in num_daughters.items():
            g.nodes[("gen", m)]["num_daughters"] = nd

    return g


def find_representative_elements(g, elem_to_cp, cp_to_elem, elem_type, pid_type=0):
    elems = [(g.nodes[e]["pt"], e) for e in g.nodes
             if e[0] == "elem" and g.nodes[e]["typ"] == elem_type]
    for _, elem in sorted(elems, key=lambda x: x[0], reverse=True):
        cps = list(g.predecessors(elem))
        if pid_type:
            cps = [cp for cp in cps if abs(g.nodes[cp]["pid"]) == pid_type]
        cps_weight = [(g.edges[(cp, elem)]["weight"], cp) for cp in cps
                      if cp not in cp_to_elem and cp[0] == "cp"]
        if cps_weight:
            cp = max(cps_weight, key=lambda x: x[0])[1]
            elem_to_cp[elem] = cp
            cp_to_elem[cp]   = elem


def prepare_normalized_table(g):
    all_elements = [n for n in g.nodes if n[0] == "elem"]
    all_elements.sort(key=lambda x: (
        0 if g.nodes[x]["typ"] == 1 else
        1 if g.nodes[x]["typ"] == 2 else
        2 if g.nodes[x]["typ"] == 4 else 3,
        x[1]
    ))

    elem_to_primary_cp = {}
    cp_to_primary_elem = {}
    for et in (1, 2, 4):
        find_representative_elements(g, elem_to_primary_cp, cp_to_primary_elem, et)

    primary_pairs = set()
    for elem, cp in elem_to_primary_cp.items():
        primary_pairs.add((cp, elem))

    elem_to_all_cps = defaultdict(list)
    elem_to_cand    = {}
    for elem in all_elements:
        for pred in g.predecessors(elem):
            if pred[0] == "cp":
                # only include this CP if this elem won the priority for it
                if (pred, elem) in primary_pairs:
                    elem_to_all_cps[elem].append(pred)
        for succ in g.successors(elem):
            if succ[0] == "pfcand":
                elem_to_cand[elem] = succ
                break

    n       = len(all_elements)
    Xelem   = np.recarray((n,), dtype=[(name, np.float32) for name in elem_branches])
    ytarget = np.recarray((n,), dtype=[(name, np.float32) for name in particle_feature_order])
    ycand   = np.recarray((n,), dtype=[(name, np.float32) for name in particle_feature_order])
    Xelem.fill(0.0); ytarget.fill(0.0); ycand.fill(0.0)

    attrs = g.nodes

    for ielem, elem in enumerate(all_elements):
        nd = attrs[elem]
        for branch in elem_branches:
            if branch in nd:
                Xelem[branch][ielem] = nd[branch]

    for ielem, elem in enumerate(all_elements):
        elem_type = int(Xelem["typ"][ielem])
        cps       = elem_to_all_cps.get(elem, [])

        if cps:
            highest_energy_cp = max(cps, key=lambda cp: attrs[cp]["energy"])
            pid = attrs[highest_energy_cp]["pid"]

            total_energy = 0.0
            ws = defaultdict(float)
            for cp in cps:
                w         = g.edges[(cp, elem)].get('weight', 1.0)
                cp_energy = attrs[cp]["energy"] * w
                total_energy += cp_energy
                for key in ('charge', 'pt', 'eta', 'sin_phi', 'cos_phi',
                            'energy', 'ispu', 'cp_to_track', 'cp_to_cluster'):
                    if key in attrs[cp]:
                        ws[key] += attrs[cp][key] * cp_energy

            if total_energy > 0:
                inv = 1.0 / total_energy
                ytarget["pid"][ielem]           = pid
                ytarget["charge"][ielem]         = ws['charge']        * inv
                ytarget["pt"][ielem]             = ws['pt']             * inv
                ytarget["eta"][ielem]            = ws['eta']            * inv
                ytarget["sin_phi"][ielem]        = ws['sin_phi']        * inv
                ytarget["cos_phi"][ielem]        = ws['cos_phi']        * inv
                ytarget["energy"][ielem]         = total_energy
                ytarget["ispu"][ielem]           = ws['ispu']           * inv
                ytarget["cp_to_track"][ielem]    = ws['cp_to_track']    * inv
                ytarget["cp_to_cluster"][ielem]  = ws['cp_to_cluster']  * inv
        else:
            nd = attrs[elem]
            ytarget["charge"][ielem]  = nd.get("charge", 0.0)
            ytarget["eta"][ielem]     = nd["eta"]
            ytarget["sin_phi"][ielem] = math.sin(nd["phi"])
            ytarget["cos_phi"][ielem] = math.cos(nd["phi"])
            ytarget["ispu"][ielem]    = 1.0

        ytarget["simulatorStatus"][ielem] = 1 if cps else 0
        ytarget["jet_idx"][ielem]         = -1

    for ielem, elem in enumerate(all_elements):
        cand = elem_to_cand.get(elem)
        if cand is not None:
            nd = attrs[cand]
            for feature in particle_feature_order:
                if feature in nd:
                    ycand[feature][ielem] = nd[feature]

    return Xelem, ycand, ytarget


def _process_file_worker(args):
    input_file, num_events, start_event, use_superclustering = args
    return process_file_no_progress(input_file, num_events, start_event, use_superclustering=use_superclustering)


def process_file_no_progress(input_file, num_events=-1, start_event=0, use_superclustering=False):
    tf = uproot.open(input_file)
    trees = {}
    tree_names = {
        'tkst':   'ticlDumper/ticlTracksterLinks',
        'simtkst':'ticlDumper/simtrackstersCP',
        'cand':   'ticlDumper/candidates',
        'simcan': 'ticlDumper/simTICLCandidate',
        'track':  'ticlDumper/tracks',
        'assoc':  'ticlDumper/associations',
        'tkstEG': 'ticlDumper/ticlTracksterLinksSuperclusteringDNN',
        'genpar': 'ticlDumper/genparticles'
    }
    for key, tname in tree_names.items():
        try:    trees[key] = tf[tname]
        except: trees[key] = None

    if trees['tkst'] is None:
        return []

    total_events = trees['tkst'].num_entries
    if num_events == -1:
        num_events = total_events - start_event
    else:
        num_events = min(num_events, total_events - start_event)

    all_data = []
    pt_min   = 3.0


    # Batch read ALL events at once — much faster than per-event reading
    all_events, num_events = read_all_events(trees, start_event, num_events)
    pt_min = 3.0

    for iev, ev in enumerate(all_events):
        try:
            g     = make_graph(ev, iev, use_superclustering=use_superclustering)
            Xelem, ycand, ytarget = prepare_normalized_table(g)

            truth_jets  = compute_truth_jets(g, pt_min=pt_min)
            targetjets  = compute_target_jets(ytarget, pt_min=pt_min)
            genmet      = compute_genmet(g)
            ytarget_ji  = assign_jet_indices(ytarget, targetjets, pt_min=pt_min)

            stable_gen = []
            for node in g.nodes:
                if node[0] == "gen" and g.nodes[node].get("status", 0) == 1:
                    pid = g.nodes[node]["pid"]
                    if pid not in _NEUTRINO_PIDS:
                        stable_gen.append([
                            pid,
                            g.nodes[node]["pt"],
                            g.nodes[node]["eta"],
                            g.nodes[node]["phi"],
                            g.nodes[node]["energy"]
                        ])
            stable_gen_array = np.array(stable_gen, dtype=np.float32) \
                               if stable_gen else np.array([], dtype=np.float32)

            all_data.append({
                "Xelem":            Xelem,
                "ycand":            ycand,
                "ytarget":          ytarget_ji,
                "genjet":           truth_jets,
                "targetjet":        targetjets,
                "genmet":           genmet,
                "pythia":           stable_gen_array,
                "event_idx":        start_event + iev,
                "file_name":        os.path.basename(input_file),
                "global_event_idx": len(all_data)
            })
        except Exception as e:
            print(f"\nERROR event {start_event + iev} in {input_file}: {e}")
            traceback.print_exc()
    return all_data


def process_files(input_list, output_file, num_events=-1, events_per_file=-1,
                  num_workers=None, events_per_pkl=-1, use_superclustering=False):
    input_list = [f.strip() for f in input_list if f.strip() and not f.startswith('#')]

    if num_workers is None:
        num_workers = min(len(input_list), max(1, multiprocessing.cpu_count() - 1))

    tasks     = []
    remaining = num_events
    for f in input_list:
        if num_events > 0 and remaining <= 0:
            break
        if events_per_file > 0:
            n = events_per_file
        elif num_events > 0:
            n = remaining
        else:
            n = -1
        tasks.append((f, n, 0, use_superclustering))
        if num_events > 0:
            remaining -= (n if n > 0 else 9999999)

    all_data   = []
    total_done = 0
    chunk_idx  = 0

    print(f"Launching {num_workers} parallel workers for {len(tasks)} files ...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_file_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Files", unit="file") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    file_data = future.result()
                    all_data.extend(file_data)
                    total_done += len(file_data)
                    pbar.set_postfix(events=total_done)

                    # Save incrementally if events_per_pkl is set
                    if events_per_pkl > 0:
                        while len(all_data) >= events_per_pkl:
                            chunk = all_data[:events_per_pkl]
                            all_data = all_data[events_per_pkl:]
                            base = output_file.replace('.pkl', '')
                            out_path = f"{base}_{chunk_idx:04d}.pkl"
                            print(f"\n  Auto-saving {len(chunk)} events to {out_path} ...")
                            with open(out_path, 'wb') as f:
                                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                            print(f"  Saved {out_path}")
                            chunk_idx += 1

                except Exception as e:
                    print(f"\nERROR in worker for {task[0]}: {e}")
                pbar.update(1)

    return all_data, chunk_idx


def main():
    parser = argparse.ArgumentParser(description='Process TICL graph data (fast)')
    parser.add_argument('--input',           type=str, required=True)
    parser.add_argument('--output',          type=str, default='ticl_graph_data.pkl')
    parser.add_argument('--num-events',      type=int, default=-1)
    parser.add_argument('--events-per-file', type=int, default=-1)
    parser.add_argument('--max-files',       type=int, default=-1)
    parser.add_argument('--num-workers',     type=int, default=None,
                        help='Parallel worker processes (default: #CPUs-1)')
    parser.add_argument('--events-per-pkl',  type=int, default=-1,
                        help='Split output into multiple PKLs with this many events each (-1 = single file)')
    parser.add_argument('--use-superclustering',
                        dest='use_superclustering',
                        action='store_true',
                        default=False,
                        help='Use SuperclusteringDNN for EM tracksters (default: False, uses ticlTracksterLinks)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found!"); return

    if args.input.endswith('.root'):
        input_files = [args.input]
    elif args.input.endswith('.txt'):
        with open(args.input) as f:
            input_files = [l.strip() for l in f
                           if l.strip() and not l.startswith('#') and l.strip().endswith('.root')]
        if args.max_files > 0:
            input_files = input_files[:args.max_files]
        print(f"Found {len(input_files)} ROOT files")
    else:
        print("ERROR: input must be .root or .txt"); return

    missing = [f for f in input_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: {len(missing)} files missing (first: {missing[0]})"); return

    all_data, chunk_idx = process_files(input_files, args.output,
                             args.num_events, args.events_per_file,
                             args.num_workers, args.events_per_pkl,
                             use_superclustering=args.use_superclustering)

    if not all_data and chunk_idx == 0:
        print("ERROR: no events processed!"); return

    # Save any remaining events
    if args.events_per_pkl > 0:
        if all_data:
            base = args.output.replace('.pkl', '')
            out_path = f"{base}_{chunk_idx:04d}.pkl"
            print(f"\nSaving remaining {len(all_data)} events to {out_path} ...")
            with open(out_path, 'wb') as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_idx += 1
        print(f"\nDone! Total {chunk_idx} PKL files saved.")
    else:
        print(f"\nSaving {len(all_data)} events to {args.output} ...")
        with open(args.output, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elements   = sum(len(e['Xelem'])                  for e in all_data)
    total_tracks     = sum(np.sum(e['Xelem']['typ'] == 1)   for e in all_data)
    total_tracksters = sum(np.sum(e['Xelem']['typ'] == 4)   for e in all_data)
    total_em_ts      = sum(np.sum(e['Xelem']['typ'] == 2)   for e in all_data)
    total_cp         = sum(np.sum(e['ytarget']['pid'] > 0)  for e in all_data)
    # Quick sanity check on genmet
    genmet_vals = np.array([e['genmet'][0] for e in all_data])
    n = len(all_data)
    print(f"\nDone!  Events: {n}  |  Files: {len(input_files)}")
    print(f"  avg elements/event:    {total_elements/n:.1f}")
    print(f"  avg tracks/event:      {total_tracks/n:.1f}")
    print(f"  avg HAD tracksters/ev: {total_tracksters/n:.1f}")
    print(f"  avg EM  tracksters/ev: {total_em_ts/n:.1f}")
    print(f"  avg CaloParticles/ev:  {total_cp/n:.1f}")
    print(f"  genMET mean={genmet_vals.mean():.2f}  median={np.median(genmet_vals):.2f}  max={genmet_vals.max():.2f} GeV")


if __name__ == "__main__":
    main()
