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
import math
import traceback

#python3 postprocessing_ticl.py --input 211_0pu.txt --output ticl_graph_data_pion_0pu.pkl

# To prevent threading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Feature definitions
elem_branches = [
    "typ", "pt", "eta", "phi", "energy", "layer", "charge", 
    "px", "py", "pz",
    "sigma_x", "sigma_y", "sigma_z", "deltap", "sigmadeltap",
    "num_hits", "cluster_flags", "corr_energy", "corr_energy_err",
    "vx", "vy", "vz", "pterror", "etaerror", "phierror", "lambd", "lambdaerror",
    "theta", "thetaerror", "time", "timeerror", "etaerror1", "etaerror2",
]

particle_feature_order = [
    "pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
    "ispu", "generatorStatus", "simulatorStatus", "cp_to_track",
    "cp_to_cluster", "jet_idx"
]

def compute_truth_jets(g, pt_min=3.0):  
    # Get all gen particle nodes
    gen_nodes = [n for n in g.nodes if n[0] == "gen"]
    
    if not gen_nodes:
        return np.array([])
    
    # Select stable particles (status=1) and exclude neutrinos (pid 12,14,16)
    stable_particles = []
    for node in gen_nodes:
        status = g.nodes[node].get("status", 0)
        pid = g.nodes[node].get("pid", 0)
        
        # status=1 means stable final-state particle
        # Skip neutrinos (12,14,16) as they don't interact 
        if status == 1 and pid not in [12, 14, 16]:
            stable_particles.append(node)
    
    if not stable_particles:
        return np.array([])
    
    # Create arrays for FastJet
    try:
        # Create list of PseudoJets - FastJet's native format
        pseudojets = []
        for node in stable_particles:
            px = g.nodes[node]["pt"] * math.cos(g.nodes[node]["phi"])
            py = g.nodes[node]["pt"] * math.sin(g.nodes[node]["phi"])
            pz = g.nodes[node]["pt"] * math.sinh(g.nodes[node]["eta"]) if abs(g.nodes[node]["eta"]) < 10 else 0
            e = g.nodes[node]["energy"]
            
            # Create PseudoJet from px, py, pz, e
            pj = fastjet.PseudoJet(px, py, pz, e)
            pseudojets.append(pj)
        
        # Define jet algorithm (anti-kT, R=0.4)
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        
        # Cluster jets - pass the list of PseudoJets directly
        cluster = fastjet.ClusterSequence(pseudojets, jetdef)
        jets = cluster.inclusive_jets(ptmin=pt_min)  # Use pt_min, not min_pt!
        
        # Convert to numpy array
        if len(jets) > 0:
            genjet = np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets])
        else:
            genjet = np.array([])
        
        return genjet
        
    except ImportError as e:
        print(f"Warning: Required package not installed: {e}")
        return np.array([])
    except Exception as e:
        print(f"Warning: Jet clustering failed: {e}")
        return np.array([])
    

def get_charge(pid):
    abs_pid = abs(pid)
    if pid in [130, 22, 1, 2, 310]:
        return 0.0
    elif abs_pid in [11, 13, 211, 321]:  
        return -math.copysign(1.0, pid)
    else:
        return 0.0

def is_charged_particle(pid):
    abs_pid = abs(pid)
    return abs_pid in [211, 13, 11, 321]

def read_event(trees, iev):
    ev = {}
    
    # hadronic tracksters
    tkst_arrays = trees['tkst'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", "raw_pt"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["ts_energy"] = tkst_arrays["raw_energy"][0]
    ev["ts_pt"] = tkst_arrays["raw_pt"][0]
    ev["ts_eta"] = tkst_arrays["barycenter_eta"][0]
    ev["ts_phi"] = tkst_arrays["barycenter_phi"][0]

    # EM tracksters (SuperClusterDNN)
    tkst_arrays = trees['tkstEG'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", "raw_pt"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["tsEG_energy"] = tkst_arrays["raw_energy"][0]
    ev["tsEG_pt"] = tkst_arrays["raw_pt"][0]
    ev["tsEG_eta"] = tkst_arrays["barycenter_eta"][0]
    ev["tsEG_phi"] = tkst_arrays["barycenter_phi"][0]


    # simtrackstersCP
    simtkst_arrays = trees['simtkst'].arrays(
        ["regressed_energy", "barycenter_eta", "barycenter_phi", "pdgID", "trackIdx", "CPidx"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["simtkst_energy"] = simtkst_arrays["regressed_energy"][0]
    ev["simtkst_eta"] = simtkst_arrays["barycenter_eta"][0]
    ev["simtkst_phi"] = simtkst_arrays["barycenter_phi"][0]
    ev["simtkst_pdgid"] = simtkst_arrays["pdgID"][0]
    ev["simtkst_trackIdx"] = simtkst_arrays["trackIdx"][0]
    ev["simtkst_CPidx"] = simtkst_arrays["CPidx"][0]

    # candidates
    tcan_arrays = trees['cand'].arrays(
        ["candidate_pt", "candidate_eta", "candidate_phi", "candidate_energy",
         "trackstersLinks_in_candidate", "candidate_pdgId", "track_in_candidate"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["tcan_pt"] = tcan_arrays["candidate_pt"][0]
    ev["tcan_eta"] = tcan_arrays["candidate_eta"][0]
    ev["tcan_phi"] = tcan_arrays["candidate_phi"][0]
    ev["tcan_energy"] = tcan_arrays["candidate_energy"][0]
    ev["trkst_indcies"] = tcan_arrays["trackstersLinks_in_candidate"][0]
    ev["trks_indcies"] = tcan_arrays["track_in_candidate"][0]
    ev["candidate_pdgId"] = tcan_arrays["candidate_pdgId"][0]

    # Read simcandidates
    simcan_arrays = trees['simcan'].arrays(
        ["simTICLCandidate_pdgId", "simTICLCandidate_pt", "simTICLCandidate_eta", 
         "simTICLCandidate_phi", "simTICLCandidate_regressed_energy", 
         "simTICLCandidate_tracks_in_candidate", "simTICLCandidate_simTracksterCPIndex",
         "simTICLCandidate_ispu"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["simcan_pdgid"] = simcan_arrays["simTICLCandidate_pdgId"][0]
    ev["simcan_pt"] = simcan_arrays["simTICLCandidate_pt"][0]
    ev["simcan_eta"] = simcan_arrays["simTICLCandidate_eta"][0]
    ev["simcan_phi"] = simcan_arrays["simTICLCandidate_phi"][0]
    ev["simcan_energy"] = simcan_arrays["simTICLCandidate_regressed_energy"][0]
    ev["simcan_trkId"] = simcan_arrays["simTICLCandidate_tracks_in_candidate"][0]
    ev["simcan_simTracksterCPIndex"] = simcan_arrays["simTICLCandidate_simTracksterCPIndex"][0]
    ev["simcan_ispu"] = simcan_arrays["simTICLCandidate_ispu"][0]

    gen_arrays = trees['genpar'].arrays(
        ["GenPart_status", "GenPart_genPartIdxMother", "GenPart_eta",
         "GenPart_phi", "GenPart_pdgId", "GenPart_mass","GenPart_pt", "GenPart_energy"
         ],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["genpar_pdgid"] = gen_arrays["GenPart_pdgId"][0]
    ev["genpar_mass"] = gen_arrays["GenPart_mass"][0]
    ev["genpar_eta"] = gen_arrays["GenPart_eta"][0]
    ev["genpar_phi"] = gen_arrays["GenPart_phi"][0]
    ev["genpar_status"] = gen_arrays["GenPart_status"][0]
    ev["genpar_genPartIdxMother"] = gen_arrays["GenPart_genPartIdxMother"][0]
    ev["genpar_pt"] = gen_arrays["GenPart_pt"][0]
    ev["genpar_energy"] = gen_arrays["GenPart_energy"][0]

    # tracks
    track_arrays = trees['track'].arrays(
        ["track_pt", "track_p", "track_eta", "track_hgcal_phi", "track_charge", "track_id"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["track_pt"] = track_arrays["track_pt"][0]
    ev["track_p"] = track_arrays["track_p"][0]
    ev["track_eta"] = track_arrays["track_eta"][0]
    ev["track_phi"] = track_arrays["track_hgcal_phi"][0]
    ev["track_charge"] = track_arrays["track_charge"][0]
    ev["track_id"] = track_arrays["track_id"][0]

    # Read associations
    assoc_arrays = trees['assoc'].arrays(
        ["ticlTracksterLinks_recoToSim_CP_score", "ticlTracksterLinks_simToReco_CP_score",
         "ticlTracksterLinks_simToReco_CP_sharedE", "ticlTracksterLinks_recoToSim_CP",
         "ticlTracksterLinks_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score",
         "ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP",
         "ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["ticlTracksterLinks_recoToSim_CP_score"] = assoc_arrays["ticlTracksterLinks_recoToSim_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_score"] = assoc_arrays["ticlTracksterLinks_simToReco_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_sharedE"] = assoc_arrays["ticlTracksterLinks_simToReco_CP_sharedE"][0]
    ev["ticlTracksterLinks_recoToSim_CP"] = assoc_arrays["ticlTracksterLinks_recoToSim_CP"][0]
    ev["ticlTracksterLinks_simToReco_CP"] = assoc_arrays["ticlTracksterLinks_simToReco_CP"][0]

    ev["ticlTracksterLinksDNN_recoToSim_CP_score"] = assoc_arrays["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP_score"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP_score"] = assoc_arrays["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_score"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"] = assoc_arrays["ticlTracksterLinksSuperclusteringDNN_simToReco_CP_sharedE"][0]
    ev["ticlTracksterLinksDNN_recoToSim_CP"] = assoc_arrays["ticlTracksterLinksSuperclusteringDNN_recoToSim_CP"][0]
    ev["ticlTracksterLinksDNN_simToReco_CP"] = assoc_arrays["ticlTracksterLinksSuperclusteringDNN_simToReco_CP"][0]


    return ev
#Collect connections from hadronic tracksters (type 4)
def collect_hadronic_connections(ev):
    connections = []
    n_ts = len(ev["ts_energy"])
    
    reco_to_sim_scores = ev["ticlTracksterLinks_recoToSim_CP_score"]
    sim_to_reco_scores = ev["ticlTracksterLinks_simToReco_CP_score"]
    sim_to_reco_sharedE = ev["ticlTracksterLinks_simToReco_CP_sharedE"]
    sim_to_reco_index = ev["ticlTracksterLinks_simToReco_CP"]
    
    for sim_idx in range(len(sim_to_reco_sharedE)):
        if len(sim_to_reco_sharedE[sim_idx]) > 0:
            for idx, (trackster_idx, shared_energy) in enumerate(
                    zip(sim_to_reco_index[sim_idx], sim_to_reco_sharedE[sim_idx])):
                if shared_energy <= 0:
                    continue
                reco_score = reco_to_sim_scores[trackster_idx][0] if trackster_idx < len(reco_to_sim_scores) and len(reco_to_sim_scores[trackster_idx]) > 0 else 1.0
                sim_score = sim_to_reco_scores[sim_idx][idx] if sim_idx < len(sim_to_reco_scores) and idx < len(sim_to_reco_scores[sim_idx]) else 1.0
                if reco_score > 0.6 or sim_score > 0.9:
                    continue
                    
                cp_energy = ev["simcan_energy"][sim_idx]
                element_energy = ev["ts_energy"][trackster_idx]
                cp_pid = ev["simcan_pdgid"][sim_idx]
                cp_eta = ev["simcan_eta"][sim_idx]

                connections.append({
                    'cp_idx': sim_idx,
                    'cp_pid': cp_pid,
                    'cp_energy': cp_energy,
                    'cp_eta': cp_eta,
                    'element_idx': trackster_idx,
                    'element_type': 4,  # Hadronic trackster
                    'shared_energy': shared_energy,
                    'element_energy': element_energy,
                    'is_charged': is_charged_particle(cp_pid)
                })
    
    return connections
#Collect connections from EM tracksters (SuperClusterDNN) - types 2 and 3
def collect_em_connections(ev):
    connections = []
    n_ts_eg = len(ev["tsEG_energy"])
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    
    reco_to_sim_scores = ev["ticlTracksterLinksDNN_recoToSim_CP_score"]
    sim_to_reco_scores = ev["ticlTracksterLinksDNN_simToReco_CP_score"]
    sim_to_reco_sharedE = ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"]
    sim_to_reco_index = ev["ticlTracksterLinksDNN_simToReco_CP"]

    if sim_to_reco_sharedE is None or len(sim_to_reco_sharedE) == 0:
        return connections
    
    for sim_idx in range(len(sim_to_reco_sharedE)):
        if len(sim_to_reco_sharedE[sim_idx]) > 0:
            for idx, (trackster_idx, shared_energy) in enumerate(
                    zip(sim_to_reco_index[sim_idx], sim_to_reco_sharedE[sim_idx])):
                if shared_energy <= 0:
                    continue
                reco_score = reco_to_sim_scores[trackster_idx][0] if trackster_idx < len(reco_to_sim_scores) and len(reco_to_sim_scores[trackster_idx]) > 0 else 1.0
                sim_score = sim_to_reco_scores[sim_idx][idx] if sim_idx < len(sim_to_reco_scores) and idx < len(sim_to_reco_scores[sim_idx]) else 1.0
                if reco_score > 0.6 or sim_score > 0.9:
                    continue
                
                # Determine element type based on CP PID and track match
                cp_pid = ev["simcan_pdgid"][sim_idx]
                cp_energy = ev["simcan_energy"][sim_idx]
                cp_eta = ev["simcan_eta"][sim_idx]

                element_energy = ev["tsEG_energy"][trackster_idx]
                
                has_track = len(ev["simcan_trkId"][sim_idx]) > 0
                if abs(cp_pid) == 11 or (abs(cp_pid) == 22 and has_track):
                    elem_type = 2  # Electron type
                elif abs(cp_pid) == 22:
                    # Pure photon
                    elem_type = 3  # Photon type
                else:
                    continue
                connections.append({
                    'cp_idx': sim_idx,
                    'cp_pid': cp_pid,
                    'cp_energy': cp_energy,
                    'cp_eta': cp_eta,
                    'element_idx': n_ts + n_tracks + trackster_idx,
                    'element_type': elem_type,
                    'shared_energy': shared_energy,
                    'element_energy': element_energy,
                    'is_charged': is_charged_particle(cp_pid)
                })
    return connections
#Collect connections from tracks (type 1)
def collect_track_connections(ev):
    connections = []
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    
    track_id_to_idx = {tid: i for i, tid in enumerate(ev["track_id"])}
    
    for cp_idx in range(len(ev["simcan_trkId"])):
        track_indices = ev["simcan_trkId"][cp_idx]
        if len(track_indices) == 0:
            continue
            
        cp_energy = ev["simcan_energy"][cp_idx]
        cp_pid = ev["simcan_pdgid"][cp_idx]
        cp_eta = ev["simcan_eta"][cp_idx]
        
        for track_id in track_indices:  
            track_idx = track_id_to_idx.get(track_id)
            if track_idx is None:
                continue
                
            track_p = ev["track_p"][track_idx]
            connections.append({
                'cp_idx': cp_idx,
                'cp_pid': cp_pid,
                'cp_energy': cp_energy,
                'cp_eta': cp_eta,
                'element_idx': n_ts + track_idx,  # Offset for tracks after tracksters
                'element_type': 1,  # Track
                'shared_energy': track_p,
                'element_energy': track_p,
                'is_charged': is_charged_particle(cp_pid)
            })
    
    return connections
#Collect connections for muons (type 5) from candidates
def collect_muon_connections(ev):
    connections = []
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    n_ts_eg = len(ev["tsEG_energy"])
    
    # Muon offset: after tracks and all tracksters
    muon_offset = n_ts + n_tracks + n_ts_eg
    
    track_id_to_idx = {tid: i for i, tid in enumerate(ev["track_id"])}
    
    for cp_idx in range(len(ev["simcan_trkId"])):
        cp_pid = ev["simcan_pdgid"][cp_idx]
        
        # Only process muons
        if abs(cp_pid) != 13:
            continue
            
        track_indices = ev["simcan_trkId"][cp_idx]
        if len(track_indices) == 0:
            continue
            
        cp_energy = ev["simcan_energy"][cp_idx]
        cp_eta = ev["simcan_eta"][cp_idx]
        
        # Find matching muon candidate
        for cand_idx in range(len(ev["tcan_pt"])):
            if abs(ev["candidate_pdgId"][cand_idx]) == 13:
                # Check if this candidate uses the same track
                track_data = ev["trks_indcies"][cand_idx]
                if hasattr(track_data, '__len__'):
                    cand_track_indices = track_data
                else:
                    cand_track_indices = [track_data] if track_data != -1 else []
                # Match via track indices
                for track_id in track_indices:
                    track_idx = track_id_to_idx.get(track_id)
                    if track_idx is not None and track_idx in cand_track_indices:
                        # Found matching muon
                        connections.append({
                            'cp_idx': cp_idx,
                            'cp_pid': cp_pid,
                            'cp_energy': cp_energy,
                            'cp_eta': cp_eta,
                            'element_idx': muon_offset + cand_idx,
                            'element_type': 5,  # Muon element
                            'shared_energy': cp_energy,  # Use full CP energy
                            'element_energy': ev["tcan_energy"][cand_idx],
                            'is_charged': True
                        })
                        break

    return connections
#Split CPs that match multiple tracks (for hadrons only)
def split_caloparticles(connections, ev):
    cp_groups = defaultdict(list)
    for conn in connections:
        cp_groups[conn['cp_idx']].append(conn)
    
    split_cps = []
    new_cp_idx = len(ev["simcan_energy"])
    n_ts = len(ev["ts_energy"])
    n_ts_eg = len(ev["tsEG_energy"])
    n_tracks = len(ev["track_pt"])
    
    for cp_idx, conns in cp_groups.items():
        if len(conns) == 0:
            continue
        cp_pid = conns[0]['cp_pid']
        cp_energy = conns[0]['cp_energy']
        cp_eta = conns[0]['cp_eta']  
        is_charged = conns[0]['is_charged']
        cp_ispu = float(ev["simcan_ispu"][cp_idx]) if cp_idx < len(ev["simcan_ispu"]) else 0.0


        # ==== ETA SIGN VALIDATION ====
        valid_elements = []
        for conn in conns:
            elem_idx = conn['element_idx']
            elem_eta = None
            
            if conn['element_type'] == 1:  # Track
                track_idx = elem_idx - n_ts
                if 0 <= track_idx < len(ev["track_eta"]):
                    elem_eta = ev["track_eta"][track_idx]
            elif conn['element_type'] == 4:  # Hadronic trackster
                if 0 <= elem_idx < n_ts:
                    elem_eta = ev["ts_eta"][elem_idx]
            elif conn['element_type'] in [2, 3]:  # EM tracksters
                tsEgIndx = elem_idx - (n_ts + n_tracks)
                if 0 <= tsEgIndx < n_ts_eg:
                    elem_eta = ev["tsEG_eta"][tsEgIndx]
            elif conn['element_type'] == 5:  # Muon
                # Use CP eta for muons
                elem_eta = cp_eta
            
            if elem_eta is not None:
                if cp_eta * elem_eta > 0:  # Same hemisphere
                    valid_elements.append(conn)
        
        if len(valid_elements) == 0:
            continue

        # Separate by type
        tracks = [e for e in valid_elements if e['element_type'] == 1]
        had_tracksters = [e for e in valid_elements if e['element_type'] == 4]
        em_tracksters = [e for e in valid_elements if e['element_type'] in [2, 3]]
        muons = [e for e in valid_elements if e['element_type'] == 5]
        # NO SPLITTING for electrons, photons, muons
        if abs(cp_pid) in [11, 22, 13]:
    
            # ELECTRONS
            if abs(cp_pid) == 11:
                relevant = [e for e in (tracks + em_tracksters) if e['element_type'] in [1, 2]]
                if relevant:
                    total_shared = sum(e['shared_energy'] for e in relevant)
                    for elem in relevant:
                        weight = elem['shared_energy'] / total_shared if total_shared > 0 else 1.0/len(relevant)
                        split_cps.append({
                            'original_idx': cp_idx,
                            'new_idx': new_cp_idx,
                            'element_idx': elem['element_idx'],
                            'element_type': elem['element_type'],
                            'cp_energy': cp_energy,
                            'cp_pid': cp_pid,
                            'weight': weight,
                            'is_winner': True,
                            'cp_eta': cp_eta,
                            'ispu': cp_ispu,
                        })
                    new_cp_idx += 1
    
            # PHOTONS
            elif abs(cp_pid) == 22:
                # Only EM tracksters matter
                if em_tracksters:
                    total_shared = sum(e['shared_energy'] for e in em_tracksters)
                    for em_elem in em_tracksters:
                        weight = em_elem['shared_energy'] / total_shared if total_shared > 0 else 1.0/len(em_tracksters)
                        split_cps.append({
                            'original_idx': cp_idx,
                            'new_idx': new_cp_idx,
                            'element_idx': em_elem['element_idx'],
                            'element_type': 3,  #type 3 for photons
                            'cp_energy': cp_energy,
                            'cp_pid': cp_pid,
                            'weight': weight,
                            'is_winner': True,
                            'cp_eta': cp_eta,
                            'ispu': cp_ispu,
                        })
                    new_cp_idx += 1
    
            # MUONS
            elif abs(cp_pid) == 13:
                if muons:
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx': new_cp_idx,
                        'element_idx': muons[0]['element_idx'],
                        'element_type': 5,
                        'cp_energy': cp_energy,
                        'cp_pid': cp_pid,
                        'weight': 1.0,
                        'is_winner': True,
                        'cp_eta': cp_eta,
                        'ispu': cp_ispu,
                    })
                    new_cp_idx += 1

        # No SPLITTING for hadrons (charged hadrons with single tracks)
        elif is_charged and len(tracks) == 1:
            # Split among tracks
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx': new_cp_idx,
                    'element_idx': tracks[0]['element_idx'],
                    'element_type': 1,
                    'cp_energy': cp_energy,
                    'cp_pid': cp_pid,
	            'cp_fraction': 1.0,
                    'weight': 1,
                    'is_winner': True,
                    'cp_eta': cp_eta,
                    'ispu': cp_ispu,
	        })
                new_cp_idx += 1
            
                # Hadronic tracksters get dummy CPs
                for had in had_tracksters:
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx': new_cp_idx,
                        'element_idx': had['element_idx'],
                        'element_type': 4,
                        'cp_energy': 0.0,
                        'cp_pid': cp_pid,
                        'weight': 0.0,
                        'is_winner': False,
                        'cp_eta': cp_eta,
                        'ispu': cp_ispu,
                    })
                new_cp_idx += 1
        
        # SPLITTING for hadrons (charged hadrons with multiple tracks)
        elif is_charged and len(tracks) > 1:
            # Split among tracks
            total_shared = sum(t['shared_energy'] for t in tracks)
            for track in tracks:
                split_fraction = track['shared_energy'] / total_shared if total_shared > 0 else 1.0/len(tracks)
                split_energy = cp_energy * split_fraction
                
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx': new_cp_idx,
                    'element_idx': track['element_idx'],
                    'element_type': 1,
                    'cp_energy': split_energy,
                    'cp_pid': cp_pid,
                    'weight': split_fraction,
                    'is_winner': True,
                    'cp_eta': cp_eta,
                    'ispu': cp_ispu,
                })
            new_cp_idx += 1
            
            # Hadronic tracksters get dummy CPs
            for had in had_tracksters:
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx': new_cp_idx,
                    'element_idx': had['element_idx'],
                    'element_type': 4,
                    'cp_energy': 0.0,
                    'cp_pid': cp_pid,
                    'weight': 0.0,
                    'is_winner': False,
                    'cp_eta': cp_eta,
                    'ispu': cp_ispu,
                })
            new_cp_idx += 1
        
        # Default case: single CP linking to all elements
        else:
            all_elements = had_tracksters
            total_shared = sum(elem['shared_energy'] for elem in all_elements)
            
            for elem in all_elements:
                split_fraction = elem['shared_energy'] / total_shared if total_shared > 0 else 1.0/len(all_elements)
                split_energy = cp_energy * split_fraction
                
                split_cps.append({
                    'original_idx': cp_idx,
                    'new_idx': new_cp_idx,
                    'element_idx': elem['element_idx'],
                    'element_type': elem['element_type'],
                    'cp_energy': split_energy,
                    'cp_pid': cp_pid,
                    'weight': split_fraction,
                    'is_winner': True,
                    'cp_eta': cp_eta,
                    'ispu': cp_ispu,
                })
            new_cp_idx += 1

    return split_cps

def make_graph(ev, iev):
    """Create a directed graph for one event."""
    g = nx.DiGraph()
    
    n_ts = len(ev["ts_energy"])           # Hadronic tracksters
    n_tracks = len(ev["track_pt"])         # Tracks
    n_ts_eg = len(ev["tsEG_energy"])       # EM tracksters
    
    # Add hadronic tracksters (type 4)
    for its in range(n_ts):
        energy = ev["ts_energy"][its]
        eta = ev["ts_eta"][its]
        phi = ev["ts_phi"][its]
        pt = ev["ts_pt"][its]
        
        theta = 2 * math.atan(math.exp(-eta))
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = energy * math.cos(theta)
        
        node_id = ("elem", its)
        g.add_node(
            node_id,
            typ=4,
            pt=float(pt),
            energy=float(energy),
            eta=float(eta),
            phi=float(phi),
            charge=0.0,
            px=float(px),
            py=float(py),
            pz=float(pz),
        )
    
    # Add tracks (type 1)
    for itrk in range(n_tracks):
        node_id = ("elem", n_ts + itrk)
        pz = ev["track_pt"][itrk] * math.sinh(ev["track_eta"][itrk]) if abs(ev["track_eta"][itrk]) < 10 else 0
        
        g.add_node(
            node_id,
            typ=1,
            pt=float(ev["track_pt"][itrk]),
            energy=float(ev["track_p"][itrk]),
            eta=float(ev["track_eta"][itrk]),
            phi=float(ev["track_phi"][itrk]),
            layer=0,
            charge=float(ev["track_charge"][itrk]),
            px=float(ev["track_pt"][itrk] * math.cos(ev["track_phi"][itrk])),
            py=float(ev["track_pt"][itrk] * math.sin(ev["track_phi"][itrk])),
            pz=float(pz)
        )
    
    # Add EM tracksters (types 2 and 3)
    for its in range(n_ts_eg):
        node_id = ("elem", n_ts + n_tracks + its)
        energy = ev["tsEG_energy"][its]
        eta = ev["tsEG_eta"][its]
        phi = ev["tsEG_phi"][its]
        pt = ev["tsEG_pt"][its]
        
        theta = 2 * math.atan(math.exp(-eta))
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = energy * math.cos(theta)
        
        # Type will be set later based on CP matching
        # For now, we add a temporary type
        g.add_node(
            node_id,
            typ=-1,  # Will be updated when adding edges
            pt=float(pt),
            energy=float(energy),
            eta=float(eta),
            phi=float(phi),
            charge=0.0,
            px=float(px),
            py=float(py),
            pz=float(pz),
        )
    # Add muon elements (type 5) from candidates
    muon_offset = n_ts + n_tracks + n_ts_eg
    for ipf in range(len(ev["tcan_pt"])):
        if abs(ev["candidate_pdgId"][ipf]) == 13:
            node_id = ("elem", muon_offset + ipf)
            g.add_node(
                node_id,
                typ=5,
                pt=float(ev["tcan_pt"][ipf]),
                energy=float(ev["tcan_energy"][ipf]),
                eta=float(ev["tcan_eta"][ipf]),
                phi=float(ev["tcan_phi"][ipf]),
                charge=get_charge(ev["candidate_pdgId"][ipf]),
                px=float(ev["tcan_pt"][ipf] * math.cos(ev["tcan_phi"][ipf])),
                py=float(ev["tcan_pt"][ipf] * math.sin(ev["tcan_phi"][ipf])),
                pz=float(ev["tcan_pt"][ipf] * math.sinh(ev["tcan_eta"][ipf])),
            )
    
    # Collect all connections
    had_connections = collect_hadronic_connections(ev)
    em_connections = collect_em_connections(ev)
    track_connections = collect_track_connections(ev)
    muon_connections = collect_muon_connections(ev)
    
    all_connections = had_connections + em_connections + track_connections + muon_connections
    
    # Split CPs
    split_cps = split_caloparticles(all_connections, ev)


    # Add split CP nodes and edges
    # First, group split_cps by original_idx to calculate totals
    cp_totals = defaultdict(lambda: {'track': 0.0, 'cluster': 0.0})

    # First pass: calculate totals for each original CP
    for split_cp in split_cps:
        original_idx = split_cp['original_idx']
        element_type = split_cp['element_type']
        weight = split_cp['weight']
    
        if element_type == 1:  # Track
            cp_totals[original_idx]['track'] += weight
        else:  # All other types (2,3,4,5) are cluster-like
            cp_totals[original_idx]['cluster'] += weight

    # Second pass: add nodes with calculated values
    for split_cp in split_cps:
        cp_idx = split_cp['new_idx']
        original_idx = split_cp['original_idx']
        element_idx = split_cp['element_idx']
        element_type = split_cp['element_type']
    
        if original_idx < len(ev["simcan_energy"]):
            cp_energy = split_cp['cp_energy']
            cp_eta = ev["simcan_eta"][original_idx]
            cp_phi = ev["simcan_phi"][original_idx]
            cp_pid = ev["simcan_pdgid"][original_idx]
            weight = split_cp['weight']
            
            theta = 2 * math.atan(math.exp(-cp_eta))
            cp_pt = cp_energy * math.sin(theta)
            
            node_id = ("cp", cp_idx)
        
            # Get calculated totals for this original CP
            track_total = cp_totals[original_idx]['track']
            cluster_total = cp_totals[original_idx]['cluster']
            
            g.add_node(
                node_id,
                pid=abs(int(cp_pid)),
                pt=float(cp_pt),
                eta=float(cp_eta),
                sin_phi=math.sin(cp_phi),
                cos_phi=math.cos(cp_phi),
                energy=float(cp_energy),
                ispu=float(split_cp['ispu']),
                generatorStatus=0,
                simulatorStatus=1,
                cp_to_track=float(track_total),
                cp_to_cluster=float(cluster_total),
                jet_idx=-1,
                px=float(cp_pt * math.cos(cp_phi)),
                py=float(cp_pt * math.sin(cp_phi)),
                pz=float(cp_energy * math.cos(theta)),
                charge=get_charge(cp_pid),
                original_cp_idx=original_idx
            )
        
            elem_node = ("elem", element_idx)
            if elem_node in g.nodes:
                if element_type in [2, 3] and g.nodes[elem_node]["typ"] == -1:
                    g.nodes[elem_node]["typ"] = element_type
                    
                g.add_edge(node_id, elem_node, weight=weight, is_winner=split_cp['is_winner'])

    # Add candidates (for comparison only)
    n_tcan = len(ev["tcan_pt"])
    for ipf in range(n_tcan):
        node_id = ("pfcand", ipf)
        pid = ev["candidate_pdgId"][ipf] if ipf < len(ev["candidate_pdgId"]) else 211
        charge = get_charge(pid)
        
        g.add_node(
            node_id,
            pid=abs(int(pid)),
            charge=float(charge),
            pt=float(ev["tcan_pt"][ipf]),
            eta=float(ev["tcan_eta"][ipf]),
            sin_phi=math.sin(ev["tcan_phi"][ipf]),
            cos_phi=math.cos(ev["tcan_phi"][ipf]),
            energy=float(ev["tcan_energy"][ipf]),
            ispu=0.0,
            generatorStatus=0,
            simulatorStatus=0,
            cp_to_track=0.0,
            cp_to_cluster=0.0,
            jet_idx=-1,
        )
    
    # Link elements to candidates
    for cand_idx in range(n_tcan):
        cand_node = ("pfcand", cand_idx)
        
        # Get trackster indices for this candidate
        if cand_idx < len(ev["trkst_indcies"]):
            trackster_data = ev["trkst_indcies"][cand_idx]
            if hasattr(trackster_data, '__len__'):  
                trackster_indices = trackster_data
            else:
                trackster_indices = [trackster_data] if trackster_data != -1 else []
        else:
            trackster_indices = []
        
        # Get track indices for this candidate  
        if cand_idx < len(ev["trks_indcies"]):
            track_data = ev["trks_indcies"][cand_idx]
            if hasattr(track_data, '__len__'):  
                track_indices = track_data
            else:
                track_indices = [track_data] if track_data != -1 else []
        else:
            track_indices = []
        
        # Link to hadronic tracksters
        for ts_idx in trackster_indices:
            if ts_idx >= 0 and ts_idx < n_ts:
                elem_node = ("elem", ts_idx)
                if elem_node in g.nodes:
                    g.add_edge(elem_node, cand_node, weight=1.0)
        
        # Link to tracks
        for trk_idx in track_indices:
            if trk_idx >= 0 and trk_idx < n_tracks:
                elem_node = ("elem", n_ts + trk_idx)
                if elem_node in g.nodes:
                    g.add_edge(elem_node, cand_node, weight=1.0)
        
        # Link muon candidates to muon elements
        if abs(ev["candidate_pdgId"][cand_idx]) == 13:
            muon_node = ("elem", muon_offset + cand_idx)
            if muon_node in g.nodes:
                g.add_edge(muon_node, cand_node, weight=1.0)
                
    # adding edges and updating type
    if "genpar_pdgid" in ev and len(ev["genpar_pdgid"]) > 0:
        n_gen = len(ev["genpar_pdgid"])
        
        # First pass: add all gen particle nodes
        for igen in range(n_gen):
            # basic info
            pid = abs(int(ev["genpar_pdgid"][igen]))
            status = int(ev["genpar_status"][igen])
            pt = float(ev["genpar_pt"][igen]) if "genpar_pt" in ev else 0.0
            eta = float(ev["genpar_eta"][igen])
            phi = float(ev["genpar_phi"][igen])
            energy = float(ev["genpar_energy"][igen]) if "genpar_energy" in ev else 0.0
            mass = float(ev["genpar_mass"][igen])
            
            # Calculate px, py, pz from pt, eta, phi
            px = pt * math.cos(phi)
            py = pt * math.sin(phi)
            pz = pt * math.sinh(eta) if abs(eta) < 10 else 0
            
            node_id = ("gen", igen)
            g.add_node(
                node_id,
                pid=pid,
                pt=pt,
                eta=eta,
                phi=phi,
                energy=energy,
                mass=mass,
                status=status,
                px=px,
                py=py,
                pz=pz,
                num_daughters=0,  # Will be updated in second pass
                is_stable=(status == 1),  # stable
            )
        for igen in range(n_gen):
            mother_idx = ev["genpar_genPartIdxMother"][igen]
            if mother_idx >= 0 and mother_idx < n_gen:
                daughter_node = ("gen", igen)
                mother_node = ("gen", mother_idx)
                
                if mother_node in g.nodes and daughter_node in g.nodes:
                    # edge from mother to daughter
                    g.add_edge(mother_node, daughter_node)                    
                    # Increment daughter count for mother
                    g.nodes[mother_node]["num_daughters"] += 1

    return g
#Prepare normalized tables for one event
def prepare_normalized_table(g):
    all_elements = [n for n in g.nodes if n[0] == "elem"]
    
    # Sort elements by type (tracks first, then others)
    all_elements.sort(key=lambda x: (
        0 if g.nodes[x]["typ"] == 1 else 
        1 if g.nodes[x]["typ"] == 5 else  # muons next
        2 if g.nodes[x]["typ"] in [2, 3] else  # EM tracksters
        3,  # hadronic tracksters last
        x[1]
    ))
    
    Xelem = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in elem_branches],
    )
    Xelem.fill(0.0)
    
    ytarget = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in particle_feature_order],
    )
    ytarget.fill(0.0)
    
    ycand = np.recarray(
        (len(all_elements),),
        dtype=[(name, np.float32) for name in particle_feature_order],
    )
    ycand.fill(0.0)
    
    # Map elements to CPs (multiple CPs possible)
    elem_to_cps = defaultdict(list)
    for elem in all_elements:
        for pred in g.predecessors(elem):
            if pred[0] == "cp":
                elem_to_cps[elem].append(pred)
    
    # Map elements to candidates
    elem_to_cand = {}
    for elem in all_elements:
        for succ in g.successors(elem):
            if succ[0] == "pfcand":
                elem_to_cand[elem] = succ
                break
    
    # Fill Xelem
    for ielem, elem in enumerate(all_elements):
        for branch in elem_branches:
            if branch in g.nodes[elem]:
                Xelem[branch][ielem] = g.nodes[elem][branch]
    
    # Fill ytarget (merge multiple CPs if needed)
    for ielem, elem in enumerate(all_elements):
        cps = elem_to_cps.get(elem, [])
        
        if cps:
            # Multiple CPs - merge them with energy weighting
            total_energy = 0
            weighted_sum = {
                'pid': 0, 'charge': 0, 'pt': 0, 'eta': 0, 
                'sin_phi': 0, 'cos_phi': 0, 'energy': 0,
                'ispu': 0, 'cp_to_track': 0, 'cp_to_cluster': 0
            }
            
            for cp in cps:
                edge_data = g.edges[(cp, elem)]
                weight = edge_data.get('weight', 1.0)
                cp_energy = g.nodes[cp]["energy"] * weight
                total_energy += cp_energy
                
                for key in weighted_sum.keys():
                    if key in g.nodes[cp]:
                        weighted_sum[key] += g.nodes[cp][key] * cp_energy
            
            if total_energy > 0:
                for key in weighted_sum.keys():
                    weighted_sum[key] /= total_energy
                
                # Get majority pid
                pid_counts = defaultdict(int)
                for cp in cps:
                    pid_counts[g.nodes[cp]["pid"]] += 1
                majority_pid = max(pid_counts.items(), key=lambda x: x[1])[0]
                
                ytarget["pid"][ielem] = majority_pid
                ytarget["charge"][ielem] = weighted_sum['charge']
                ytarget["pt"][ielem] = weighted_sum['pt']
                ytarget["eta"][ielem] = weighted_sum['eta']
                ytarget["sin_phi"][ielem] = weighted_sum['sin_phi']
                ytarget["cos_phi"][ielem] = weighted_sum['cos_phi']
                ytarget["energy"][ielem] = total_energy
                ytarget["ispu"][ielem] = weighted_sum['ispu']
                ytarget["cp_to_track"][ielem] = weighted_sum['cp_to_track']
                ytarget["cp_to_cluster"][ielem] = weighted_sum['cp_to_cluster']
        else:
            # No CP - pileup/noise
            ytarget["pid"][ielem] = 0
            ytarget["charge"][ielem] = g.nodes[elem].get("charge", 0.0)
            ytarget["pt"][ielem] = 0.0
            ytarget["eta"][ielem] = g.nodes[elem]["eta"]
            ytarget["sin_phi"][ielem] = math.sin(g.nodes[elem]["phi"])
            ytarget["cos_phi"][ielem] = math.cos(g.nodes[elem]["phi"])
            ytarget["energy"][ielem] = 0.0
            ytarget["ispu"][ielem] = 1.0  # Flag as pileup
            ytarget["cp_to_track"][ielem] = 0.0
            ytarget["cp_to_cluster"][ielem] = 0.0
        
        ytarget["generatorStatus"][ielem] = 0
        ytarget["simulatorStatus"][ielem] = 1 if cps else 0
        ytarget["jet_idx"][ielem] = -1
    
    # Fill ycand
    for ielem, elem in enumerate(all_elements):
        cand = elem_to_cand.get(elem)
        if cand is not None:
            for feature in particle_feature_order:
                if feature in g.nodes[cand]:
                    ycand[feature][ielem] = g.nodes[cand][feature]
    
    return Xelem, ycand, ytarget

def process_file(input_file, output_file, num_events=-1, start_event=0):
    """Process all events in a file and save to pickle."""
    print(f"Opening {input_file}")
    
    # Open ROOT file
    tf = uproot.open(input_file)
    
    # Get trees
    trees = {}
    tree_names = {
        'tkst': 'ticlDumper/ticlTracksterLinks',
        'simtkst': 'ticlDumper/simtrackstersCP',
        'cand': 'ticlDumper/candidates',
        'simcan': 'ticlDumper/simTICLCandidate',
        'track': 'ticlDumper/tracks',
        'assoc': 'ticlDumper/associations',
        'tkstEG':'ticlDumper/ticlTracksterLinksSuperclusteringDNN',
        'genpar':'ticlDumper/genparticles'
    }
    
    for key, tree_name in tree_names.items():
        try:
            trees[key] = tf[tree_name]
        except:
            print(f"  WARNING: Could not open tree {tree_name}")
            trees[key] = None
    
    if trees['tkst'] is None:
        print(f"ERROR: Missing trackster tree in {input_file}!")
        return []
    
    # Get number of events
    total_events = trees['tkst'].num_entries
    if num_events == -1:
        num_events = total_events - start_event
    else:
        num_events = min(num_events, total_events - start_event)
    
    print(f"  Processing events {start_event} to {start_event + num_events - 1} out of {total_events}")
    
    all_data = []
    
    # Process events
    for iev in tqdm(range(start_event, start_event + num_events), desc=f"Processing {os.path.basename(input_file)}"):
        try:
            # Read event
            ev = read_event(trees, iev)
            
            # Create graph
            g = make_graph(ev, iev)
            
            # Prepare normalized tables
            Xelem, ycand, ytarget = prepare_normalized_table(g)
            
            # Collect data with original file info
            event_data = {
                "Xelem": Xelem,
                "ycand": ycand,
                "ytarget": ytarget,
                "event_idx": iev,
                "file_name": os.path.basename(input_file),
                "global_event_idx": len(all_data)  # Global index across all files
            }
            
            all_data.append(event_data)
            
        except Exception as e:
            print(f"\nERROR processing event {iev} in {input_file}: {e}")
            continue
    
    return all_data


def process_files(input_list, output_file, num_events=-1, events_per_file=-1):
    """Process multiple ROOT files from a list."""
    all_data = []
    total_events_processed = 0
    
    # Filter out comments and empty lines
    input_list = [f.strip() for f in input_list if f.strip() and not f.startswith('#')]
    
    # Use tqdm for files with proper formatting
    for input_file in tqdm(input_list, desc="Processing files", unit="file", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        # Determine how many events to process from this file
        if events_per_file > 0:
            file_num_events = events_per_file
        elif num_events > 0:
            remaining_events = num_events - total_events_processed
            if remaining_events <= 0:
                break
            file_num_events = remaining_events
        else:
            file_num_events = -1  # Process all events
        
        file_data = process_file_no_progress(input_file, file_num_events, start_event=0)
        
        if file_data:
            all_data.extend(file_data)
            total_events_processed += len(file_data)
            
            # Update postfix with event count
            #tqdm.write(f"  {os.path.basename(input_file)}: {len(file_data)} events")
        
        # Stop if we've reached the total event limit
        if num_events > 0 and total_events_processed >= num_events:
            tqdm.write(f"\nReached target of {num_events} events")
            break
    
    return all_data

def process_file_no_progress(input_file, num_events=-1, start_event=0):
    """Process events in a file WITHOUT per-event progress bar."""
    # Open ROOT file
    tf = uproot.open(input_file)
    
    # Get trees
    trees = {}
    tree_names = {
        'tkst': 'ticlDumper/ticlTracksterLinks',
        'simtkst': 'ticlDumper/simtrackstersCP',
        'cand': 'ticlDumper/candidates',
        'simcan': 'ticlDumper/simTICLCandidate',
        'track': 'ticlDumper/tracks',
        'assoc': 'ticlDumper/associations',
        'tkstEG':'ticlDumper/ticlTracksterLinksSuperclusteringDNN',
        'genpar':'ticlDumper/genparticles'
    }
    
    for key, tree_name in tree_names.items():
        try:
            trees[key] = tf[tree_name]
        except:
            trees[key] = None
    
    if trees['tkst'] is None:
        return []
    
    # Get number of events
    total_events = trees['tkst'].num_entries
    if num_events == -1:
        num_events = total_events - start_event
    else:
        num_events = min(num_events, total_events - start_event)
    
    all_data = []
    
    # Process events WITHOUT tqdm
    for iev in range(start_event, start_event + num_events):
        try:
            # Read event
            ev = read_event(trees, iev)
            
            # Create graph
            g = make_graph(ev, iev)
            
            # Prepare normalized tables
            Xelem, ycand, ytarget = prepare_normalized_table(g)
            pt_min = 1
            truth_jets = compute_truth_jets(g, pt_min=pt_min)

            stable_gen = []
            for node in g.nodes:
                if node[0] == "gen" and g.nodes[node].get("status", 0) == 1:
                    pid = g.nodes[node]["pid"]
                    if pid not in [12, 14, 16]:  # Skip neutrinos
                        stable_gen.append([
                            g.nodes[node]["pid"],
                            g.nodes[node]["pt"],
                            g.nodes[node]["eta"],
                            g.nodes[node]["phi"],
                            g.nodes[node]["energy"]
                        ])

            stable_gen_array = np.array(stable_gen) if stable_gen else np.array([])
            event_data = {
                "Xelem": Xelem,
                "ycand": ycand,
                "ytarget": ytarget,
                "genjet": truth_jets,
                "stable_gen": stable_gen_array,
                "event_idx": iev,
                "file_name": os.path.basename(input_file),
                "global_event_idx": len(all_data)
            }
            all_data.append(event_data)
        except Exception as e:
             print(f"\nERROR in event {iev}: {e}")
             traceback.print_exc()
             continue
    
    return all_data
def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Process TICL graph data')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input ROOT file (.root) or text file with list of ROOT files (.txt)')
    parser.add_argument('--output', type=str, default='ticl_graph_data.pkl', 
                       help='Output pickle file')
    parser.add_argument('--num-events', type=int, default=-1, 
                       help='Total number of events to process across all files (-1 for all)')
    parser.add_argument('--events-per-file', type=int, default=-1,
                       help='Number of events to process per file (-1 for all)')
    parser.add_argument('--max-files', type=int, default=-1,
                       help='Maximum number of files to process (-1 for all)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} not found!")
        return
    
    # Determine if input is a single ROOT file or a list of files
    input_files = []
    
    if args.input.endswith('.root'):
        input_files = [args.input]
        print(f"Processing single ROOT file: {args.input}")
        
    elif args.input.endswith('.txt'):
        print(f"Reading file list from: {args.input}")
        with open(args.input, 'r') as f:
            input_files = [line.strip() for line in f]
        
        # Filter for ROOT files
        input_files = [f for f in input_files if f.endswith('.root')]
        
        if args.max_files > 0:
            input_files = input_files[:args.max_files]
        
        print(f"Found {len(input_files)} ROOT files to process")
        if len(input_files) == 0:
            print("ERROR: No ROOT files found in the text file!")
            return
            
    else:
        print(f"ERROR: Input file must be .root or .txt, got {args.input}")
        return
    
    # Verify all files exist
    missing_files = []
    for f in input_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"ERROR: {len(missing_files)} files not found:")
        for f in missing_files[:5]:
            print(f"  {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files)-5} more")
        return
    
    print(f"\nStarting processing of {len(input_files)} files...")
    
    # Process files
    all_data = process_files(input_files, args.output, args.num_events, args.events_per_file)
    
    # Save to pickle
    if all_data:
        print(f"\n{'='*60}")
        print(f"Saving {len(all_data)} events from {len(input_files)} files to {args.output}")
        
        with open(args.output, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"\nProcessing complete!")
        print(f"  Total events processed: {len(all_data)}")
        print(f"  Files processed: {len(input_files)}")
        print(f"  Output file: {args.output}")
        
        # Summary statistics
        if len(all_data) > 0:
            total_elements = sum(len(event['Xelem']) for event in all_data)
            total_tracks = sum(np.sum(event['Xelem']['typ'] == 1) for event in all_data)
            total_tracksters = sum(np.sum(event['Xelem']['typ'] == 4) for event in all_data)
            total_cp = sum(np.sum(event['ytarget']['pid'] > 0) for event in all_data)
            
            print(f"\nSummary statistics:")
            print(f"  Average elements per event: {total_elements/len(all_data):.1f}")
            print(f"  Average tracks per event: {total_tracks/len(all_data):.1f}")
            print(f"  Average tracksters per event: {total_tracksters/len(all_data):.1f}")
            print(f"  Average CaloParticles per event: {total_cp/len(all_data):.1f}")
    else:
        print("ERROR: No events were processed!")

if __name__ == "__main__":
    main()
'''
#For debugging only
#Process multiple ROOT files from a list.
def process_files(input_list, output_file, num_events=-1, events_per_file=-1):
    all_data = []
    total_events_processed = 0
    
    for i, input_file in enumerate(input_list):
        input_file = input_file.strip()
        if not input_file or input_file.startswith('#'):
            continue
            
        print(f"\nProcessing file {i+1}/{len(input_list)}: {input_file}")
        
        # Determine how many events to process from this file
        if events_per_file > 0:
            file_num_events = events_per_file
        elif num_events > 0:
            remaining_events = num_events - total_events_processed
            if remaining_events <= 0:
                break
            file_num_events = remaining_events
        else:
            file_num_events = -1  # Process all events
        
        file_data = process_file(input_file, output_file, file_num_events, start_event=0)
        
        if file_data:
            all_data.extend(file_data)
            total_events_processed += len(file_data)
            print(f"  Added {len(file_data)} events, total so far: {total_events_processed}")
        
        # Stop if we've reached the total event limit
        if num_events > 0 and total_events_processed >= num_events:
            print(f"Reached target of {num_events} events")
            break
    
    return all_data

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Process TICL graph data')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input ROOT file (.root) or text file with list of ROOT files (.txt)')
    parser.add_argument('--output', type=str, default='ticl_graph_data.pkl', 
                       help='Output pickle file')
    parser.add_argument('--num-events', type=int, default=-1, 
                       help='Total number of events to process across all files (-1 for all)')
    parser.add_argument('--events-per-file', type=int, default=-1,
                       help='Number of events to process per file (-1 for all)')
    parser.add_argument('--max-files', type=int, default=-1,
                       help='Maximum number of files to process (-1 for all)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} not found!")
        return
    
    # Determine if input is a single ROOT file or a list of files
    input_files = []
    
    if args.input.endswith('.root'):
        # Single ROOT file
        input_files = [args.input]
        print(f"Processing single ROOT file: {args.input}")
        
    elif args.input.endswith('.txt'):
        # Text file containing list of ROOT files
        print(f"Reading file list from: {args.input}")
        with open(args.input, 'r') as f:
            input_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Filter for ROOT files
        input_files = [f for f in input_files if f.endswith('.root')]
        
        if args.max_files > 0:
            input_files = input_files[:args.max_files]
        
        print(f"Found {len(input_files)} ROOT files to process")
        if len(input_files) == 0:
            print("ERROR: No ROOT files found in the text file!")
            return
            
    else:
        print(f"ERROR: Input file must be .root or .txt, got {args.input}")
        return
    
    # Verify all files exist
    missing_files = []
    for f in input_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"ERROR: {len(missing_files)} files not found:")
        for f in missing_files[:5]:  # Show first 5 missing files
            print(f"  {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files)-5} more")
        return
    
    # Process files
    all_data = process_files(input_files, args.output, args.num_events, args.events_per_file)
    
    # Save to pickle
    if all_data:
        print(f"\nSaving {len(all_data)} events from {len(input_files)} files to {args.output}")
        with open(args.output, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"\nProcessing complete!")
        print(f"  Total events processed: {len(all_data)}")
        print(f"  Files processed: {len(input_files)}")
        print(f"  Output file: {args.output}")
        
        # Summary statistics
        total_elements = sum(len(event['Xelem']) for event in all_data)
        total_tracks = sum(np.sum(event['Xelem']['typ'] == 1) for event in all_data)
        total_tracksters = sum(np.sum(event['Xelem']['typ'] == 4) for event in all_data)
        total_cp = sum(np.sum(event['ytarget']['pid'] > 0) for event in all_data)
        
        print(f"\nSummary statistics:")
        print(f"  Average elements per event: {total_elements/len(all_data):.1f}")
        print(f"  Average tracks per event: {total_tracks/len(all_data):.1f}")
        print(f"  Average tracksters per event: {total_tracksters/len(all_data):.1f}")
        print(f"  Average CaloParticles per event: {total_cp/len(all_data):.1f}")
    else:
        print("ERROR: No events were processed!")

if __name__ == "__main__":
    main()



'''
