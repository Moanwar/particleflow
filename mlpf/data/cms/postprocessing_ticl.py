import uproot
import numpy as np
import networkx as nx
import argparse
import os
import math
import pickle
from tqdm import tqdm
from collections import defaultdict

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

def delta_r(e1, e2, p1, p2):
    delta = e1 - e2
    dphi = p1 - p2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return math.sqrt(delta * delta + dphi * dphi)

def get_charge(pid):
    abs_pid = abs(pid)
    if pid in [130, 22, 1, 2, 310]:
        return 0.0
    elif abs_pid in [11, 13]:  
        return -math.copysign(1.0, pid)
    elif abs_pid in [211, 321]:  
        return math.copysign(1.0, pid)
    else:
        return 0.0

def is_charged_particle(pid):
    abs_pid = abs(pid)
    return abs_pid in [211, 13, 11, 321]

def read_event(trees, iev):
    """Read all necessary data for one event."""
    ev = {}
    
    # Read all trees
    tkst_arrays = trees['tkst'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", "raw_pt"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["ts_energy"] = tkst_arrays["raw_energy"][0]
    ev["ts_pt"] = tkst_arrays["raw_pt"][0]
    ev["ts_eta"] = tkst_arrays["barycenter_eta"][0]
    ev["ts_phi"] = tkst_arrays["barycenter_phi"][0]
    
    simtkst_arrays = trees['simtkst'].arrays(
        ["raw_energy", "barycenter_eta", "barycenter_phi", "pdgID", "trackIdx"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["simtkst_energy"] = simtkst_arrays["raw_energy"][0]
    ev["simtkst_eta"] = simtkst_arrays["barycenter_eta"][0]
    ev["simtkst_phi"] = simtkst_arrays["barycenter_phi"][0]
    ev["simtkst_pdgid"] = simtkst_arrays["pdgID"][0]
    ev["simtkst_trackIdx"] = simtkst_arrays["trackIdx"][0]
    
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

    
    simcan_arrays = trees['simcan'].arrays(
        ["simTICLCandidate_pdgId", "simTICLCandidate_pt", "simTICLCandidate_eta", 
         "simTICLCandidate_phi", "simTICLCandidate_raw_energy", 
         "simTICLCandidate_tracks_in_candidate", "simTICLCandidate_simTracksterCPIndex"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["simcan_pdgid"] = simcan_arrays["simTICLCandidate_pdgId"][0]
    ev["simcan_pt"] = simcan_arrays["simTICLCandidate_pt"][0]
    ev["simcan_eta"] = simcan_arrays["simTICLCandidate_eta"][0]
    ev["simcan_phi"] = simcan_arrays["simTICLCandidate_phi"][0]
    ev["simcan_energy"] = simcan_arrays["simTICLCandidate_raw_energy"][0]
    ev["simcan_trkId"] = simcan_arrays["simTICLCandidate_tracks_in_candidate"][0]
    ev["simcan_simTracksterCPIndex"] = simcan_arrays["simTICLCandidate_simTracksterCPIndex"][0]

    track_arrays = trees['track'].arrays(
        ["track_pt", "track_p", "track_hgcal_eta", "track_hgcal_phi", "track_charge"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["track_pt"] = track_arrays["track_pt"][0]
    ev["track_p"] = track_arrays["track_p"][0]
    ev["track_eta"] = track_arrays["track_hgcal_eta"][0]
    ev["track_phi"] = track_arrays["track_hgcal_phi"][0]
    ev["track_charge"] = track_arrays["track_charge"][0]
    
    assoc_arrays = trees['assoc'].arrays(
        ["ticlTracksterLinks_recoToSim_CP_score", "ticlTracksterLinks_simToReco_CP_score",
         "ticlTracksterLinks_simToReco_CP_sharedE", "ticlTracksterLinks_recoToSim_CP",
         "ticlTracksterLinks_simToReco_CP"],
        entry_start=iev, entry_stop=iev+1, library="np"
    )
    ev["ticlTracksterLinks_recoToSim_CP_score"] = assoc_arrays["ticlTracksterLinks_recoToSim_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_score"] = assoc_arrays["ticlTracksterLinks_simToReco_CP_score"][0]
    ev["ticlTracksterLinks_simToReco_CP_sharedE"] = assoc_arrays["ticlTracksterLinks_simToReco_CP_sharedE"][0]
    ev["ticlTracksterLinks_recoToSim_CP"] = assoc_arrays["ticlTracksterLinks_recoToSim_CP"][0]
    ev["ticlTracksterLinks_simToReco_CP"] = assoc_arrays["ticlTracksterLinks_simToReco_CP"][0]

    return ev

def collect_cp_element_connections(ev):
    """Collect all CP-element connections."""
    connections = []
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    
    # Trackster connections
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
                
                cp_energy = ev["simtkst_energy"][sim_idx]
                element_energy = ev["ts_energy"][trackster_idx]
                cp_pid = ev["simtkst_pdgid"][sim_idx]
                
                elem_weight = shared_energy / element_energy if element_energy > 0 else 0
                cp_fraction = shared_energy / cp_energy if cp_energy > 0 else 0
                
                connections.append({
                    'cp_idx': sim_idx,
                    'cp_pid': cp_pid,
                    'cp_energy': cp_energy,
                    'element_idx': trackster_idx,
                    'element_type': 4,
                    'shared_energy': shared_energy,
                    'element_energy': element_energy,
                    'elem_weight': elem_weight,
                    'cp_fraction': cp_fraction,
                    'reco_score': reco_score,
                    'sim_score': sim_score,
                    'is_charged': is_charged_particle(cp_pid)
                })
    
    # Track connections from simcandidates
    for cp_idx in range(len(ev["simcan_trkId"])):
        track_indices = ev["simcan_trkId"][cp_idx]
        if len(track_indices) == 0:
            continue
            
        cp_energy = ev["simtkst_energy"][cp_idx]
        cp_pid = ev["simtkst_pdgid"][cp_idx]
        
        for track_idx in track_indices:
            if track_idx < 0 or track_idx >= n_tracks:
                continue
            
            track_p = ev["track_p"][track_idx]
            
            connections.append({
                'cp_idx': cp_idx,
                'cp_pid': cp_pid,
                'cp_energy': cp_energy,
                'element_idx': n_ts + track_idx,
                'element_type': 1,
                'shared_energy': track_p,
                'element_energy': track_p,
                'elem_weight': 1.0,
                'cp_fraction': track_p / cp_energy if cp_energy > 0 else 0,
                'reco_score': 0.0,
                'sim_score': 0.0,
                'is_charged': is_charged_particle(cp_pid)
            })
    
    return connections

def split_caloparticles_tracksters(connections, ev):
    """Split CPs that match multiple elements."""
    cp_groups = defaultdict(list)
    for conn in connections:
        cp_groups[conn['cp_idx']].append(conn)
    
    split_cps = []
    new_cp_idx = len(ev["simtkst_energy"])
    
    for cp_idx, conns in cp_groups.items():
        if len(conns) == 1:
            conn = conns[0]
            split_cps.append({
                'original_idx': cp_idx,
                'new_idx': cp_idx,
                'element_idx': conn['element_idx'],
                'cp_energy': conn['cp_energy'],
                'cp_pid': conn['cp_pid'],
                'cp_fraction': 1.0,
                'is_winner': True
            })
        else:
            cp_pid = conns[0]['cp_pid']
            cp_energy = conns[0]['cp_energy']
            is_charged = conns[0]['is_charged']
            
            tracks = [c for c in conns if c['element_type'] == 1]
            tracksters = [c for c in conns if c['element_type'] == 4]
            
            if is_charged and tracks:
                if len(tracks) == 1:
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx': new_cp_idx,
                        'element_idx': tracks[0]['element_idx'],
                        'cp_energy': cp_energy,
                        'cp_pid': cp_pid,
                        'cp_fraction': 1.0,
                        'is_winner': True
                    })
                    new_cp_idx += 1
                else:
                    total_shared = sum(t['shared_energy'] for t in tracks)
                    for track in tracks:
                        split_fraction = track['shared_energy'] / total_shared if total_shared > 0 else 1.0 / len(tracks)
                        split_energy = cp_energy * split_fraction
                        split_cps.append({
                            'original_idx': cp_idx,
                            'new_idx': new_cp_idx,
                            'element_idx': track['element_idx'],
                            'cp_energy': split_energy,
                            'cp_pid': cp_pid,
                            'cp_fraction': split_fraction,
                            'is_winner': True
                        })
                        new_cp_idx += 1
                
                for trackster in tracksters:
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx': new_cp_idx,
                        'element_idx': trackster['element_idx'],
                        'cp_energy': 0.0,
                        'cp_pid': cp_pid,
                        'cp_fraction': 0.0,
                        'is_winner': False
                    })
                    new_cp_idx += 1
            else:
                total_shared = sum(c['shared_energy'] for c in conns)
                for conn in conns:
                    split_fraction = conn['shared_energy'] / total_shared if total_shared > 0 else 1.0 / len(conns)
                    split_energy = cp_energy * split_fraction
                    split_cps.append({
                        'original_idx': cp_idx,
                        'new_idx': new_cp_idx,
                        'element_idx': conn['element_idx'],
                        'cp_energy': split_energy,
                        'cp_pid': cp_pid,
                        'cp_fraction': split_fraction,
                        'is_winner': True
                    })
                    new_cp_idx += 1
    
    return split_cps

def make_graph(ev, iev):
    """Create a directed graph for one event."""
    g = nx.DiGraph()
    
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    
    # Add tracksters as elements
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
    
    # Add tracks as elements
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
    
    # CP-element connections and splitting
    connections = collect_cp_element_connections(ev)
    split_cps = split_caloparticles_tracksters(connections, ev)
    
    # Add split CP nodes and edges
    for split_cp in split_cps:
        cp_idx = split_cp['new_idx']
        original_idx = split_cp['original_idx']
        element_idx = split_cp['element_idx']
        
        if original_idx < len(ev["simtkst_energy"]):
            cp_fraction = split_cp['cp_fraction']
            cp_energy = ev["simtkst_energy"][original_idx] * cp_fraction
            cp_eta = ev["simtkst_eta"][original_idx]
            cp_phi = ev["simtkst_phi"][original_idx]
            cp_pid = ev["simtkst_pdgid"][original_idx]
            
            theta = 2 * math.atan(math.exp(-cp_eta))
            cp_pt = cp_energy * math.sin(theta)
            
            node_id = ("cp", cp_idx)
            g.add_node(
                node_id,
                pid=abs(int(cp_pid)),
                pt=float(cp_pt),
                eta=float(cp_eta),
                sin_phi=math.sin(cp_phi),
                cos_phi=math.cos(cp_phi),
                energy=float(cp_energy),
                ispu=0.0,
                generatorStatus=0,
                simulatorStatus=1,
                cp_to_track=0.0,
                cp_to_cluster=1.0,
                jet_idx=-1,
                px=float(cp_pt * math.cos(cp_phi)),
                py=float(cp_pt * math.sin(cp_phi)),
                pz=float(cp_energy * math.cos(theta)),
                charge=get_charge(cp_pid),
                original_cp_idx=original_idx
            )
            
            elem_node = ("elem", element_idx)
            if elem_node in g.nodes:
                g.add_edge(node_id, elem_node, weight=1.0, cp_fraction=cp_fraction, is_winner=split_cp['is_winner'])
    
    # Add candidates
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
            # Use SAME LOGIC as debugger
            if hasattr(trackster_data, '__len__'):  # It's a list/array/STLVector
                trackster_indices = trackster_data
            else:  # It's a single integer
                trackster_indices = [trackster_data] if trackster_data != -1 else []
        else:
            trackster_indices = []
    
        # Get track indices for this candidate  
        if cand_idx < len(ev["trks_indcies"]):
            track_data = ev["trks_indcies"][cand_idx]
            # Use SAME LOGIC as debugger
            if hasattr(track_data, '__len__'):  # It's a list/array/STLVector
                track_indices = track_data
            else:  # It's a single integer
                track_indices = [track_data] if track_data != -1 else []
        else:
            track_indices = []
    
        # Link to tracksters
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
                
    return g

def prepare_normalized_table(g):
    """Prepare normalized tables for one event."""
    all_elements = [n for n in g.nodes if n[0] == "elem"]
    all_elements.sort(key=lambda x: (0 if g.nodes[x]["typ"] == 1 else 1, x[1]))
    
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
    
    # Map elements to CPs
    elem_to_cp = {}
    for elem in all_elements:
        preds = list(g.predecessors(elem))
        cp_preds = [p for p in preds if p[0] == "cp"]
        if cp_preds:
            elem_to_cp[elem] = cp_preds[0]
    
    # Map elements to candidates
    elem_to_cand = {}
    for elem in all_elements:
        succs = list(g.successors(elem))
        cand_succs = [s for s in succs if s[0] == "pfcand"]
        if cand_succs:
            elem_to_cand[elem] = cand_succs[0]
    
    # Fill Xelem
    for ielem, elem in enumerate(all_elements):
        for branch in elem_branches:
            if branch in g.nodes[elem]:
                Xelem[branch][ielem] = g.nodes[elem][branch]
    
    # Fill ytarget
    for ielem, elem in enumerate(all_elements):
        cp = elem_to_cp.get(elem)
        if cp is not None:
            edge_data = g.edges[(cp, elem)]
            is_winner = edge_data.get('is_winner', True)
            
            if is_winner:
                ytarget["pid"][ielem] = g.nodes[cp]["pid"]
                ytarget["charge"][ielem] = g.nodes[cp]["charge"]
                ytarget["pt"][ielem] = g.nodes[cp]["pt"]
                ytarget["eta"][ielem] = g.nodes[cp]["eta"]
                ytarget["sin_phi"][ielem] = g.nodes[cp]["sin_phi"]
                ytarget["cos_phi"][ielem] = g.nodes[cp]["cos_phi"]
                ytarget["energy"][ielem] = g.nodes[cp]["energy"]
                ytarget["ispu"][ielem] = g.nodes[cp]["ispu"]
                ytarget["generatorStatus"][ielem] = g.nodes[cp]["generatorStatus"]
                ytarget["simulatorStatus"][ielem] = g.nodes[cp]["simulatorStatus"]
                ytarget["cp_to_track"][ielem] = g.nodes[cp]["cp_to_track"]
                ytarget["cp_to_cluster"][ielem] = g.nodes[cp]["cp_to_cluster"]
                ytarget["jet_idx"][ielem] = g.nodes[cp]["jet_idx"]
            else:
                ytarget["pid"][ielem] = g.nodes[cp]["pid"]
                ytarget["charge"][ielem] = g.nodes[cp]["charge"]
                ytarget["pt"][ielem] = 0.0
                ytarget["eta"][ielem] = g.nodes[elem]["eta"]
                ytarget["sin_phi"][ielem] = math.sin(g.nodes[elem]["phi"])
                ytarget["cos_phi"][ielem] = math.cos(g.nodes[elem]["phi"])
                ytarget["energy"][ielem] = 0.0
                ytarget["ispu"][ielem] = 0.0
                ytarget["generatorStatus"][ielem] = 0
                ytarget["simulatorStatus"][ielem] = 1
                ytarget["cp_to_track"][ielem] = 0.0
                ytarget["cp_to_cluster"][ielem] = 1.0
                ytarget["jet_idx"][ielem] = -1
        else:
            ytarget["pid"][ielem] = 0
            ytarget["charge"][ielem] = g.nodes[elem].get("charge", 0.0)
            ytarget["pt"][ielem] = 0.0
            ytarget["eta"][ielem] = g.nodes[elem]["eta"]
            ytarget["sin_phi"][ielem] = math.sin(g.nodes[elem]["phi"])
            ytarget["cos_phi"][ielem] = math.cos(g.nodes[elem]["phi"])
            ytarget["energy"][ielem] = 0.0
            ytarget["ispu"][ielem] = 1.0
            ytarget["generatorStatus"][ielem] = 0
            ytarget["simulatorStatus"][ielem] = 0
            ytarget["cp_to_track"][ielem] = 0.0
            ytarget["cp_to_cluster"][ielem] = 0.0
            ytarget["jet_idx"][ielem] = -1
    
    # Fill ycand
    for ielem, elem in enumerate(all_elements):
        cand = elem_to_cand.get(elem)
        if cand is not None:
            for feature in particle_feature_order:
                if feature in g.nodes[cand]:
                    ycand[feature][ielem] = g.nodes[cand][feature]
    
    return Xelem, ycand, ytarget

def process_file(input_file, output_file, num_events=-1):
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
        'assoc': 'ticlDumper/associations'
    }
    
    for key, tree_name in tree_names.items():
        try:
            trees[key] = tf[tree_name]
        except:
            print(f"  WARNING: Could not open tree {tree_name}")
            trees[key] = None
    
    if trees['tkst'] is None:
        print("ERROR: Missing trackster tree!")
        return
    
    # Get number of events
    total_events = trees['tkst'].num_entries
    if num_events == -1 or num_events > total_events:
        num_events = total_events
    
    print(f"Processing {num_events} events out of {total_events}")
    
    all_data = []
    
    # Process events
    for iev in tqdm(range(num_events), desc="Processing events"):
        try:
            # Read event
            ev = read_event(trees, iev)
            
            # Create graph
            g = make_graph(ev, iev)
            
            # Prepare normalized tables
            Xelem, ycand, ytarget = prepare_normalized_table(g)
            
            # Collect data
            event_data = {
                "Xelem": Xelem,
                "ycand": ycand,
                "ytarget": ytarget,
                "event_idx": iev
            }
            
            all_data.append(event_data)
            
        except Exception as e:
            print(f"\nERROR processing event {iev}: {e}")
            continue
    
    # Save to pickle
    print(f"\nSaving {len(all_data)} events to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    # Statistics
    print(f"\nProcessing complete!")
    print(f"  Input file: {input_file}")
    print(f"  Events processed: {len(all_data)}/{num_events}")
    print(f"  Output file: {output_file}")

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Process TICL graph data')
    parser.add_argument('--input', type=str, required=True, help='Input ROOT file')
    parser.add_argument('--output', type=str, default='ticl_graph_data.pkl', help='Output pickle file')
    parser.add_argument('--num-events', type=int, default=-1, help='Number of events to process (-1 for all)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} not found!")
        return
    
    process_file(args.input, args.output, args.num_events)

if __name__ == "__main__":
    main()
