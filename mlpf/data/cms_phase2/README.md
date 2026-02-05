https://github.com/jpata/particleflow/wiki/CMS

# TICL data Preprocessing for MLPF

## Overview
This README contains the preprocessing pipeline to convert TICLDumper reco data into graph-structured inputs compatible with MLPF. 

## Quick Start

### 1. Environment Setup (any uptodate cmssw release)
cmsrel CMSSW_16_0_0
cd CMSSW_16_0_0/src
cmsenv
git cms-init
git cms-addpkg RecoHGCal
(Optional for more data input from TICLDumper)
git remote add moanwar https://github.com/moanwar/cmssw
git fetch moanwar
git cherry-pick ae98c9b32ba

### 2. Run TICL v5 workflows :
For TICLv5 workflows, there are a few workflows that should run with `*.203` to activate the TICLv5 procModifier. These workflows mainly include particle gun sim, from-vertex samples, and CloseBy samples from the HGCAL surface. For track inclusion, the from-vertex samples should be used.

Example :
29888.203_SinglePiPt25Eta1p7_2p7+Run4D110PU_ticl_v5 (single pion from vertex  + pu)
29688.203_SinglePiPt25Eta1p7_2p7+Run4D110_ticl_v5 (single pion from vertex)

**Example Workflows:**
- `29688.203_SinglePiPt25Eta1p7_2p7+Run4D110_ticl_v5` (single pion from vertex, no pileup)
- `29888.203_SinglePiPt25Eta1p7_2p7+Run4D110PU_ticl_v5` (single pion from vertex, with pileup)

**To Run the wf via runTheMatrix:**
runTheMatrix.py -w upgrade -l 29688.203 -j 0
cd 29688.203_SinglePiPt25Eta1p7_2p7+Run4D110_ticl_v5
Edit step3.py and add after 'from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn':
from RecoHGCal.TICL.customiseTICLFromReco import customiseTICLForDumper
process = customiseTICLForDumper(process, histoName = "histo.root")
process.ticlDumper.saveLCs = True
process.ticlDumper.saveTICLCandidate = True
process.ticlDumper.saveSimTICLCandidate = True
process.ticlDumper.saveTracks = True
process.ticlDumper.saveSuperclustering = False
process.ticlDumper.saveRecoSuperclusters = False
Then run: cmsRun step3.py

### 3. Process Output to Graph Format
python3 ticl_graph_preprocess.py --input histo.root --output ticl_graph.pkl --num-events 10

## Graph Structure

The TICL graph is a directed graph G = (Nodes, Edges) designed to match MLPF input conventions:

NODES:

1. Elements (elem) - Reco Inputs
   - type=1: Tracks (TrackBase) with features: pt, eta, phi, charge, px, py, pz, layer, etc.
   - type=4: Tracksters (TICL clusters) with features: energy, eta, phi, pt, px, py, pz, etc.
   All features defined in elem_branches list.

2. CaloParticles (cp) - Sim Truth
   - Real Split CPs (is_winner=True): Winning elements receive full CP energy with kinematics, PID, charge
   - Dummy Split CPs (is_winner=False): Losing elements receive PID only (0 energy)
   Features defined in particle_feature_order list.

3. Candidates (pfcand) - Reco inputs
   - PID: TICLCandidate PDG ID from candidate_pdgId
   - Charge: TICLCandidate charge(based on track asscoaution in candidate or not)
   - Kinematics: pt, eta, phi, energy from TICLCandidate
   - PU Flag: ispu=0.0 (unknown for reconstruction)

EDGES:

1. cp -> elem - Truth-to-Reco associations
   - weight: 1.0 (after splitting, each CP fully belongs to one element)
   - cp_fraction: Fraction of original CP's energy
   - is_winner: Whether element receives CP energy (True) or just PID (False)

2. elem -> pfcand - reco-to-reco associations
   - weight: 1.0 (binary association)
   - Logic: PF-style: charged particles -> tracks, neutral particles -> highest energy trackster

## Data Processing 
Step 1: Read TICL Dumper Output
- Extract tracks, tracksters, CaloParticles, and TICLCandidates from histo.root files
- Load association matrices between truth and reconstruction

Step 2: Build Element-Truth Connections
- Collect all CP-element connections via:
  * Trackster CP links (energy-weighted)
  * Track CP links (from SimTICLcandidates)
- Find shared energy fractions and matching scores

Step 3: Split multi-associated CaloParticles
Important Step: When a CaloParticle matches multiple elements:
For charged particles (PID = 211, 13, 11, 321):
if tracks_present:
    # Split energy among tracks only
    # Tracksters receive dummy CPs (energy=0, PID preserved)
else:
    # Split energy among all matched elements
For neutral particles:
# Split energy among all matched elements proportionally to shared energy fraction 

Step 4: Construct Directed Graph
1. add element nodes (tracks + tracksters) with kinematic features
2. add split CP nodes with truth information
3. add candidate nodes with reco information
4. connect CP->elem (truth associations)
5. connect elem->cand (reco associations)

Step 5: Create Normalized Output Tables
Xelem    # [n_elements x n_elem_features] - Input features
ytarget  # [n_elements x n_particle_features] - Truth labels (from CPs)
ycand    # [n_elements x n_particle_features] - Baseline PF labels

## 3. Run the Preprocessing
python3 postprocessing_ticl.py --input histo.root --output ticl_graph.pkl --num-events 100