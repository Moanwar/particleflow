import os
import os.path as osp
from glob import glob
import tqdm
import uproot
import awkward as ak
import numpy as np
import ROOT
from array import array


def load_branch_with_highest_cycle(self, file, branch_name):
    all_keys = file.keys()
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    return file[highest_cycle_key]


def plot_connection_scores(histo_path, output_root_file):
    """
    Process ROOT files and create plots for connection scores and shared energy
    """
    files = glob(f"{histo_path}/*.root")[:20]
    print(f"Found {len(files)} ROOT files")
    
    # Create output ROOT file
    out_file = ROOT.TFile(output_root_file, "RECREATE")
    
    # Define histogram configuration
    nbins = 100
    
    # Create 1D histograms
    histograms_1d = {
        "ncandidate": ROOT.TH1F("ncandidate", "# Candidate;ncandidate;Count", 10, 0, 10),
        "raw_energy": ROOT.TH1F("raw_energy", "Raw Energy;Energy;Count", nbins, 0, 1000),
        "gen_energy": ROOT.TH1F("gen_energy", "Gen Energy;Energy;Count", nbins, 0, 1000),
        "raw_response": ROOT.TH1F("raw_response", "Raw Response;Response;Count", 20, 0, 2),
        "reg_response": ROOT.TH1F("reg_response", "Reg Response;Response;Count", 20, 0, 2),
        "recoToSim_score": ROOT.TH1F("recoToSim_score", "RecoToSim Score;Score;Count", nbins, 0, 1),
        "simToReco_score": ROOT.TH1F("simToReco_score", "SimToReco Score;Score;Count", nbins, 0, 1),
        "shared_energy_recoToSim": ROOT.TH1F("shared_energy_recoToSim", "Shared Energy (RecoToSim);Energy;Count", nbins, 0, 1),
        "shared_energy_simToReco": ROOT.TH1F("shared_energy_simToReco", "Shared Energy (SimToReco);Energy;Count", nbins, 0, 1)
    }
    
    # Create 2D histograms
    histograms_2d = {
        "resp_vs_GenE": ROOT.TH2F("resp_vs_GenE", "Reg Response vs Gen Energy; Response; Gen Energy",
                                  nbins, 0, 600, 20, 0, 3),
        "score_correlation": ROOT.TH2F("score_correlation", "Score Correlation;RecoToSim Score;SimToReco Score", 
                                      nbins, 0, 1, nbins, 0, 1),
        "recoToSimscore_vs_energy": ROOT.TH2F("recoToSimscore_vs_energy", "recoToSimScore vs Shared Energy;recoToSimScore;Shared Energy", 
                                             nbins, 0, 1, nbins, 0, 1),
        "simToRecoscore_vs_energy": ROOT.TH2F("simToRecoscore_vs_energy", "simToRecoScore vs Shared Energy;simToRecoScore;Shared Energy", 
                                             nbins, 0, 1, nbins, 0, 1),
        "recoToSimFrac_vs_simToRecoFrac": ROOT.TH2F("recoToSimFrac_vs_simToRecoFrac", "RecoToSim frac vs SimToReco frac Energy;RecoToSim fracE ;SimToReco fracE", 
                                                   nbins, 0, 1, nbins, 0, 1),
        "nCand_vs_recoToSim_score": ROOT.TH2F("nCand_vs_recoToSim_score", "nCandidate vs RecoToSim Score; nCandidate; RecoToSim Score",
                                              10, 0, 10, nbins, 0, 1),
        "nCand_vs_simToReco_score": ROOT.TH2F("nCand_vs_simToReco_score", "nCandidate vs SimToReco Score; nCandidate; SimToReco Score",
                                              10, 0, 10, nbins, 0, 1),
        "nCand_vs_shared_energy": ROOT.TH2F("nCand_vs_shared_energy", "nCandidate vs Shared Energy; nCandidate; Shared Energy",
                                            10, 0, 10, nbins, 0, 1)
    }
    
    # Score thresholds
    sim2reco_threshold = 0.9
    reco2sim_threshold = 0.6
    
    # Process each file
    for file_index, file_path in enumerate(files):
        print(f"Processing file {file_index + 1}/{len(files)}: {file_path}")
        
        try:
            # Open file and load data
            file = uproot.open(file_path)
            
            associations = file['ticlDumper/associations']
            tracksters = file['ticlDumper/ticlTracksterLinks']
            allsimtrackstersCP = file['ticlDumper/simtrackstersCP']
            allsimtrackstersSC = file['ticlDumper/simtrackstersSC']
            
            # Load arrays
            all_tracksters = tracksters.arrays(['raw_energy', "vertices_indexes", 'regressed_energy'])
            calo_particle = allsimtrackstersCP.arrays(['raw_energy', 'regressed_energy', 'barycenter_eta', 'pdgID'])
            simTracksters = allsimtrackstersCP.arrays(['NTracksters'])
            
            connections = associations.arrays([
                'ticlTracksterLinks_recoToSim_CP_sharedE',
                'ticlTracksterLinks_recoToSim_CP_score',
                'ticlTracksterLinks_recoToSim_CP',
                'ticlTracksterLinks_simToReco_CP_score',
                'ticlTracksterLinks_simToReco_CP_sharedE',
                'ticlTracksterLinks_simToReco_CP',
                'ticlTracksterLinks_recoToSim_SC_sharedE',
                'ticlTracksterLinks_recoToSim_SC_score',
                'ticlTracksterLinks_recoToSim_SC',
                'ticlTracksterLinks_simToReco_SC_score',
                'ticlTracksterLinks_simToReco_SC_sharedE',
                'ticlTracksterLinks_simToReco_SC'
            ])
            
            # Create mask for events with tracksters
            event_with_tracksters = [True for _ in range(len(all_tracksters.raw_energy))]
            connection_filtered = connections[event_with_tracksters]
            
            # Mask simToReco scores below threshold
            masked_simToReco = ak.mask(connection_filtered.ticlTracksterLinks_simToReco_CP_score,
                                       connection_filtered.ticlTracksterLinks_simToReco_CP_score < sim2reco_threshold)
            
            # Get indices of best reco tracksters
            best_reco_idx = ak.Array([
                [ak.where(~ak.is_none(inner))[0].tolist() for inner in outer] 
                for outer in masked_simToReco
            ])
            
            num_events = len(all_tracksters.vertices_indexes)
            
            # Process each event
            for event_idx in range(num_events):
                # Process each sim trackster
                for simts_idx in range(len(calo_particle[event_idx].raw_energy)):
                    # Apply selection cuts
                    if calo_particle[event_idx].regressed_energy[simts_idx] < 1:
                        continue
                    if simTracksters[event_idx].NTracksters < 1:
                        continue
                    if len(best_reco_idx) == 0:
                        continue
                    
                    # Get simToReco information
                    simToRecoScore = connection_filtered.ticlTracksterLinks_simToReco_CP_score[event_idx, simts_idx]
                    simToRecoE = connection_filtered.ticlTracksterLinks_simToReco_CP_sharedE[event_idx, simts_idx]
                    tsindex = connection_filtered.ticlTracksterLinks_simToReco_CP[event_idx, simts_idx]
                    tsEnergy = all_tracksters[event_idx].raw_energy[tsindex]
                    
                    # Calculate fractional energy
                    mask_score = simToRecoScore < sim2reco_threshold
                    frac_energy = simToRecoE[mask_score] / tsEnergy[mask_score]
                    
                    # Fill shared energy histogram for simToReco
                    for frac in frac_energy:
                        if frac is not None:
                            histograms_1d["shared_energy_simToReco"].Fill(frac)
                    
                    # Get best trackster indices
                    best_trackster_indices = best_reco_idx[event_idx, simts_idx]
                    nCount = 0
                    
                    # Process each best trackster
                    for idx in best_trackster_indices:
                        trackster_idx = connection_filtered.ticlTracksterLinks_simToReco_CP[event_idx, simts_idx, idx]
                        simScore = connection_filtered.ticlTracksterLinks_simToReco_CP_score[event_idx, simts_idx, idx]
                        RecoTosimScore = connection_filtered.ticlTracksterLinks_recoToSim_CP_score[event_idx, trackster_idx]
                        recoToSimE = connection_filtered.ticlTracksterLinks_recoToSim_CP_sharedE[event_idx, trackster_idx]
                        trackstersE = all_tracksters.raw_energy[event_idx, trackster_idx]
                        trackstersRegE = all_tracksters.regressed_energy[event_idx, trackster_idx]
                        
                        # Apply reco2sim threshold
                        reco_mask = RecoTosimScore < reco2sim_threshold
                        recoToSimE_filtered = recoToSimE[reco_mask]
                        energyFrac = recoToSimE_filtered / trackstersE
                        
                        # Calculate additional quantities
                        genEn = calo_particle.raw_energy[event_idx, simts_idx]
                        contamination = abs(recoToSimE_filtered - trackstersE) / genEn
                        genEnFrac = recoToSimE_filtered / genEn
                        raw_res = trackstersE / genEn
                        reg_res = trackstersRegE / genEn
                        RecoTosimScore_filtered = RecoTosimScore[reco_mask]
                        
                        # Fill histograms for valid entries
                        for score, energy in zip(RecoTosimScore_filtered, energyFrac):
                            if (score is not None and energy is not None and 
                                genEnFrac[0] > 0.5 and contamination[0] < 0.1):
                                
                                nCount += 1
                                
                                # Fill 1D histograms
                                histograms_1d["simToReco_score"].Fill(simScore)
                                histograms_1d["recoToSim_score"].Fill(score)
                                histograms_1d["shared_energy_recoToSim"].Fill(energy)
                                histograms_1d["raw_energy"].Fill(trackstersE)
                                histograms_1d["gen_energy"].Fill(genEn)
                                histograms_1d["raw_response"].Fill(raw_res)
                                histograms_1d["reg_response"].Fill(reg_res)
                                
                                # Fill 2D histograms
                                histograms_2d["score_correlation"].Fill(score, simScore)
                                histograms_2d["recoToSimscore_vs_energy"].Fill(score, energy)
                                histograms_2d["simToRecoscore_vs_energy"].Fill(simScore, genEnFrac[0])
                                histograms_2d["recoToSimFrac_vs_simToRecoFrac"].Fill(contamination[0], genEnFrac[0])
                                histograms_2d["nCand_vs_recoToSim_score"].Fill(nCount, score)
                                histograms_2d["nCand_vs_simToReco_score"].Fill(nCount, simScore)
                                histograms_2d["nCand_vs_shared_energy"].Fill(nCount, energy)
                                histograms_2d["resp_vs_GenE"].Fill(genEn, reg_res)
                    
                    # Fill ncandidate histogram for this sim trackster
                    histograms_1d["ncandidate"].Fill(nCount)
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # Write all histograms to file
    out_file.cd()
    
    # Write 1D histograms
    for hist_name, hist in histograms_1d.items():
        hist.Write()
    
    # Write 2D histograms
    for hist_name, hist in histograms_2d.items():
        hist.Write()
    
    # Create and save canvas with key plots
    canvas = ROOT.TCanvas("canvas", "Connection Scores", 800, 600)
    canvas.Divide(2, 2)
    
    canvas.cd(1)
    histograms_1d["recoToSim_score"].Draw()
    
    canvas.cd(2)
    histograms_1d["simToReco_score"].Draw()
    
    canvas.cd(3)
    histograms_1d["shared_energy_recoToSim"].Draw()
    
    canvas.cd(4)
    histograms_1d["shared_energy_simToReco"].Draw()
    
    
    # Close the output file
    out_file.Close()
    
    print(f"Analysis complete. Results saved to {output_root_file}")


if __name__ == "__main__":
    # QUESTION 2: Variable names don't match - should this be fixed?
    input_part = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/Link_single_firstDisk/211/histo/"
    output_part = "connection_scores_part2_normal_211.root"
    plot_connection_scores(input_part, output_part)
