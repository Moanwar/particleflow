import bz2
import pickle
import datetime

import awkward as ak
import numpy as np
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

ELEM_LABELS_TICL = [0, 1, 2, 4]
ELEM_NAMES_TICL = ["NONE", "TRACK", "EM_TS", "HAD_TS"]

CLASS_LABELS_TICL = [0, 211, 130, 22, 11, 13]
CLASS_NAMES_TICL = ["none", "ch.had", "n.had", "gamma", "ele", "mu"]

X_FEATURES = [
    "typ_idx",
    "pt", "eta", "sin_phi", "cos_phi", "energy",
    "charge", "px", "py", "pz","em_energy", "bary_z", "nhits",
]

Y_FEATURES = [
    "typ_idx",
    "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
    "ispu", "generatorStatus", "simulatorStatus", "cp_to_track", "cp_to_cluster", "jet_idx",
]

NUM_SPLITS = 10


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0
    if pdgid in [22, 11, 13]:
        return pdgid
    if abs(charge) > 0:
        return 211
    return 130


def prepare_data_ticl(fn):
    Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias = [], [], [], [], [], [], []
    try:
        data = pickle.load(open(fn, "rb"))
    except Exception as e:
        print(f"Could not open file {fn}: {e}")
        return Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias

    for event in data:
        Xelem   = ak.Array(event["Xelem"])
        ytarget = ak.Array(event["ytarget"])
        ycand   = ak.Array(event["ycand"])

        # Add sin/cos phi to Xelem
        Xelem["sin_phi"] = np.sin(Xelem["phi"])
        Xelem["cos_phi"] = np.cos(Xelem["phi"])

        # Map typ to index
        Xelem["typ_idx"] = np.array(
            [ELEM_LABELS_TICL.index(int(i)) if int(i) in ELEM_LABELS_TICL else 0
             for i in Xelem["typ"]], dtype=np.float32)

        # Add generatorStatus as zeros (not available in TICL data)
        ytarget["generatorStatus"] = np.zeros(len(ytarget), dtype=np.float32)
        ycand["generatorStatus"]   = np.zeros(len(ycand),   dtype=np.float32)

        # Map pid to class index for ytarget
        pids_remapped = [map_pdgid_to_candid(abs(int(pid)), q)
                         for pid, q in zip(ytarget["pid"], ytarget["charge"])]
        ytarget["typ_idx"] = np.array(
            [CLASS_LABELS_TICL.index(pid) if pid in CLASS_LABELS_TICL else 0
             for pid in pids_remapped], dtype=np.float32)

        # Map pid to class index for ycand
        ycand["typ_idx"] = np.array(
            [CLASS_LABELS_TICL.index(abs(int(i))) if abs(int(i)) in CLASS_LABELS_TICL else 0
             for i in ycand["pid"]], dtype=np.float32)

        Xelem_flat   = ak.to_numpy(np.stack([Xelem[k]   for k in X_FEATURES], axis=-1))
        ytarget_flat = ak.to_numpy(np.stack([ytarget[k] for k in Y_FEATURES], axis=-1))
        ycand_flat   = ak.to_numpy(np.stack([ycand[k]   for k in Y_FEATURES], axis=-1))

        # Fix empty array shapes for tfds tensor shape requirements
        genjet   = event["genjet"]
        targetjet = event["targetjet"]
        pythia   = event["pythia"]

        if len(genjet) == 0:
            genjet = np.zeros((0, 4), dtype=np.float32)
        if len(targetjet) == 0:
            targetjet = np.zeros((0, 4), dtype=np.float32)
        if len(pythia) == 0:
            pythia = np.zeros((0, 5), dtype=np.float32)

        Xs.append(Xelem_flat)
        ytargets.append(ytarget_flat)
        ycands.append(ycand_flat)
        genmets.append(float(event["genmet"][0]))
        genjets.append(genjet)
        targetjets.append(targetjet)
        ypythias.append(pythia)

    return Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias


def generate_examples(files):
    for fi in files:
        print(datetime.datetime.now(), "started reading file", fi)
        Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias = prepare_data_ticl(str(fi))
        if len(Xs) == 0:
            print(f"Error: file {fi} is broken")
            continue
        for ii in range(len(Xs)):
            yield str(fi) + "_" + str(ii), {
                "X":          Xs[ii].astype(np.float32),
                "ytarget":    ytargets[ii].astype(np.float32),
                "ycand":      ycands[ii].astype(np.float32),
                "genmet":     np.float32(genmets[ii]),
                "genjets":    genjets[ii].astype(np.float32),
                "targetjets": targetjets[ii].astype(np.float32),
                "pythia":     ypythias[ii].astype(np.float32),
            }


def split_list(lst, x):
    sublist_size = len(lst) // x
    result = [lst[i * sublist_size:(i + 1) * sublist_size] for i in range(x - 1)]
    result.append(lst[(x - 1) * sublist_size:])
    return result


def split_sample(path, builder_config, num_splits=NUM_SPLITS, train_frac=0.8):
    files = sorted(list(path.glob("*.pkl*")))
    print(f"Found {len(files)} files in {path}")
    assert len(files) > 0
    idx_test = int(train_frac * len(files))
    files_train = files[:idx_test]
    files_test  = files[idx_test:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split  = split_list(files_test,  num_splits)
    return {
        "train": generate_examples(files_train_split[split_index]),
        "test":  generate_examples(files_test_split[split_index]),
    }
