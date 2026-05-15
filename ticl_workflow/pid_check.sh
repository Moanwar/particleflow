python3 - << 'PYEOF'
import awkward as ak
import numpy as np
import glob

pred_path = "/cms/data/store/user/moanwar/mlpf_data/experiments_ttbar/preds_step_10000/cms_pf_ticl_nopu/"
files = sorted(glob.glob(pred_path + "*.parquet"))[:100]

CLASS_NAMES = ["none", "ch.had", "n.had", "gamma", "ele", "mu"]
all_tgt, all_prd = [], []
all_X = []

for f in files:
    arr = ak.from_parquet(f)
    all_tgt.extend(ak.to_numpy(ak.flatten(arr['particles']['target']['cls_id'])).tolist())
    all_prd.extend(ak.to_numpy(ak.flatten(arr['particles']['pred']['cls_id'])).tolist())
    all_X.extend(ak.to_numpy(ak.flatten(arr['inputs'])).tolist())

all_tgt = np.array(all_tgt)
all_prd = np.array(all_prd)
all_X   = np.array(all_X)

print("=== Confusion matrix (rows=target, cols=predicted) ===")
print(f"{'':>10}", end="")
for name in CLASS_NAMES:
    print(f"{name:>10}", end="")
print()

for i, tname in enumerate(CLASS_NAMES):
    if i == 0: continue
    tgt_mask = all_tgt == i
    if tgt_mask.sum() == 0: continue
    print(f"{tname:>10}", end="")
    for j in range(len(CLASS_NAMES)):
        n = int(np.sum((all_tgt == i) & (all_prd == j)))
        pct = 100 * n / tgt_mask.sum()
        print(f"{pct:>9.1f}%", end="")
    print(f"  (N={tgt_mask.sum()})")

print("\n=== n.had confusion by element type ===")
ELEM = {1:'Track', 2:'EM_TS', 3:'HAD_TS'}
for typ, tname in ELEM.items():
    mask = (all_X[:,0] == typ) & (all_tgt == 2)  # n.had targets
    if mask.sum() == 0: continue
    total = mask.sum()
    print(f"\n{tname} with n.had target (N={total}):")
    for j, pname in enumerate(CLASS_NAMES):
        n = int(np.sum(mask & (all_prd == j)))
        print(f"  predicted {pname}: {n} ({100*n/total:.1f}%)")
PYEOF
