# check_files.py
import uproot, sys, os

output    = "zll_0pu_v0.txt"
directory="/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/zll_0pu_v0/histo/"

files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.root')]

good, bad = [], []
for f in sorted(files):
    try:
        tf = uproot.open(f)
        tf['ticlDumper/tracks']  # check key exists
        tf['ticlDumper/simTICLCandidate']
        good.append(f)
        print(f"OK: {f}")
    except Exception as e:
        bad.append(f)
        print(f"BAD: {f} — {e}", file=sys.stderr)

with open(output, 'w') as out:
    for f in good:
        out.write(f"{f}\n")
        print(f"\nGood: {len(good)}  Bad: {len(bad)}")
	

