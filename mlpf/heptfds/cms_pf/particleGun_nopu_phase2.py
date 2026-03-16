"""TICL PF dataset."""
import os
import cms_utils_phase2
import numpy as np
import tensorflow_datasets as tfds
import pathlib

X_FEATURES = cms_utils_phase2.X_FEATURES
Y_FEATURES = cms_utils_phase2.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.
TICL-based PF reconstruction in HGCAL.
"""
_CITATION = ""


class CmsPfTiclNopu(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for TICL PF dataset."""
    
    # Required by TFDS even if not using manual_dir
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Using environment variables for path control"
    
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "1.0.0"))
    RELEASE_NOTES = {
        "1.0.0": "First TICL version",
    }
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name=str(group))
        for group in range(1, cms_utils_phase2.NUM_SPLITS + 1)
    ]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        # Allow TFDS data dir to be set via environment variable
        if "data_dir" not in kwargs and "TFDS_DATA_DIR" in os.environ:
            kwargs["data_dir"] = os.environ["TFDS_DATA_DIR"]
        super(CmsPfTiclNopu, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "X":          tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=np.float32),
                "ytarget":    tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                "ycand":      tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                "genmet":     tfds.features.Scalar(dtype=np.float32),
                "genjets":    tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "pythia":     tfds.features.Tensor(shape=(None, 5), dtype=np.float32),
            }),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features=X_FEATURES,
                y_features=Y_FEATURES
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        import pickle
        
        # Get paths from environment variables with defaults
        input_dir = os.environ.get("INPUT_DIR", "/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/mlpf/data/cms/raw")
        output_base = os.environ.get("OUTPUT_DIR", "/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/mlpf/data/cms/processed")
        
        # Construct full paths
        input_pkl = os.path.join(input_dir, "ticl_graph_data_prt.pkl")
        split_output_dir = pathlib.Path(output_base) / "ticl_particleGun_0pu"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        #print("="*50)
        #print("PATH CONFIGURATION:")
        #print(f"INPUT_DIR: {input_dir}")
        #print(f"Input file: {input_pkl}")
        #print(f"OUTPUT_DIR: {output_base}")
        #print(f"Split files will be saved to: {split_output_dir}")
        #print(f"Final TFDS dataset will be saved to: {self._data_dir or 'Default TFDS location'}")
        #print("="*50)
        
        # Check if input file exists
        if not os.path.exists(input_pkl):
            raise FileNotFoundError(f"Input file not found: {input_pkl}")
        
        # Split only if not already done
        existing = sorted(split_output_dir.glob("*.pkl"))
        if not existing:
            print(f"Splitting {input_pkl} into chunks...")
            print("Loading pickle file...")
            data = pickle.load(open(input_pkl, "rb"))
            print(f"Loaded {len(data)} events")
            
            chunk_size = 100
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            print(f"Creating {len(chunks)} chunk files...")
            
            for i, chunk in enumerate(chunks):
                out_path = split_output_dir / f"events_{i:04d}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                if (i+1) % 10 == 0:
                    print(f"  Created {i+1}/{len(chunks)} files")
                    
            print(f"Split complete: {len(chunks)} files created in {split_output_dir}")
        else:
            print(f"Found {len(existing)} existing files in {split_output_dir}, skipping split")

        return cms_utils_phase2.split_sample(
            split_output_dir,
            self.builder_config,
            num_splits=cms_utils_phase2.NUM_SPLITS
        )

    def _generate_examples(self, files):
        return cms_utils_phase2.generate_examples(files)
