# Efficient-Dilated-Squeeze-and-Excitation-Neural-Operator-for-Differential-Equations
D-SENO is a lightweight neural operator for fast, accurate PDE surrogates. It combines dilated convolution blocks (wide receptive fields) with squeeze-excitation modules (channel recalibration) to capture long-range physics efficiently. Achieves up to ~20× faster training while matching/outperforming prior methods.

## Run: D-SENO (Experiments)

1) Install dependencies:
   
   pip install -r requirements.txt

2) Run experiments for d_seno:

    python scripts/d_seno_airfoil.py --data_dir /path/to/npy/files
    python scripts/d_seno_pipe.py --data_dir /path/to/npy/files
    python scripts/d_seno_darcy.py --data_dir /path/to/npy/files
    python scripts/d_seno_ns.py --data_dir /path/to/npy/files

3) Run experiments for fno_plus:

    python scripts/fno_plus_airfoil.py --data_dir /path/to/npy/files
    python scripts/fno_plus_pipe.py --data_dir /path/to/npy/files
    python scripts/fno_plus_darcy.py --data_dir /path/to/npy/files
    python scripts/fno_plus_ns.py --data_dir /path/to/npy/files
