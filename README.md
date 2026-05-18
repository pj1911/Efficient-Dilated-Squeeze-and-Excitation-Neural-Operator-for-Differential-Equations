# Efficient-Dilated-Squeeze-and-Excitation-Neural-Operator-for-Differential-Equations:
D-SENO is a lightweight neural operator for fast, accurate PDE surrogates. It combines dilated convolution blocks (wide receptive fields) with squeeze-excitation modules (channel recalibration) to capture long-range physics efficiently. Achieves up to ~20× faster training while matching/outperforming prior methods.

# Link to the paper:
https://openreview.net/pdf?id=Xl942THEUa

# Dataset: 
The datasets used in this study are publicly available: airfoil potential flow and Poiseuille pipe flow datasets are available at \url{https://github.com/neuraloperator/Geo-FNO}, and the Darcy flow and Navier-Stokes datasets are available at \url{https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-}.

Please download the datasets first and set --data_dir to the directory containing the .npy files.

## Installation:

pip install -r requirements.txt

## Run: D-SENO (Experiments):

After installing the dependencies, run the following commands from the root directory of this repository.

### 1) Run D-SENO (all tasks):
`python scripts/d_seno_airfoil.py --data_dir /path/to/npy/files`<br>
`python scripts/d_seno_pipe.py --data_dir /path/to/npy/files`<br>
`python scripts/d_seno_darcy.py --data_dir /path/to/npy/files`<br>
`python scripts/d_seno_ns.py --data_dir /path/to/npy/files`

### 2) Run FNO+ (all tasks):
`python scripts/fno_plus_airfoil.py --data_dir /path/to/npy/files`<br>
`python scripts/fno_plus_pipe.py --data_dir /path/to/npy/files`<br>
`python scripts/fno_plus_darcy.py --data_dir /path/to/npy/files`<br>
`python scripts/fno_plus_ns.py --data_dir /path/to/npy/files`
