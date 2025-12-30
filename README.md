# UTMF v5.2  
### Unified Temporal–Measurement Framework  
**Subset Design and Stability Regimes for Multifractal Detrended Fluctuation Analysis**

---

## Overview

UTMF v5.2 is a **deterministic, fully reproducible measurement framework** for
Multifractal Detrended Fluctuation Analysis (MFDFA).
It is designed to make multifractal analysis **transparent, stable, and accessible**
across heterogeneous physical datasets.

This release focuses explicitly on:

- subset length design
- adaptive versus fixed subset strategies
- run-to-run stability
- reproducible multifractal measurement

UTMF v5.2 does **not** introduce a new multifractal formalism.
Instead, it provides a **clear methodological baseline** for applying MFDFA
without hidden tuning or domain-specific heuristics.

---

## Design Philosophy

UTMF v5.2 is intentionally simple.

- The full framework is provided as **a single Python cell**
- No advanced object-oriented structure is required
- All configuration choices are explicit and logged
- Every run produces a complete, self-contained JSON archive

This design is deliberate:
UTMF is meant to be usable by **non-specialists**, students, and researchers
who want reliable multifractal measurements without turning MFDFA into a black box.

Programmers are of course free to refactor or modularize the code.
UTMF itself prioritizes **clarity over abstraction**.

---

## What Is Included

This repository contains:

- The complete UTMF v5.2 analysis cell
- The reference `jedi_mfdfa` implementation used internally
- Deterministic subset design logic
- Export of fully reproducible `FULL_DETAILS` JSON outputs
- Compatibility with downstream analysis scripts (e.g. figure reproduction)

Raw datasets are **not** included due to size and licensing constraints.

---

## Relationship to Other Work

UTMF v5.2 builds directly on:

- **jedi_mfdfa**  
  A minimal and robust implementation of MFDFA

- **UTMF-Core**  
  A domain-agnostic multifractal measurement framework

- **UTMF v5.1**  
  A cross-domain coherence study using the same measurement pipeline

This repository focuses specifically on **subset design and stability**.
Coherence indices (TCI, MCI, TMCI) are computed by UTMF but are analysed separately
in dedicated work.

---

## Dataset Download Information (UTMF v5.2)

The UTMF v5.2 runs analysed in the accompanying paper use publicly available datasets
from multiple physical domains.
For licensing and size reasons, raw data are **not bundled** in this repository.

**Total raw data volume:** approximately 12 GB

### Recommended folder structure (Google Colab)

```
/MyDrive/
└── Datasets_UTMF/
    ├── Datasets/
    └── UTMF_outputs/
```


Mount your Google Drive, place the datasets in `/Datasets_UTMF/Datasets/`,
run the UTMF cell, and results will be written to `/Datasets_UTMF/UTMF_outputs/`.
UTMF v5.2 is designed to be executed as a single notebook cell.
Running the cell from top to bottom performs the full analysis.

A full UTMF v5.2 run may take 10 minutes to several hours depending on hardware and 
selected datasets.

Tested in Google Colab (Python 3.10).
Local execution may require manual dependency resolution.

---

### Datasets used in UTMF v5.2

#### **LIGO — Gravitational Wave Open Science Center**
- Strain data (O4a, 16 kHz)
- HDF5 files (`L-L1_GWOSC_O4a_16KHZ_R1-*.hdf5`)

Download:
https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1368195220/1389456018/simple/

---

#### **NIST Atomic Spectra Database**
- Atomic emission spectra (CSV)

Dataset used in UTMF configuration:
- `NIST_elements`  
  Download (pre-packaged for UTMF):
  https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/Datasets/NIST_3.zip

Unzip and use the CSV files for UTMF analysis.

---

#### **NANOGrav Pulsar Timing Residuals**
- 15-year data release
- Narrowband pulsar timing residuals

Dataset used:
- `NANOGrav15yr_PulsarTiming_v2.1.0`

Download:
https://zenodo.org/records/16051178/files/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz

Preparation notebook:
https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/prepare_NANOGrav_15yr_data.ipynb

---

#### **Quantum Random Number Generator (QRNG)**
- Australian National University QRNG
- API-based random bit streams

Website:
https://qrng.anu.edu.au/

No local download required; incorporated directly in the UTMF configuration.

---

## Reproducibility

Each UTMF v5.2 execution produces a single `FULL_DETAILS` JSON file.
This file contains:

- All subset-level MFDFA results
- Subset design parameters
- Dataset metadata and quality metrics
- Configuration snapshot
- A frozen copy of the exact `jedi_mfdfa` code used

All figures and statistics reported in the paper can be reproduced **without rerunning UTMF**
by post-processing these JSON files.

A dedicated figure reproduction script is provided in the accompanying Zenodo archive.

---

## Citation

If you use UTMF v5.2, please cite:

- the UTMF v5.2 software release
- the accompanying methodological paper
- the `jedi_mfdfa` reference implementation

Full citation details are provided via Zenodo.

---

## Author

**Jedi Markus Strive**  
Independent Researcher  
ORCID: 0009-0000-7663-7946  
Contact: crisplatform@gmail.com

---

## Disclaimer

UTMF v5.2 does not claim to define a unique or optimal MFDFA strategy.
Subset design choices are empirically motivated and intended as a transparent baseline.

Users are explicitly encouraged to experiment with alternative parameters,
subset strategies, and extensions.


