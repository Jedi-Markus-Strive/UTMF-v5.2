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

## 5. Dataset Download Information (UTMF v5.2)

UTMF v5.2 operates on publicly available datasets drawn from a wide range of
physical experiments and surveys. For licensing, size, and provenance reasons,
**raw datasets are not bundled** with this repository.

UTMF v5.2 is designed such that datasets are loaded externally, while all
measurement results, configuration parameters, subset strategies, and
multifractal outputs are archived internally in fully reproducible JSON files.

---

### Recommended directory structure (Google Colab)

```
/MyDrive/
└── Datasets_UTMF/
    ├── Datasets/
    └── UTMF_outputs/
```

Workflow:

1. Mount Google Drive
2. Place downloaded datasets in `/Datasets_UTMF/Datasets/`
3. Run the single UTMF v5.2 analysis cell
4. Results are written to `/Datasets_UTMF/UTMF_outputs/`

---

## Supported datasets

### **LIGO — Gravitational Wave Open Science Center (GWOSC)**

Strain data from observing run O4a (16 kHz sampling).

* File format: HDF5
* Example filenames: `L-L1_GWOSC_O4a_16KHZ_R1-*.hdf5`

Archive:
[https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1368195220/1389456018/simple/](https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1368195220/1389456018/simple/)

**Example datasets used in UTMF v5.x configurations:**

* `L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5` (486 MB)
* `L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5` (486 MB)
* `L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5` (486 MB)
* `L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5` (486 MB)

---

### **Planck — ESA Legacy Archive**

Cosmic Microwave Background (CMB) sky maps.

* File format: FITS
* Typical products: SMICA IQU component maps

Archive:
[https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)

**Example datasets used in UTMF v5.x:**

* `COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits` (384 MB)
* `COM_CMB_IQU-smica_2048_R3.00_full.fits` (1.88 GB)
* `LFI_SkyMap_070_1024_R3.00_full.fits` (480 MB)

---

### **DESI — Dark Energy Spectroscopic Instrument**

Large-scale structure catalogs.

* File format: FITS
* Example: luminous red galaxy (LRG) catalogs

Portal:
[https://data.desi.lbl.gov/doc/releases/dr1/](https://data.desi.lbl.gov/doc/releases/dr1/)

**Dataset used in UTMF v5.x:**

* `LRG_full.dat.fits` (2.77 GB)

---

### **CERN Open Data Portal**

High-energy physics event data.

* File format: ROOT
* Example: dilepton event selections

Portal:
[https://opendata.cern.ch/record/15007](https://opendata.cern.ch/record/15007)

**Dataset used in UTMF v5.x:**

* `data_B.exactly2lep.root` (451 MB)

A small helper file for ROOT-to-array conversion is provided:

* `data_B.exactly2lep.h5` (315 KB)
  Store this file alongside the ROOT file.

---

### **NIST Atomic Spectra Database**

Atomic emission spectra.

* File format: CSV
* Individual element spectra

Database:
[https://physics.nist.gov/PhysRefData/ASD/lines_form.html](https://physics.nist.gov/PhysRefData/ASD/lines_form.html)

**Pre-packaged dataset used in UTMF:**

* `NIST_elements`
  Download:
  [https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/Datasets/NIST_3.zip](https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/Datasets/NIST_3.zip)

Unzip and use the CSV files directly for UTMF analysis.

---

### **NANOGrav Pulsar Timing Arrays**

Pulsar timing residuals.

* File format: timing residual files
* Data release: 15-year narrowband set

Zenodo archive:
[https://zenodo.org/records/16051178](https://zenodo.org/records/16051178)

**Dataset used in UTMF v5.x:**

* `NANOGrav15yr_PulsarTiming_v2.1.0` (639 MB)

Preparation notebook:
[https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/prepare_NANOGrav_15yr_data.ipynb](https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/prepare_NANOGrav_15yr_data.ipynb)

---

### **Gaia Archive (DR3)**

Astrometric source catalogs.

* File format: TSV
* Query-based download

Access via VizieR:
[https://vizier.cds.unistra.fr/viz-bin/VizieR-4](https://vizier.cds.unistra.fr/viz-bin/VizieR-4)

**Example query used in UTMF v5.x:**

* Catalog: `I/355/gaiadr3`
* Rows: `1–999999`
* Format: TSV
* Columns: All
* Rename output file to `gaia_dr3` or update the path in the UTMF configuration.

---

### **Quantum Random Number Generator (QRNG)**

True quantum randomness.

* Provider: Australian National University
* Access: API-based
* No local download required

Website:
[https://qrng.anu.edu.au/](https://qrng.anu.edu.au/)

QRNG streams are incorporated directly in the UTMF configuration.

---

### Notes on UTMF v5.2 usage

* Not all datasets need to be used in a single run.
* Subset design, scale selection, and stability analysis are handled internally
  by UTMF v5.2.
* All run outputs are written as self-contained JSON archives, enabling full
  reproducibility without redistributing raw data.


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


