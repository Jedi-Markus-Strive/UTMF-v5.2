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

## Accompanying paper

This repository accompanies the paper  
**Subset Design and Stability Regimes in Multifractal Detrended Fluctuation Analysis**  
(M. Eversdijk, 2025)  
DOI: https://doi.org/10.5281/zenodo.18098538

The paper documents the empirical stability analysis of MFDFA performed with
UTMF v5.2 and `jedi_mfdfa`. All results are fully reproducible from the
archived metadata and scripts provided here and on Zenodo.

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

## Methodological lineage

This work is part of a reproducible multifractal measurement framework:

- jedi_mfdfa (MFDFA reference implementation)
- UTMF-Core (domain-agnostic measurement framework)
- UTMF v5.x (execution & validation environment)
- FAT (derived asymmetry observables)

Each component is archived on Zenodo with a persistent DOI.

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

## 5. Dataset Download Information (UTMF v5.2)

UTMF v5.2 operates on publicly available datasets drawn from a wide range of
physical experiments and surveys. For licensing, size, and provenance reasons,
**raw datasets are not bundled** with this repository.

UTMF v5.2 is designed such that datasets are loaded externally, while all
measurement results, configuration parameters, subset strategies, and
multifractal outputs are archived internally in fully reproducible JSON files.

UTMF v5.2 is designed to be executed as a single notebook cell.
Running the cell from top to bottom performs the full analysis.

A full UTMF v5.2 run may take 10 minutes to several hours depending on hardware and 
selected datasets.

Tested in Google Colab (Python 3.10).
Local execution may require manual dependency resolution.

---

### Recommended directory structure (Google Colab)

```
/MyDrive/
└── Datasets_UTMF/
    ├── Datasets/
    └── UTMF_outputs/
```

## **All datasets used for UTMF v5.2 listed below.** (Total 11.2GB) 
#### - Direct downloadlinks.
#### - For Colab: Create: /MyDrive/Datasets_UTMF/UTMF_outputs/
#### - Place the datasets in folder: /Datasets_UTMF/
#### - Mount Drive
#### - Run UTMF v5.2
#### - Results are returned in folder: /UTMF_outputs/
-----
- **[LIGO – GWOSC](https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1368195220/1389456018/simple/)**  
  HDF5 strain files (e.g., `L-L1_GWOSC_O4a_16KHZ_R1-*.hdf5`).
                                                           
  **Datasets used in UTMF v5.2 configuration:**
- `L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1384120320/L-L1_GWOSC_O4a_16KHZ_R1-1384779776-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1367343104/L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1369440256/L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5) (486MB)
- `L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5` [Download](https://gwosc.org/archive/data/O4a_16KHZ_R1/1389363200/L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5) (486MB)
---
- **[Planck – ESA Archive](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)**  
  FITS CMB maps (e.g., SMICA IQU maps such as `COM_CMB_IQU-smica_2048_R3.00_full.fits`).

  **Datasets used in UTMF v5.2 configuration:**
- `COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits` [Download](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits) (384MB)
- `COM_CMB_IQU-smica_2048_R3.00_full.fits`      [Download](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits) (1.88GB)
- `LFI_SkyMap_070_1024_R3.00_survey-1.fits`     [Download](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_full.fits) (480MB)
---
- **[DESI – Data Release Portal](https://data.desi.lbl.gov/doc/releases/dr1/)**  
  LRG FITS catalogs (e.g., `LRG_full.dat.fits`).

  **Dataset used in UTMF v5.2 configuration:**
- `LRG_full.dat.fits` [Download](https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/LRG_full.dat.fits) (2.77GB)
---  
- **[CERN Open Data](https://opendata.cern.ch/record/15007)**  
  ROOT event files (e.g., `data_B.exactly2lep.root`).

  **Dataset used in UTMF v5.2 configuration:**
- `data_B.exactly2lep.root` [Download:](https://opendata.cern.ch/record/15007/files/data_B.exactly2lep.root) (451MB)
                                                                                              
    ➕ This repository includes a helpfile: `data_B.exactly2lep.h5` [Download:](https://github.com/Jedi-Markus-Strive/UTMF-CRISP/raw/refs/heads/main/Datasets/data_B.exactly2lep.h5) (315KB) (helpfile for .root, store it at the same location as .root-file.)
---
- **[NIST Atomic Spectra Database](https://physics.nist.gov/PhysRefData/ASD/lines_form.html)**                          
  CSV spectra 
  
  **Dataset used in UTMF v5.2 configuration:**                                                                        
- `NIST_elements` [Download](https://github.com/Jedi-Markus-Strive/UTMF-v5.1-Coherence/raw/refs/heads/main/downloads/NIST_elements.zip)** (8.5MB) (Complete dataset as used in UTMF v5.2, unzip and use the CSV's (57.5MB) for UTMF analysis.)
---
- **[NANOGrav Data Releases](https://zenodo.org/records/16051178)**  
- Pulsar timing residuals (e.g., `NG15yr narrowband` files).

  **Dataset used in UTMF v5.2 configuration:**
- `NANOGrav15yr_PulsarTiming_v2.1.0` [Download:](https://zenodo.org/records/16051178/files/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz?download=1) (639MB) ([Unzip](https://github.com/Jedi-Markus-Strive/UTMF-v5.1-Coherence/blob/master/downloads/prepare_NANOGrav_15yr_data.ipynb) the file, use for UTMF analysis.)
---
- **[Gaia Archive (DR3)](https://vizier.cds.unistra.fr/viz-bin/VizieR-4)**  
  Source catalogs in TSV format (e.g., `gaia_dr3.tsv`).                                                                 
  **Dataset used in UTMF v5.2:**                                                                                      
      **Select:**                                                                                                       
        1- 'gaiadr3'                                                                                                   
        2. Table: `I/355/gaiadr3`  
        3. Rows: `1-999999`  
        4. Format: Tab-Separated Values  
        5. Columns: All                                                                                
        6. Rename file to 'gaia_dr3' or update path in config.                                                          
---
- **[ANU Quantum Random Numbers (QRNG)](https://qrng.anu.edu.au/)**  
  API-based quantum random sequences (no download required, incorporated in UTMF v5.2 configuration).
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


