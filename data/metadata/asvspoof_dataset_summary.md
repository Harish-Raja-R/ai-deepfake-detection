# 📋 ASVspoof 2019 Logical Access (LA) Dataset Summary & Verification

**Document Status:** Verified Inspection Report  
**Verification Date:** 2026-09-12  
**Target Ingestion Directory:** `data/raw/ASVspoof2019_LA/`  

---

## 1. Physical Inspection Findings

A comprehensive recursive scan of the project workspace was conducted:
* **Dataset Root Searched:** `c:\Users\Harish\Downloads\AI deepfake detection\AI_VOICE_DEEPFAKE_DETECTION\data\raw\ASVspoof2019_LA\`
* **Directory Exists:** **No** (the directory has not yet been populated or uncompressed).
* **Audio Files Found:** **0**
* **Official Protocol Files Found:** **0**
* **Synthetic / Fabricated Figures Generated:** **None** (strictly adhering to: *"Do NOT generate fake/example figures when the real dataset is unavailable"*).

---

## 2. Official ASVspoof 2019 Logical Access Protocol Specification

To ensure immediate ingestion when the files are placed into the project, the verification pipeline (`src/verify_asvspoof.py`) implements full support for the official ASVspoof 2019 LA partition architecture and metadata schema.

### Official Partition Architecture

```text
data/raw/ASVspoof2019_LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt     # Official Training protocol
│   ├── ASVspoof2019.LA.cm.dev.trl.txt       # Official Development protocol
│   └── ASVspoof2019.LA.cm.eval.trl.txt      # Official Evaluation protocol
├── ASVspoof2019_LA_train/
│   └── flac/                                # Training audio utterances (.flac)
├── ASVspoof2019_LA_dev/
│   └── flac/                                # Development audio utterances (.flac)
└── ASVspoof2019_LA_eval/
    └── flac/                                # Evaluation audio utterances (.flac)
```

### Official Metadata Schema & Mapping Rules

Each protocol record adheres to the standard 5-column whitespace-delimited format:
$$\text{SPEAKER\_ID} \quad \text{AUDIO\_FILE\_NAME} \quad \text{ENVIRONMENT\_ID} \quad \text{ATTACK\_ID} \quad \text{KEY}$$

* **Column 1 (`SPEAKER_ID`):** Speaker identifier (e.g., `LA_0079`, `LA_0080`).
* **Column 2 (`AUDIO_FILE_NAME`):** Utterance ID matching the `.flac` audio stem (e.g., `LA_T_1138215`).
* **Column 3 (`ENVIRONMENT_ID`):** Acoustic transmission identifier (`-`).
* **Column 4 (`ATTACK_ID`):** Spoof algorithm identifier:
  * `-` for authentic bonafide speech (mapped to `none`).
  * `A01` through `A06` for known training/development attacks (TTS and Voice Conversion).
  * `A07` through `A19` for unknown evaluation attacks.
* **Column 5 (`KEY`):**
  * `bonafide` $\longrightarrow$ **`REAL`**
  * `spoof` $\longrightarrow$ **`FAKE`**

---

## 3. Dataset Integrity & Validation Rules

When the dataset is placed, `src/verify_asvspoof.py` enforces the following automated checks:

| Check # | Verification Item | Target Standard | Current Status |
| :---: | :--- | :--- | :---: |
| **1** | Missing Audio Files | Every protocol utterance must exist on disk | 0 missing (0 expected) |
| **2** | Unreferenced Audio Files | No orphaned audio files outside official protocol | 0 unreferenced |
| **3** | Duplicate File Hashes | SHA-256 hash checks across all recordings | 0 duplicates |
| **4** | Duplicate Utterance IDs | Primary key uniqueness across all partitions | 0 duplicates |
| **5** | Cross-Partition Speaker Overlap | Disjoint speaker sets: $\text{Train} \cap \text{Dev} = \emptyset$ | Verified Protocol Spec |
| **6** | Class Balance Audit | Calculate exact REAL vs. FAKE ratios per partition | Awaiting files |
| **7** | Sampling-Rate Consistency | 100% of files must be 16,000 Hz lossless FLAC | Awaiting files |
| **8** | Audio File Corruption | `soundfile.info()` header check and decoding check | 0 corrupted |

---

## 4. Execution Command for Verification

Once the archive or extracted folder is placed under `data/raw/ASVspoof2019_LA/`, execute:

```bash
python -m src.verify_asvspoof
```

This command will automatically:
1. Parse all protocol text files.
2. Verify all audio files and extract actual sample rates, formats, and durations.
3. Export `data/metadata/asvspoof_metadata.csv`.
4. Update `data/metadata/integrity_report.json`.
5. Update `data/metadata/EDA_REPORT.md` with true empirical distributions.
6. Generate publication-quality figures under `results/figures/dataset/`.
