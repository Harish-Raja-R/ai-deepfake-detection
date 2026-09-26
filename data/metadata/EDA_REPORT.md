# 📊 Exploratory Data Analysis (EDA) Report: ASVspoof 2019 LA

**Project:** AI Voice Deepfake Detection  
**Phase:** Phase 2A — Dataset Integration & Verification  
**Evaluation Target:** ASVspoof 2019 Logical Access (LA)  
**Inspection Status:** Awaiting Dataset Placement under `data/raw/ASVspoof2019_LA/`  

---

## 1. Verified Audit Status

An inspection of the local project repository was executed to evaluate the availability of the ASVspoof 2019 LA dataset.

* **Target Directory:** `data/raw/ASVspoof2019_LA/`
* **Current Status:** Not Found / Empty
* **Audio Files Detected:** `0`
* **Metadata Records Processed:** `0`

> [!IMPORTANT]
> In strict accordance with project requirements:
> - No fake or synthetic data statistics have been fabricated.
> - No placeholder visualizations have been generated in the absence of authentic data files.
> - All verification modules are compiled, operational, and prepared for instant processing once the dataset is located in `data/raw/ASVspoof2019_LA/`.

---

## 2. Integrity Check Metrics & Baseline

The automated verification engine (`src/verify_asvspoof.py`) has been configured to assess 8 critical data integrity dimensions upon file detection:

1. **Protocol Reference Match:** Evaluates whether $100\%$ of protocol entries correspond to accessible `.flac` files.
2. **Orphan File Detection:** Scans for audio files that lack corresponding protocol keys.
3. **Cryptographic Deduplication:** Computes SHA-256 hashes to flag exact duplicate recordings across partitions.
4. **Utterance ID Uniqueness:** Confirms primary key uniqueness across all utterances.
5. **Cross-Partition Speaker Disjointness:** Confirms that no speaker in the training partition ($\text{Train}$) is present in development ($\text{Dev}$) or evaluation ($\text{Eval}$) sets.
6. **Class Balance Quantification:** Computes authentic ratios between Bonafide (`REAL`) and Spoofed (`FAKE`) audio samples.
7. **Acoustic Standardization:** Verifies uniform $16\text{ kHz}$ sampling rate and bit depth across all audio streams.
8. **Audio Stream Integrity:** Executes decoding checks via `soundfile` to identify truncated or corrupted headers.

---

## 3. Anticipated Visualizations (Generated Upon Data Placement)

The script `src/verify_asvspoof.py` will render the following plots directly into `results/figures/dataset/`:
* `class_distribution_by_partition.png`: REAL vs. FAKE frequency counts across partitions.
* `spoof_real_distribution.png`: Overall proportion of authentic versus synthesized speech.
* `duration_distribution.png`: Histogram and KDE density of utterance durations (in seconds).
* `sampling_rate_distribution.png`: Sampling rate verification across all audio files.
* `speaker_distribution.png`: Top speaker contributions by utterance count.

---

## 4. Current Integrity Summary Output

```json
{
    "dataset_root_found": null,
    "status": "DATASET_NOT_FOUND",
    "error": "Directory not found under data/raw/ASVspoof2019_LA",
    "official_partitions_found": [],
    "total_files": 0,
    "real_count": 0,
    "fake_count": 0,
    "speaker_count": 0,
    "attack_type_count": 0,
    "missing_files_count": 0,
    "corrupted_files_count": 0,
    "duplicate_hashes_count": 0,
    "cross_partition_speaker_overlap": false,
    "passed_integrity_checks": false
}
```
