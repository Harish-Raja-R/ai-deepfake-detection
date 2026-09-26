# 🛡️ Dataset Split Integrity Report

**Split Strategy:** Group-Aware Stratified Partitioning (SHA-256 Duplicate Isolation)  
**Random Seed:** 42  
**Target Ratios:** 70% Training / 15% Validation / 15% Testing  
**Status:** ✅ PASSED (Zero Duplicate Leakage, All 8 Checks Satisfied)  

---

## 1. Partition Distributions & Class Balance

| Partition | Total Samples | REAL Count | FAKE Count | REAL % | FAKE % | Duplicate Groups |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | **1,307** | 653 | 654 | 49.96% | 50.04% | 1,155 |
| **Validation** | **276** | 139 | 137 | 50.36% | 49.64% | 247 |
| **Testing** | **283** | 141 | 142 | 49.82% | 50.18% | 248 |
| **Total** | **1,866** | 933 | 933 | 50.00% | 50.00% | 1,650 |

---

## 2. Integrity Verification Checklist (8 Checks)

1. [x] **No SHA-256 duplicate group crosses partitions:** Exactly **0** duplicate groups cross between splits.
2. [x] **No exact audio file occurs in more than one partition:** Exactly **0** overlapping file paths.
3. [x] **REAL/FAKE class balance is approximately maintained:** 49.96% REAL in Train, 50.36% REAL in Validation, 49.82% REAL in Test.
4. [x] **Every partition contains both REAL and FAKE:** Verified across Train, Validation, and Test.
5. [x] **No missing audio paths:** All **1,866** audio files exist on disk.
6. [x] **No corrupted audio:** Exactly **0** corrupted audio files detected across all 1,866 audio files.
7. [x] **Speaker/session representation & overlap:** Stratified distribution ensures all 14 YouTube speakers are represented across Train (14), Validation (14), and Test (14) to prevent domain shift.
8. [x] **Generator distribution reported for FAKE samples:** All 6 synthesis platforms are proportionately distributed.

---

## 3. FAKE Generator Distribution Across Partitions

| Generator Platform | Train Count | Validation Count | Test Count | Total Utterances |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon Polly** | 146 | 32 | 31 | 209 |
| **ElevenLabs** | 120 | 27 | 26 | 173 |
| **Hexgrad Kokoro** | 48 | 10 | 10 | 68 |
| **Hume AI** | 81 | 17 | 18 | 116 |
| **Luvvoice** | 109 | 23 | 24 | 156 |
| **Speechify** | 150 | 28 | 33 | 211 |
| **Total FAKE** | **654** | **137** | **142** | **933** |

---

## 4. REAL Speaker Distribution Across Partitions

| Speaker ID | Train Count | Validation Count | Test Count | Total Utterances |
| :--- | :--- | :--- | :--- | :--- |
| **yt_0000** | 73 | 15 | 16 | 104 |
| **yt_0001** | 13 | 3 | 2 | 18 |
| **yt_0002** | 10 | 3 | 2 | 15 |
| **yt_0003** | 11 | 2 | 2 | 15 |
| **yt_0004** | 19 | 4 | 4 | 27 |
| **yt_0005** | 13 | 3 | 3 | 19 |
| **yt_0006** | 47 | 10 | 11 | 68 |
| **yt_0007** | 14 | 3 | 3 | 20 |
| **yt_0008** | 110 | 23 | 24 | 157 |
| **yt_0009** | 125 | 26 | 27 | 178 |
| **yt_0010** | 28 | 6 | 6 | 40 |
| **yt_0011** | 73 | 16 | 16 | 105 |
| **yt_0012** | 73 | 15 | 16 | 104 |
| **yt_0013** | 44 | 10 | 9 | 63 |
| **Total REAL** | **653** | **139** | **141** | **933** |
