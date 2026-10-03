import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import (
    VoiceDeepfakeDetector,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    CNN_EER_THRESHOLD,
    STANDARD_THRESHOLD,
)

def test_samples():
    detector = VoiceDeepfakeDetector(DEFAULT_MODEL_PATH, DEFAULT_STATS_PATH)

    samples = [
        ("REAL (Human Speech 1)", "data/raw/real/yt_0000_p2_part_167.flac"),
        ("REAL (Human Speech 2)", "data/raw/real/yt_0000_part_001.flac"),
        ("FAKE: ElevenLabs", "data/raw/fake/el_0001_c_part_002.flac"),
        ("FAKE: Amazon Polly", "data/raw/fake/po_0001_c_part_001.flac"),
        ("FAKE: Speechify", "data/raw/fake/sp_0001_c_part_006.flac"),
        ("FAKE: Luvvoice", "data/raw/fake/lv_0001_c_part_018.flac"),
        ("FAKE: Kokoro TTS", "data/raw/fake/hg_0001_c_part_005.flac"),
        ("FAKE: Hume AI", "data/raw/fake/hu_0001_c_part_012.flac"),
    ]

    print('=' * 105)
    print(f"{'Category':<22} | {'Filename':<24} | {'P(FAKE)':<8} | {'Std (0.50)':<20} | {'EER (0.2879)':<22}")
    print('-' * 105)

    for cat, rel_p in samples:
        p = Path(rel_p)
        res_std = detector.predict_file(p, threshold=STANDARD_THRESHOLD)
        res_eer = detector.predict_file(p, threshold=CNN_EER_THRESHOLD)
        p_fake = res_std['probability_fake']
        v_std = res_std['verdict']
        v_eer = res_eer['verdict']
        print(f"{cat:<22} | {p.name:<24} | {p_fake:<8.4f} | {v_std:<20} | {v_eer:<22}")

    print('=' * 105)

if __name__ == '__main__':
    test_samples()
