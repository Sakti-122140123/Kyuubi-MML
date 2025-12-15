import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ---------------------------------
# KONFIGURASI
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sesuaikan dengan nama file kamu
# Misal: audio_prob_for_fusion.csv, dst.
AUDIO_CSV  = "audio_prob_for_fusion.csv"   # atau "audio_prob.csv"
LYRICS_CSV = "lyrics_prob_for_fusion.csv"  # atau "lyric_prob.csv"
MIDI_CSV   = "midi_prob_for_fusion.csv"

# Di CSV, true_label kamu = 0..4
LABELS = [0, 1, 2, 3, 4]
EMOTION_CLASSES = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]


def load_modal_csv(filename, modal_name):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{modal_name}: file {path} tidak ditemukan.")

    df = pd.read_csv(path)

    required = ["id", "true_label"] + [f"prob_cluster_{i}" for i in range(1, 6)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{modal_name}: kolom hilang: {missing}")

    df["true_label"] = df["true_label"].astype(int)
    return df


def add_prefix(df, prefix):
    df = df.copy()
    for i in range(1, 6):
        df.rename(columns={f"prob_cluster_{i}": f"prob_cluster_{i}_{prefix}"}, inplace=True)
    return df


def normalize_weights(ws):
    s = sum(ws)
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [w / s for w in ws]


def main():
    # 1) Load semua CSV unimodal
    audio_raw  = load_modal_csv(AUDIO_CSV,  "audio")
    lyrics_raw = load_modal_csv(LYRICS_CSV, "lyrics")
    midi_raw   = load_modal_csv(MIDI_CSV,   "midi")

    # 2) Prefix kolom probability
    audio  = add_prefix(audio_raw,  "audio")
    lyrics = add_prefix(lyrics_raw, "lyrics")
    midi   = add_prefix(midi_raw,   "midi")

    # 3) Outer join berdasarkan id + true_label (union semua lagu)
    merged = audio.merge(
        lyrics[["id", "true_label"] + [c for c in lyrics.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="outer"
    )
    merged = merged.merge(
        midi[["id", "true_label"] + [c for c in midi.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="outer"
    )

    merged["true_label"] = merged["true_label"].astype(int)
    y_true = merged["true_label"].values

    # 4) Bobot F1 unimodal (dari VALIDATION, bukan test)
    #    >>> GANTI angka di sini dengan macro F1 VAL yang sebenarnya <<<
    f1_audio_val  = 0.30   # contoh, isi dengan F1 val audio kamu
    f1_lyrics_val = 0.3956 # contoh, isi dengan F1 val lyrics kamu
    f1_midi_val   = 0.20   # contoh, isi dengan F1 val MIDI kamu

    # Hitung bobot F1-normalized untuk tiap kombinasi
    w_a_l   = normalize_weights([f1_audio_val,  f1_lyrics_val])          # [wa_al, wl_al]
    w_a_m   = normalize_weights([f1_audio_val,  f1_midi_val])            # [wa_am, wm_am]
    w_l_m   = normalize_weights([f1_lyrics_val, f1_midi_val])            # [wl_lm, wm_lm]
    w_a_l_m = normalize_weights([f1_audio_val,  f1_lyrics_val, f1_midi_val])  # [wa_alm, wl_alm, wm_alm]

    # 5) Fusi per-baris dengan logic "smart" (tergantung modalitas yang tersedia)
    y_pred = []
    combo_counts = {
        "audio_only": 0,
        "lyrics_only": 0,
        "midi_only": 0,
        "audio_lyrics": 0,
        "audio_midi": 0,
        "lyrics_midi": 0,
        "audio_lyrics_midi": 0,
    }

    for _, row in merged.iterrows():
        # cek ketersediaan (NaN -> tidak ada)
        has_a = not np.isnan(row.get("prob_cluster_1_audio", np.nan))
        has_l = not np.isnan(row.get("prob_cluster_1_lyrics", np.nan))
        has_m = not np.isnan(row.get("prob_cluster_1_midi", np.nan))

        pa = pl = pm = None

        if has_a:
            pa = np.array(
                [row[f"prob_cluster_{i}_audio"] for i in range(1, 6)],
                dtype=float
            )
        if has_l:
            pl = np.array(
                [row[f"prob_cluster_{i}_lyrics"] for i in range(1, 6)],
                dtype=float
            )
        if has_m:
            pm = np.array(
                [row[f"prob_cluster_{i}_midi"] for i in range(1, 6)],
                dtype=float
            )

        # Logic kombinasi modalitas
        if has_a and has_l and has_m:
            combo_counts["audio_lyrics_midi"] += 1
            wa, wl, wm = w_a_l_m
            p = wa * pa + wl * pl + wm * pm

        elif has_a and has_l:
            combo_counts["audio_lyrics"] += 1
            wa, wl = w_a_l
            p = wa * pa + wl * pl

        elif has_a and has_m:
            combo_counts["audio_midi"] += 1
            wa, wm = w_a_m
            p = wa * pa + wm * pm

        elif has_l and has_m:
            combo_counts["lyrics_midi"] += 1
            wl, wm = w_l_m
            p = wl * pl + wm * pm

        elif has_a:
            combo_counts["audio_only"] += 1
            p = pa

        elif has_l:
            combo_counts["lyrics_only"] += 1
            p = pl

        elif has_m:
            combo_counts["midi_only"] += 1
            p = pm

        else:
            # tidak ada modalitas sama sekali → fallback uniform
            p = np.ones(5, dtype=float) / 5.0

        # argmax → label 0..4 (TANPA +1 karena CSV pakai 0..4)
        y_pred.append(int(np.argmax(p)))

    y_pred = np.array(y_pred)

    # 6) Hitung metrik global
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    print("=== Smart late fusion (conditional, F1-weighted) ===")
    print("Accuracy :", acc)
    print("Macro F1 :", macro_f1)

    print("\nPer-class metrics:")
    for i, cls in enumerate(EMOTION_CLASSES):
        print(f"{cls}: precision={prec[i]:.3f}, recall={rec[i]:.3f}, f1={f1_per_class[i]:.3f}")

    print("\nPenggunaan kombinasi modalitas:")
    for k, v in combo_counts.items():
        print(f"{k}: {v} lagu")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=EMOTION_CLASSES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


if __name__ == "__main__":
    main()
