import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ===============================
# Konfigurasi dasar
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sesuaikan kalau namanya beda
AUDIO_CSV  = "audio_prob_for_fusion.csv"
LYRICS_CSV = "lyrics_prob_for_fusion2.csv"
MIDI_CSV   = "midi_prob_for_fusion.csv"

# Di CSV pipeline baru: true_label = 0..4
LABELS = [0, 1, 2, 3, 4]
EMOTION_CLASSES = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]


# ===============================
# Fungsi bantu
# ===============================
def load_modal_csv(filename, modal_name):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {filename} ({path}) tidak ditemukan.")

    df = pd.read_csv(path)

    required_cols = ["id", "true_label"] + [f"prob_cluster_{i}" for i in range(1, 6)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{modal_name}: kolom berikut hilang: {missing}")

    df["true_label"] = df["true_label"].astype(int)
    return df


def evaluate_probs(y_true, probs, name="model"):
    """
    y_true: (N,) label 0..4
    probs : (N, 5) -> prob untuk Cluster 1..5 (index 0..4)
    """
    y_pred = probs.argmax(axis=1)  # label 0..4 (TANPA +1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"[{name}] acc={acc:.4f} | macroF1={f1m:.4f}")
    return acc, f1m, y_pred


def confusion_plot(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"C{i+1}" for i in LABELS])  # tampilkan 1..5
    ax.set_yticklabels([f"C{i+1}" for i in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    print("Confusion matrix:\n", cm)


def normalize_weights(ws):
    s = sum(ws)
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [w / s for w in ws]


def fuse_equal(*probs_list):
    n = len(probs_list)
    return sum(probs_list) / float(n)


def fuse_weighted(probs_list, weights):
    out = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        out += w * p
    return out


def add_prefix(df, prefix):
    """Rename prob_cluster_i -> prob_cluster_i_prefix"""
    df = df.copy()
    for i in range(1, 6):
        df.rename(columns={f"prob_cluster_{i}": f"prob_cluster_{i}_{prefix}"}, inplace=True)
    return df


# ===============================
# Main
# ===============================
def main():
    print("=== Load CSV ===")
    audio_raw  = load_modal_csv(AUDIO_CSV,  "audio")
    lyrics_raw = load_modal_csv(LYRICS_CSV, "lyrics")
    midi_raw   = load_modal_csv(MIDI_CSV,   "midi")

    print("Audio  :", audio_raw.shape)
    print("Lyrics :", lyrics_raw.shape)
    print("MIDI   :", midi_raw.shape)

    # -----------------------------
    # Unimodal (langsung dari masing-masing CSV - full set test masing2)
    # -----------------------------
    print("\n=== Unimodal performance (masing-masing modalitas) ===")

    # Audio
    y_audio = audio_raw["true_label"].values
    probs_audio_full = audio_raw[[f"prob_cluster_{i}" for i in range(1, 6)]].values
    acc_a, f1_a, ypred_a = evaluate_probs(y_audio, probs_audio_full, "Audio only (full set)")

    # Lyrics
    y_lyrics = lyrics_raw["true_label"].values
    probs_lyrics_full = lyrics_raw[[f"prob_cluster_{i}" for i in range(1, 6)]].values
    acc_l, f1_l, ypred_l = evaluate_probs(y_lyrics, probs_lyrics_full, "Lyrics only (full set)")

    # MIDI
    y_midi = midi_raw["true_label"].values
    probs_midi_full = midi_raw[[f"prob_cluster_{i}" for i in range(1, 6)]].values
    acc_m, f1_m, ypred_m = evaluate_probs(y_midi, probs_midi_full, "MIDI only (full set)")

    # Untuk fusion, pakai versi dengan prefix kolom supaya merge aman
    audio  = add_prefix(audio_raw,  "audio")
    lyrics = add_prefix(lyrics_raw, "lyrics")
    midi   = add_prefix(midi_raw,   "midi")

    # -----------------------------
    # Intersection: Audio+Lyrics, Audio+MIDI, Lyrics+MIDI, Audio+Lyrics+MIDI
    # -----------------------------
    merged_al = audio.merge(
        lyrics[["id", "true_label"] + [c for c in lyrics.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="inner"
    )
    print("\nIntersection Audio+Lyrics:", merged_al.shape[0], "lagu")

    merged_am = audio.merge(
        midi[["id", "true_label"] + [c for c in midi.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="inner"
    )
    print("Intersection Audio+MIDI  :", merged_am.shape[0], "lagu")

    merged_lm = lyrics.merge(
        midi[["id", "true_label"] + [c for c in midi.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="inner"
    )
    print("Intersection Lyrics+MIDI :", merged_lm.shape[0], "lagu")

    merged_alm = merged_al.merge(
        midi[["id", "true_label"] + [c for c in midi.columns if "prob_cluster_" in c]],
        on=["id", "true_label"],
        how="inner"
    )
    print("Intersection ALL 3       :", merged_alm.shape[0], "lagu")

    # -----------------------------
    # Fusion eksperimen (per subset)
    # -----------------------------
    rows = []

    def fusion_block(name, df, use_audio=False, use_lyrics=False, use_midi=False):
        if df.shape[0] == 0:
            print(f"[{name}] WARNING: tidak ada lagu pada intersection, dilewati.")
            return None

        y = df["true_label"].values

        # kumpulkan probs & F1 unimodal di subset ini
        comps = {}
        if use_audio:
            pa = df[[f"prob_cluster_{i}_audio" for i in range(1, 6)]].values
            acc, f1, _ = evaluate_probs(y, pa, f"{name} - Audio only (subset)")
            comps["audio"] = (pa, f1)
        if use_lyrics:
            pl = df[[f"prob_cluster_{i}_lyrics" for i in range(1, 6)]].values
            acc, f1, _ = evaluate_probs(y, pl, f"{name} - Lyrics only (subset)")
            comps["lyrics"] = (pl, f1)
        if use_midi:
            pm = df[[f"prob_cluster_{i}_midi" for i in range(1, 6)]].values
            acc, f1, _ = evaluate_probs(y, pm, f"{name} - MIDI only (subset)")
            comps["midi"] = (pm, f1)

        # equal-weight fusion
        probs_list_eq = [v[0] for v in comps.values()]
        probs_eq = fuse_equal(*probs_list_eq)
        acc_eq, f1_eq, ypred_eq = evaluate_probs(y, probs_eq, f"{name} (equal-weight fusion)")

        # F1-based fusion (pakai F1 subset aktual)
        f1s = [v[1] for v in comps.values()]
        ws = normalize_weights(f1s)
        probs_w = fuse_weighted([v[0] for v in comps.values()], ws)
        acc_w, f1_w, ypred_w = evaluate_probs(y, probs_w, f"{name} (F1-weighted fusion)")

        return {
            "y_true": y,
            "comps": comps,
            "eq": (acc_eq, f1_eq, ypred_eq),
            "w":  (acc_w,  f1_w,  ypred_w),
        }

    print("\n=== Fusion: Audio + Lyrics ===")
    res_al = fusion_block("Audio+Lyrics", merged_al, use_audio=True, use_lyrics=True, use_midi=False)

    print("\n=== Fusion: Audio + MIDI ===")
    res_am = fusion_block("Audio+MIDI", merged_am, use_audio=True, use_lyrics=False, use_midi=True)

    print("\n=== Fusion: Lyrics + MIDI ===")
    res_lm = fusion_block("Lyrics+MIDI", merged_lm, use_audio=False, use_lyrics=True, use_midi=True)

    print("\n=== Fusion: Audio + Lyrics + MIDI ===")
    res_alm = fusion_block("Audio+Lyrics+MIDI", merged_alm, use_audio=True, use_lyrics=True, use_midi=True)

    # -----------------------------
    # Summary tabel ablation
    # -----------------------------
    def add_row(name, ftype, acc, f1):
        rows.append([name, ftype, acc, f1])

    # Unimodal (full set)
    add_row("Audio only (full)",  "unimodal", acc_a, f1_a)
    add_row("Lyrics only (full)", "unimodal", acc_l, f1_l)
    add_row("MIDI only (full)",   "unimodal", acc_m, f1_m)

    def add_from_res(tag, res):
        if res is None:
            return
        acc_eq, f1_eq, _ = res["eq"]
        acc_w,  f1_w,  _ = res["w"]
        add_row(f"{tag} (equal)", "fusion_equal", acc_eq, f1_eq)
        add_row(f"{tag} (F1)",    "fusion_f1",    acc_w,  f1_w)

    add_from_res("Audio+Lyrics",      res_al)
    add_from_res("Audio+MIDI",        res_am)
    add_from_res("Lyrics+MIDI",       res_lm)
    add_from_res("Audio+Lyrics+MIDI", res_alm)

    results = pd.DataFrame(rows, columns=["Model / Fusion", "Type", "Accuracy", "Macro F1"])
    print("\n=== Summary Table (Ablation) ===")
    print(results)

    out_path = os.path.join(BASE_DIR, "fusion_results_summary.csv")
    results.to_csv(out_path, index=False)
    print("\nSaved summary to:", out_path)

    # -----------------------------
    # Pilih baseline & best model (berdasarkan Macro F1)
    # -----------------------------
    unimodal = results[results["Type"] == "unimodal"].sort_values("Macro F1", ascending=False)
    baseline_name = unimodal.iloc[0]["Model / Fusion"]
    best = results.sort_values("Macro F1", ascending=False).iloc[0]
    best_name = best["Model / Fusion"]

    print("\nBaseline (unimodal terbaik):", baseline_name)
    print("Best overall model:", best_name)

    # -----------------------------
    # Confusion matrix baseline & best
    # -----------------------------
    # pred map untuk unimodal full
    baseline_pred_map = {
        "Audio only (full)":  (y_audio,  ypred_a),
        "Lyrics only (full)": (y_lyrics, ypred_l),
        "MIDI only (full)":   (y_midi,   ypred_m),
    }

    if baseline_name in baseline_pred_map:
        yb_true, yb_pred = baseline_pred_map[baseline_name]
        print("\n=== Confusion matrix: Baseline ({}) ===".format(baseline_name))
        confusion_plot(yb_true, yb_pred, title=f"Baseline: {baseline_name}")
    else:
        print("\n[WARNING] Baseline name tidak ada di mapping prediksi, skip confusion matrix baseline.")

    # Best model: cek apakah termasuk fusion
    def confusion_for_fusion(tag, res, mode):
        if res is None:
            return
        y = res["y_true"]
        if mode == "equal":
            _, _, yp = res["eq"]
        else:
            _, _, yp = res["w"]
        confusion_plot(y, yp, title=f"{tag} ({mode})")
        print("\nClassification report -", tag, f"({mode})")
        print(classification_report(y, yp, labels=LABELS, target_names=EMOTION_CLASSES, zero_division=0))

    print("\n=== Confusion matrix: Best ({}) ===".format(best_name))
    if "Audio+Lyrics+MIDI (F1)" == best_name and res_alm is not None:
        confusion_for_fusion("Audio+Lyrics+MIDI", res_alm, "F1")
    elif "Audio+Lyrics+MIDI (equal)" == best_name and res_alm is not None:
        confusion_for_fusion("Audio+Lyrics+MIDI", res_alm, "equal")
    elif "Audio+Lyrics (F1)" == best_name and res_al is not None:
        confusion_for_fusion("Audio+Lyrics", res_al, "F1")
    elif "Audio+Lyrics (equal)" == best_name and res_al is not None:
        confusion_for_fusion("Audio+Lyrics", res_al, "equal")
    elif "Audio+MIDI (F1)" == best_name and res_am is not None:
        confusion_for_fusion("Audio+MIDI", res_am, "F1")
    elif "Audio+MIDI (equal)" == best_name and res_am is not None:
        confusion_for_fusion("Audio+MIDI", res_am, "equal")
    elif "Lyrics+MIDI (F1)" == best_name and res_lm is not None:
        confusion_for_fusion("Lyrics+MIDI", res_lm, "F1")
    elif "Lyrics+MIDI (equal)" == best_name and res_lm is not None:
        confusion_for_fusion("Lyrics+MIDI", res_lm, "equal")
    elif best_name in baseline_pred_map:
        yb_true, yb_pred = baseline_pred_map[best_name]
        confusion_plot(yb_true, yb_pred, title=f"Best: {best_name}")
        print("\nClassification report - Best baseline")
        print(classification_report(yb_true, yb_pred, labels=LABELS, target_names=EMOTION_CLASSES, zero_division=0))
    else:
        print("[WARNING] Tidak bisa buat confusion matrix untuk best_name:", best_name)


if __name__ == "__main__":
    main()
