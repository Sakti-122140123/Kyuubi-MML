import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parent
AUDIO_CSV = "audio_prob_for_fusion.csv"
LYRICS_CSV = "lyrics_prob_for_fusion.csv"
MIDI_CSV = "midi_prob_for_fusion.csv"

CLUSTERS = [f"prob_cluster_{i}" for i in range(1, 6)]


def load_modal_csv(filename: str, modal_name: str) -> pd.DataFrame:
    path = BASE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"{modal_name} file missing: {path}")

    df = pd.read_csv(path)
    required_cols = {"id", "true_label", *CLUSTERS}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{modal_name}: Missing columns {missing}")

    df = df.copy()
    df["true_label"] = df["true_label"].astype(int)
    return df


def evaluate_probs(name: str, y_true: np.ndarray, probs: np.ndarray):
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"[{name}] accuracy={acc:.4f} | macroF1={f1:.4f}")
    return acc, f1


def rename_with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.copy()
    renamed.rename(columns={c: f"{c}_{prefix}" for c in CLUSTERS}, inplace=True)
    return renamed


def prefixed_cols(prefix: str):
    return [f"{c}_{prefix}" for c in CLUSTERS]


def fuse_equal(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("No probability arrays provided for fusion.")
    return sum(arrays) / len(arrays)


def fuse_weighted(arrays, weights):
    if not arrays:
        raise ValueError("No probability arrays provided for fusion.")
    out = np.zeros_like(arrays[0])
    for arr, w in zip(arrays, weights):
        out += arr * w
    return out


def normalize_weights(weights):
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def intersect_frames(modalities, modal_frames):
    if not modalities:
        raise ValueError("Modalities list cannot be empty.")

    cols = ["id", "true_label"] + prefixed_cols(modalities[0])
    merged = modal_frames[modalities[0]][cols]

    for modal in modalities[1:]:
        cols = ["id", "true_label"] + prefixed_cols(modal)
        merged = merged.merge(
            modal_frames[modal][cols],
            on=["id", "true_label"],
            how="inner",
        )

    return merged


def fusion_block(name, df, modalities):
    if df.empty:
        print(f"[{name}] WARNING: intersection is empty, skipping.")
        return None

    y = df["true_label"].values
    comps = {}
    for modal in modalities:
        probs = df[prefixed_cols(modal)].values
        acc, f1 = evaluate_probs(f"{name} - {modal} (subset)", y, probs)
        comps[modal] = (probs, f1)

    probs_list = [val[0] for val in comps.values()]
    probs_eq = fuse_equal(*probs_list)
    acc_eq, f1_eq = evaluate_probs(f"{name} (equal fusion)", y, probs_eq)

    weights = normalize_weights([val[1] for val in comps.values()])
    probs_w = fuse_weighted(probs_list, weights)
    acc_w, f1_w = evaluate_probs(f"{name} (F1 fusion)", y, probs_w)

    return {"eq": (acc_eq, f1_eq), "f1": (acc_w, f1_w)}


def main():
    print("=== Loading modal CSVs ===")
    audio_raw = load_modal_csv(AUDIO_CSV, "audio")
    lyrics_raw = load_modal_csv(LYRICS_CSV, "lyrics")
    midi_raw = load_modal_csv(MIDI_CSV, "midi")

    print(f"Audio rows : {audio_raw.shape[0]}")
    print(f"Lyrics rows: {lyrics_raw.shape[0]}")
    print(f"MIDI rows  : {midi_raw.shape[0]}")

    # unimodal evaluations on full sets
    summary_rows = []

    def add_row(name, row_type, acc, f1):
        summary_rows.append(
            {"Model / Fusion": name, "Type": row_type, "Accuracy": acc, "Macro F1": f1}
        )

    acc, f1 = evaluate_probs(
        "Audio only (full)", audio_raw["true_label"].values, audio_raw[CLUSTERS].values
    )
    add_row("Audio only (full)", "unimodal", acc, f1)

    acc, f1 = evaluate_probs(
        "Lyrics only (full)",
        lyrics_raw["true_label"].values,
        lyrics_raw[CLUSTERS].values,
    )
    add_row("Lyrics only (full)", "unimodal", acc, f1)

    acc, f1 = evaluate_probs(
        "MIDI only (full)", midi_raw["true_label"].values, midi_raw[CLUSTERS].values
    )
    add_row("MIDI only (full)", "unimodal", acc, f1)

    # prepare prefixed frames for intersections
    modal_frames = {
        "audio": rename_with_prefix(audio_raw, "audio"),
        "lyrics": rename_with_prefix(lyrics_raw, "lyrics"),
        "midi": rename_with_prefix(midi_raw, "midi"),
    }

    combo_map = {
        "Audio+Lyrics": ["audio", "lyrics"],
        "Audio+MIDI": ["audio", "midi"],
        "Lyrics+MIDI": ["lyrics", "midi"],
        "Audio+Lyrics+MIDI": ["audio", "lyrics", "midi"],
    }

    for combo_name, mods in combo_map.items():
        subset = intersect_frames(mods, modal_frames)
        print(f"\nIntersection {combo_name}: {subset.shape[0]} rows")
        res = fusion_block(combo_name, subset, mods)
        if res is None:
            continue
        acc_eq, f1_eq = res["eq"]
        acc_w, f1_w = res["f1"]
        add_row(f"{combo_name} (equal)", "fusion_equal", acc_eq, f1_eq)
        add_row(f"{combo_name} (F1)", "fusion_f1", acc_w, f1_w)

    summary = pd.DataFrame(summary_rows)
    print("\n=== Summary Table (Ablation) ===")
    print(summary)


if __name__ == "__main__":
    main()
