# Multimodal Music Emotion Recognition (MER) — Late Fusion
Tugas Besar Pembelajaran Mesin Multimodal (IF25-40304)
Kelompok 09 — Institut Teknologi Sumatera

---

## Ringkasan Proyek
Proyek ini mengembangkan sistem Multimodal Music Emotion Recognition (MER) untuk mengklasifikasikan lagu ke dalam 5 klaster emosi MIREX dengan memanfaatkan tiga modalitas utama: Audio, Lyrics, dan MIDI. Pendekatan Late Fusion digunakan untuk menggabungkan keluaran dari encoder tiap modalitas sehingga mampu mengatasi batasan pada pendekatan unimodal.

---

## Dataset
Dataset mengikuti struktur MIREX Mood Classification dan terdiri dari:

- 903 Audio
- 764 Lyrics
- 193 MIDI
- 193 data lengkap (intersection) untuk model multimodal

### Label MIREX Cluster:
1. Passionate / Rousing / Confident / Boisterous  
2. Cheerful / Fun / Sweet / Amiable  
3. Poignant / Wistful / Brooding  
4. Humorous / Quirky / Witty  
5. Aggressive / Tense / Intense  

---

## Arsitektur Model
Model baseline terdiri dari tiga cabang pemrosesan paralel. Setiap modalitas menghasilkan probabilitas/logit sebelum digabungkan melalui Late Fusion.

```
Audio  → CRNN  → Classifier_A → P_A
Lyrics → BERT  → Classifier_L → P_L
MIDI   → BiGRU → Classifier_M → P_M
                ↓
        Late Fusion (+ / concat)
                ↓
             FC Layer
                ↓
          Final Output
```

### Keuntungan Late Fusion
- Tidak sensitif terhadap missing modality  
- Masing-masing encoder dapat belajar optimal  
- Interpretasi kontribusi modalitas lebih jelas  

---

## Ringkasan EDA
- **Audio:** Mel-Spectrogram menunjukkan pola energi berbeda antar klaster  
- **Lyrics:** Didominasi kata emosional seperti *love*, *heart*, *pain*  
- **MIDI:** Distribusi pitch & velocity bervariasi  
- **t-SNE:** Embedding belum membentuk cluster tegas → fusion + model nonlinear dibutuhkan  

---

## Setup Eksperimen

### Preprocessing  
**Audio**
- Resample 22.050 Hz  
- Mono, durasi 30 detik  
- Log-mel spectrogram (128 bands)  

**Lyrics**
- Tokenisasi BERT  
- Max length 128/256  

**MIDI**
- Event extraction → embedding → BiGRU  

### Hyperparameter  
- LR: 1e-3 (audio & MIDI), 2e-5 (lyrics)  
- Optimizer: Adam / AdamW  
- Batch size: 8–16  
- Epoch: 10–20  

### Data Splitting  
- Unimodal: stratified 80/20  
- Multimodal: memakai 193 intersection samples  

---

## Hasil Baseline

### Lyrics (BERT)
- Akurasi ~40–45%  

### Audio (CRNN)
- Akurasi ~43%  

### MIDI (BiGRU)
- Paling rendah karena sedikit data  

**Kesimpulan:**  
Modalitas tunggal tidak cukup kuat → multimodal Late Fusion diperlukan.

---

## Rencana Pengembangan  
- Implementasi Late Fusion end-to-end  
- Variasi fusion: weighted, concatenation, attention fusion  
- Augmentasi audio dan MIDI  
- Penambahan fitur audio tambahan  
- Fine-tuning BERT lebih stabil  

---

## Struktur Repository
```
.
├── data/
├── src/
│   ├── audio_model/
│   ├── lyrics_model/
│   ├── midi_model/
│   ├── fusion/
│   └── utils/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Baseline_Audio.ipynb
│   ├── Baseline_Lyrics.ipynb
│   ├── Baseline_MIDI.ipynb
│   └── Fusion_Model.ipynb
├── reports/
│   ├── Proposal.pdf
│   ├── EDA.pdf
│   ├── Preliminary_Experiment.pdf
│   └── Final_Report.pdf
└── README.md
```

---

## Anggota Kelompok  
- Lois Novel E. Gurning  
- Sakti Mujahid Imani  
- Apridian Saputra  
- Joshia Fernandes Sectio Purba  
- Sikah Nubuahtul Ilmi  

---

## Lisensi  
Project ini dibuat untuk keperluan akademik mata kuliah Pembelajaran Mesin Multimodal (IF25-40304), Institut Teknologi Sumatera.
