# 🎵 PENGENALAN EMOSI MUSIK MULTIMODAL BERBASIS LATE FUSION PADA DATASET MULTI-MODAL MIREX  
**Tugas Besar Pembelajaran Mesin Multimodal (IF25-40304)**  
Kelompok 09 — Institut Teknologi Sumatera  

---

## 📌 Ringkasan Proyek  
Proyek ini mengembangkan sistem **Multimodal Music Emotion Recognition (MER)** untuk mengklasifikasikan lagu ke dalam **5 klaster emosi MIREX** dengan memanfaatkan tiga modalitas utama: **Audio, Lyrics, dan MIDI**.  

Pendekatan **Late Fusion** digunakan untuk menggabungkan informasi emosional dari setiap modalitas, yang diproses terlebih dahulu menggunakan encoder khusus:  
- **CRNN** untuk audio,  
- **BERT** untuk lirik,  
- **BiGRU** untuk MIDI.  

Setiap modalitas memiliki karakteristik emosional unik sehingga penggabungan output-nya diharapkan meningkatkan akurasi model dibandingkan pendekatan unimodal.

---

## 📂 Dataset  
Dataset multimodal mengacu pada kerangka kerja *MIREX Mood Classification* serta metodologi yang diperkenalkan oleh Panda et al. (2013).  

### Ketersediaan Data  
| Modalitas | Jumlah Sampel | Keterangan |
|----------|----------------|------------|
| Audio    | 903 sampel     | 100% tersedia |
| Lyrics   | 764 sampel     | ~85% dari audio |
| MIDI     | 193 sampel     | ~21% dari audio |

### Label Emosi (MIREX Clusters)
1. Passionate / Rousing / Confident / Boisterous / Rowdy  
2. Cheerful / Fun / Sweet / Amiable  
3. Poignant / Wistful / Brooding  
4. Humorous / Quirky / Witty  
5. Aggressive / Tense / Intense  

---

## 🧪 Arsitektur Model  
Arsitektur baseline terdiri dari tiga cabang pemrosesan paralel yang masing-masing menghasilkan logit atau probabilitas sebelum digabungkan melalui Late Fusion.

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

Keuntungan Late Fusion:  
- Tidak sensitif terhadap missing modality  
- Memungkinkan setiap encoder belajar optimal  
- Memberikan interpretabilitas kontribusi modalitas  

---

## 🔍 Exploratory Data Analysis (EDA) — Ringkasan  
Beberapa temuan utama:

### ✓ Intra-modal  
- **Audio**: Mel-Spectrogram menampilkan pola energi berbeda antar klaster.  
- **Lyrics**: Didominasi kata emosional seperti *love*, *pain*, *heart*. Variasi panjang teks besar → perlu padding/truncation.  
- **MIDI**: Distribusi pitch & velocity bervariasi; modalitas paling sedikit dan paling noisy.

### ✓ Inter-modal  
Setiap modalitas memuat informasi emosional berbeda → mendukung pentingnya pendekatan multimodal.

### ✓ Kualitas Label  
Distribusi klaster tidak seimbang sehingga perlu strategi training yang tepat.

### ✓ t-SNE  
Embedding tiap modalitas belum membentuk cluster emosional jelas → model nonlinear + fusion sangat diperlukan.

---

## ⚙️ Setup Eksperimen  
### Preprocessing  
- **Audio**:  
  - Resampling 22.050 Hz  
  - Mono, durasi seragam 30 detik  
  - Log-Mel Spectrogram (128 mel bands)  
- **Lyrics**:  
  - Tokenisasi BERT  
  - Max length 128/256  
  - Padding & truncation  
- **MIDI**:  
  - Ekstraksi event → embedding → BiGRU  

### Hyperparameter  
- Optimizer: Adam / AdamW  
- LR: 1e-3 (audio), 2e-5 (lyrics), 1e-3 (MIDI)  
- Batch Size: 8–16  
- Epoch: 10–20  

### Data Splitting  
- Unimodal: 80% train — 20% validation (stratified)  
- Multimodal baseline: seluruh intersection (193 sampel)

---

## 📈 Hasil Baseline (Unimodal)

### **Lyrics (BERT)**  
- Akurasi validasi ~40–45%  
- Kesalahan banyak pada kelas dengan kemiripan semantik  

### **Audio (CRNN)**  
- Akurasi maksimum sekitar 43%  
- Training cenderung underfitting  

### **MIDI (BiGRU)**  
- Performa rendah karena dataset sangat kecil  

**Kesimpulan:**  
Tidak ada modalitas yang cukup kuat secara individual → Multimodal Late Fusion sangat direkomendasikan.

---

## 🚀 Rencana Pengembangan  
- Membangun dan melatih **model multimodal Late Fusion end-to-end**  
- Uji beberapa strategi fusi: concatenation, weighted sum, attention-based fusion  
- Tambah fitur audio: MFCC, chroma, spectral contrast  
- Augmentasi audio (pitch/time shift)  
- Augmentasi MIDI (transposition)  
- Tuning hyperparameter lanjutan untuk stabilitas training  

---

## 📁 Struktur Repository  
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

## 👥 Anggota Kelompok  
- Lois Novel E. Gurning — 122140098  
- Sakti Mujahid Imani — 122140123  
- Apridian Saputra — 122140143  
- Joshia Fernandes Sectio Purba — 122140170  
- Sikah Nubuahtul Ilmi — 122140208  

---

## 📎 Lisensi  
Project ini dibuat untuk keperluan akademik dalam mata kuliah  
**Pembelajaran Mesin Multimodal (IF25-40304)**, Institut Teknologi Sumatera.

