# 🎵 Multimodal Music Emotion Recognition

## Pengenalan Emosi Musik Berbasis Late Fusion pada Dataset MIREX

**Tugas Besar Pembelajaran Mesin Multimodal (IF25-40304)**  
**Kelompok 09** — Institut Teknologi Sumatera  
**Semester Ganjil 2024/2025**

---

## 👥 Anggota Kelompok

| Nama                          | NIM       |
| ----------------------------- | --------- |
| Lois Novel E. Gurning         | 122140098 |
| Sakti Mujahid Imani           | 122140123 |
| Apridian Saputra              | 122140143 |
| Joshia Fernandes Sectio Purba | 122140170 |
| Sikah Nubuahtul Ilmi          | 122140208 |

---

## 📌 Ringkasan Project

Sistem **Multimodal Music Emotion Recognition (MER)** untuk mengklasifikasikan lagu ke dalam **5 klaster emosi MIREX** menggunakan tiga modalitas:

- 🎵 **Audio** - Analisis sinyal audio
- 📝 **Lyrics** - Analisis teks lirik
- 🎹 **MIDI** - Analisis data musik symbolic

### Strategi Pendekatan

**Late Fusion** - Menggabungkan output probability dari setiap modalitas untuk prediksi akhir yang lebih akurat.

### Model yang Digunakan

**Baseline (Milestone 3):**

- Audio: **CRNN** (Convolutional Recurrent Neural Network)
- Lyrics: **BERT base** (bert-base-uncased)
- MIDI: **BiGRU + Attention**

**Improved (Milestone 4):**

- Audio: **PANN** (Pre-trained Audio Neural Network - Cnn14)
- Lyrics: **DeBERTa-v3-base** (Enhanced attention mechanism)
- MIDI: **BiGRU + SVM** (Robust untuk small dataset)

> **💡 Catatan**: Project ini menerapkan **iterative improvement** dari baseline ke model yang lebih advanced, dengan dokumentasi lengkap untuk menunjukkan progression dan justifikasi.

---

## 📂 Struktur Repository

```
Kyuubi-MML/
│
├── 📄 README.md                    # Dokumentasi utama (file ini)
│
├── 📁 data/                        # Dataset & metadata
│   ├── master_tracks.csv           # Metadata 903 lagu
│   └── split_global.csv            # Train/val/test split
│
├── 📁 notebooks/                   # Jupyter Notebooks
│   ├── 01_EDA/                     # Exploratory Data Analysis
│   │   ├── 01_EDA_Multimodal.ipynb
│   │   └── 02_Data_Splitting.ipynb
│   ├── 02_Preprocessing/           # Preprocessing pipelines
│   │   ├── 01_Audio_Preprocessing.ipynb
│   │   └── 02_Lyrics_Preprocessing.ipynb
│   ├── 03_Baseline/                # Baseline models
│   │   ├── 01_Audio_CRNN.ipynb
│   │   ├── 02_Lyrics_BERT.ipynb
│   │   └── 03_MIDI_BiGRU_Attention.ipynb
│   ├── 04_Improved/                # Improved models
│   │   ├── 01_Audio_PANN.ipynb
│   │   ├── 02_Lyrics_DeBERTa.ipynb
│   │   ├── 03_MIDI_BiGRU_SVM.ipynb
│   │   └── 03_MIDI_Complete_Pipeline.ipynb
│   └── 05_Fusion/                  # Multimodal fusion
│       ├── fusion.py
│       ├── smart_fusion.py
│       └── fusion_evaluation_finale.py
│
├── 📁 results/                     # Hasil eksperimen
│   ├── baseline/                   # Baseline results
│   ├── improved/                   # Improved results
│   └── fusion/                     # Fusion results
│
├── 📁 models/                      # Saved model checkpoints
├── 📁 reports/                     # Laporan milestone
├── 📁 figures/                     # Visualisasi & plot
├── 📁 docs/                        # Dokumentasi tambahan
└── 📁 miditrainsvm/               # MIDI training artifacts

```

---

## 📊 Dataset

### Sumber

- **Framework**: MIREX (Music Information Retrieval Evaluation eXchange)
- **Reference**: Panda et al. (2013)

### Ketersediaan Data

| Modalitas | Jumlah Sampel | Coverage |
| --------- | ------------- | -------- |
| Audio     | 903           | 100%     |
| Lyrics    | 764           | ~85%     |
| MIDI      | 193           | ~21%     |

### Label Emosi (5 Cluster MIREX)

1. **Cluster 1**: Passionate / Rousing / Confident / Boisterous / Rowdy
2. **Cluster 2**: Cheerful / Fun / Sweet / Amiable
3. **Cluster 3**: Poignant / Wistful / Brooding
4. **Cluster 4**: Humorous / Quirky / Witty
5. **Cluster 5**: Aggressive / Tense / Intense

### Data Splitting

- **Train**: ~70-80%
- **Validation**: ~10-15%
- **Test**: ~10-15%
- **Strategy**: Stratified split untuk maintain class balance

---

## 🔬 Metodologi

### 1. Exploratory Data Analysis (EDA)

**Analisis Intra-Modal** - Per modalitas

- Audio: Mel-spectrogram patterns, duration distribution
- Lyrics: Word frequency, text length, common words per cluster
- MIDI: Pitch/velocity distribution, duration patterns

**Analisis Inter-Modal** - Antar modalitas

- Correlation analysis
- Audio-Lyrics-MIDI alignment
- Modality availability matrix

**Analisis Target** - Terhadap label

- Class imbalance detection
- Feature importance per cluster

**Visualisasi t-SNE**

- Feature embeddings visualization
- Cluster quality assessment

### 2. Preprocessing

**Audio:**

- Sample rate: 32,000 Hz (PANN) / 22,050 Hz (CRNN)
- Duration: 10 detik uniform
- Feature: Log-Mel Spectrogram (128 mel bands)
- Augmentation: Multi-crop strategy (start, middle, end)

**Lyrics:**

- Tokenization: BERT/DeBERTa tokenizer
- Max length: 256 tokens
- Padding & truncation
- Lowercase normalization

**MIDI:**

- Event extraction (pitch, velocity, duration)
- Embedding layer
- Sequence padding

### 3. Model Architecture

#### Baseline Models (Milestone 3)

**Audio: CRNN**

```
Input (Mel-Spec) → CNN layers → RNN layers → Dense → Softmax (5 classes)
```

- Performance: ~43% accuracy, ~0.38 macro F1
- Issue: Underfitting, butuh pre-trained model

**Lyrics: BERT Base**

```
Input (tokens) → BERT encoder → Pooler → Classifier → Softmax (5 classes)
```

- Performance: ~42% accuracy, ~0.40 macro F1
- Issue: Semantic similarity causing confusion

**MIDI: BiGRU + Attention**

```
Input (events) → Embedding → BiGRU → Attention → Dense → Softmax (5 classes)
```

- Performance: ~25% accuracy, ~0.20 macro F1
- Issue: Dataset terlalu kecil, overfitting

#### Improved Models (Milestone 4)

**Audio: PANN (Cnn14)**

```
Input → Pre-trained Cnn14 → Feature extractor → Fine-tuned classifier → Softmax
```

- Pre-trained on AudioSet
- Multi-crop inference strategy
- Expected: Better audio representation

**Lyrics: DeBERTa-v3-base**

```
Input → DeBERTa encoder (disentangled attention) → Pooler → Classifier → Softmax
```

- Enhanced mask decoder
- Layer freezing strategy (freeze lower 0-7, fine-tune upper)
- Expected: Better semantic understanding

**MIDI: BiGRU + SVM**

```
Input → BiGRU (frozen) → Feature extraction → SVM classifier (RBF kernel) → Softmax
```

- BiGRU as feature extractor
- SVM with balanced class weights
- Expected: Robust untuk small dataset, avoid overfitting

### 4. Fusion Strategy

**Simple Average Fusion**

```python
P_final = (P_audio + P_lyrics + P_midi) / 3
```

**F1-Weighted Fusion**

```python
w_i = F1_i / (F1_audio + F1_lyrics + F1_midi)
P_final = w_audio * P_audio + w_lyrics * P_lyrics + w_midi * P_midi
```

**Smart Fusion (Missing Modality Handling)**

- Adaptive per-sample fusion
- Supports partial modality combinations
- Coverage: 903 samples (semua audio)

---

## 📈 Hasil Eksperimen

### Unimodal Performance

#### Baseline Results

| Model      | Modality | Accuracy | Macro F1 | Notes              |
| ---------- | -------- | -------- | -------- | ------------------ |
| CRNN       | Audio    | ~43%     | ~0.38    | Underfitting       |
| BERT       | Lyrics   | ~42%     | ~0.40    | Semantic confusion |
| BiGRU+Attn | MIDI     | ~25%     | ~0.20    | Small dataset      |

#### Improved Results

| Model     | Modality | Improvement    | Expected Gain         |
| --------- | -------- | -------------- | --------------------- |
| PANN      | Audio    | Pre-trained    | Better representation |
| DeBERTa   | Lyrics   | Enhanced attn  | Better semantics      |
| BiGRU+SVM | MIDI     | SVM classifier | Avoid overfitting     |

### Multimodal Fusion

**Ablation Study (pada intersection samples)**

| Combination    | Strategy     | N Samples | Performance       |
| -------------- | ------------ | --------- | ----------------- |
| Audio only     | -            | 903       | Baseline unimodal |
| Lyrics only    | -            | 764       | Baseline unimodal |
| MIDI only      | -            | 193       | Baseline unimodal |
| Audio + Lyrics | Simple avg   | 764       | Multimodal boost  |
| Audio + MIDI   | Simple avg   | 193       | Multimodal boost  |
| Lyrics + MIDI  | Simple avg   | 193       | Multimodal boost  |
| All (Full)     | Smart fusion | 903       | **Best coverage** |

**Key Findings:**

- ✅ Multimodal fusion > best unimodal
- ✅ Smart fusion memberikan coverage terluas
- ✅ F1-weighted lebih baik dari simple average
- ⚠️ MIDI contribution terbatas karena dataset kecil

---

## 🎯 Milestone Progress

### ✅ Milestone 1: Proposal

- [x] Dokumen proposal (5-7 halaman)
- [x] Slide presentasi (10-15 menit)
- [x] Latar belakang, rumusan masalah, tujuan
- [x] Deskripsi dataset & rencana metode
- **Deliverable**: `reports/Proposal.pdf`

### ✅ Milestone 2: EDA Multimodal

- [x] Analisis intra-modal (Audio, Lyrics, MIDI)
- [x] Analisis inter-modal & target
- [x] Visualisasi t-SNE
- [x] Identifikasi masalah data
- **Deliverables**:
  - `notebooks/01_EDA/01_EDA_Multimodal.ipynb`
  - `reports/EDA Multimodal Kelompok 09.pdf`

### ✅ Milestone 3: Preliminary Experiment

- [x] Baseline models (CRNN, BERT, BiGRU+Attention)
- [x] Setup eksperimen & hyperparameters
- [x] Hasil baseline & learning curves
- [x] Error analysis
- [x] Rencana optimalisasi
- **Deliverables**:
  - `notebooks/03_Baseline/*.ipynb`
  - `reports/Preliminary Experiment Kelompok 09.pdf`

### ✅ Milestone 4: Laporan Akhir

- [x] Improved models (PANN, DeBERTa, BiGRU+SVM)
- [x] Multimodal fusion experiments
- [x] Evaluation & comparison
- **Deliverables**:
  - `reports/Final Project.pdf`

---

## 📚 Referensi Utama

1. **MIREX Dataset**

   - Panda et al. (2013) - Multi-modal Music Emotion Recognition

---

## 🔧 Technical Stack

**Deep Learning Frameworks:**

- PyTorch 2.0+
- Transformers (Hugging Face)
- torchaudio

**Audio Processing:**

- librosa
- pretty_midi
- PANNs (audioset_tagging_cnn)

**Machine Learning:**

- scikit-learn (SVM, metrics)
- numpy, pandas

**Visualization:**

- matplotlib, seaborn
- t-SNE

---

## 📝 Catatan Penting

### Perubahan dari Baseline ke Improved

**Motivasi Improvement:**

1. **Audio (CRNN → PANN)**

   - CRNN underfitting karena kurang data training
   - PANN pre-trained pada AudioSet (2M+ audio clips)
   - Transfer learning memberikan better feature extraction

2. **Lyrics (BERT → DeBERTa)**

   - BERT kesulitan dengan semantic similarity
   - DeBERTa punya disentangled attention mechanism
   - Lebih baik dalam contextual understanding

3. **MIDI (BiGRU+Attn → BiGRU+SVM)**
   - Dataset MIDI sangat kecil (193 samples)
   - Neural network classifier cenderung overfit
   - SVM lebih robust untuk small data
   - BiGRU tetap digunakan sebagai feature extractor

### File Naming Convention

**Baseline Results:**

- `results/baseline/audio_prob.csv` - CRNN probabilities
- `results/baseline/lyric_prob.csv` - BERT probabilities
- `results/baseline/midi_prob.csv` - BiGRU+Attention probabilities

**Improved Results:**

- `results/improved/audio_prob_for_fusion.csv` - PANN probabilities
- `results/improved/lyrics_prob_for_fusion2.csv` - DeBERTa probabilities
- `results/improved/midi_prob_for_fusion.csv` - BiGRU+SVM probabilities

---

## 🤝 Kontribusi Tim

Pembagian peran dalam project:

- **Joshia**: EDA Audio, CRNN baseline, PANN improvement
- **Apridian**: EDA Lyrics, BERT baseline, DeBERTa improvement
- **Sikah**: EDA MIDI, BiGRU baseline, BiGRU+SVM improvement
- **Louis**: Fusion strategy, evaluation, comparison
- **Sakti**: Documentation, visualization, report writing

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik dalam mata kuliah **Pembelajaran Mesin Multimodal (IF25-40304)**, Institut Teknologi Sumatera.

---

## 🙏 Acknowledgments

Terima kasih kepada:

- Dosen pengampu mata kuliah Pembelajaran Mesin Multimodal, Bapak I Wayan Wiprayoga Wisesa, S.Kom., M.Kom.
- Penyedia dataset MIREX

---

<p align="center">
  <strong>🎵 Made with ❤️ by Kelompok 09 🎵</strong>
</p>
