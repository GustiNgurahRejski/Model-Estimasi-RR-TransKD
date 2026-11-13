# RR Estimation via Transformer + Knowledge Distillation

Kode pendukung tesis: *Transformer-Based Respiratory Rate (RR) Estimation
using Knowledge Distillation for Photoplethysmogram (PPG) Signal Processing:
A Simulation Study for Wearable Device Applications*.

---

## 1. Struktur Folder

```
rr_estimation_kd/
├── config.py                     # SEMUA parameter tuneable ada di sini
├── utils.py                      # seed, device, checkpoint, logger
│
├── src/
│   ├── preprocessing.py          # interpolasi, filter, downsample, normalisasi, segmentasi
│   ├── dataset.py                # split 70/15/15, PyTorch Dataset, caching
│   ├── distillation.py           # loss function Knowledge Distillation
│   ├── engine.py                 # loop training / validation / testing
│   ├── efficiency.py             # ukur parameter, ukuran file, latency, simulasi wearable
│   └── models/
│       ├── transformer.py        # arsitektur Transformer (dipakai Teacher & Student)
│       └── baselines.py          # CNN-LSTM, ConvMixer, Plain Transformer (utk benchmark)
│
├── train_teacher.py              # skrip CLI: latih Teacher
├── train_student.py              # skrip CLI: latih Student via KD
├── evaluate.py                   # skrip CLI: evaluasi 1 checkpoint model apapun
├── run_ablation_preprocessing.py # Skenario Pengujian #1
├── run_benchmark_models.py       # Skenario Pengujian #4
├── run_efficiency_test.py        # Skenario Pengujian #2 & #3
│
├── data/
│   ├── README.md                 # cara menyiapkan dataset BIDMC mentah
│   ├── raw/bidmc_csv/             # <-- taruh file CSV BIDMC di sini
│   └── processed/                 # cache hasil preprocessing (otomatis dibuat)
│
├── checkpoints/                   # model tersimpan (.pt) - otomatis dibuat
├── results/
│   ├── logs/                      # log training per-epoch (.csv) - otomatis dibuat
│   └── *.csv                      # ringkasan tiap eksperimen
│
└── requirements.txt
```

**Kenapa dipecah per file (bukan 1 file besar)?**
Karena kamu berencana menjalankan BANYAK skenario pengujian (ablation
preprocessing, teacher vs student, simulasi wearable, benchmark arsitektur
lain), memisahkan tiap tanggung jawab membuat:
- Kode preprocessing/model/training bisa dipakai ulang di semua skenario
  tanpa copy-paste (mengurangi risiko bug beda-beda antar eksperimen).
- Kamu bisa ubah SATU hal (misal arsitektur Student) tanpa menyentuh kode
  preprocessing atau training loop.
- Semua parameter terkumpul di `config.py`, jadi kamu tinggal ubah 1 tempat
  atau kirim argumen CLI untuk mengganti skenario, sesuai permintaanmu.

---

## 2. Alur Kerja End-to-End (Detail)

### Tahap 0 — Instalasi
```bash
# buat & aktifkan virtual environment (opsional tapi disarankan)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# install PyTorch versi GPU SESUAI CUDA laptopmu (cek di nvidia-smi)
# contoh utk CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# baru install sisanya
pip install -r requirements.txt
```

Cek GPU terbaca:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Tahap 1 — Siapkan Dataset
Ikuti `data/README.md`. Intinya: taruh file CSV BIDMC (per subjek) di
`data/raw/bidmc_csv/`.

### Tahap 2 — Preprocessing (otomatis, dengan caching)
Kamu **tidak perlu** menjalankan script preprocessing terpisah. Setiap kali
`get_dataloaders(cfg)` dipanggil (di dalam `train_teacher.py`, dsb), kode akan:
1. Menghitung "cache key" dari kombinasi parameter preprocessing yang aktif
   di `cfg.preprocess` (lihat `src/dataset.py::_cache_key`).
2. Kalau cache dgn key tsb sudah ada di `data/processed/`, langsung dipakai
   (cepat, tidak mengulang filtering dsb).
3. Kalau belum ada, `src/preprocessing.py` dijalankan dari data mentah utk
   SEMUA subjek, hasilnya berupa list segmen 30 detik (750 sample @25Hz)
   beserta label RR-nya, lalu disimpan ke cache.
4. Segmen displit per SUBJEK (bukan per-window) menjadi 70% train / 15%
   validation / 15% test — supaya window dari 1 pasien yang sama tidak
   "bocor" ke lebih dari satu split.

Alur detail per subjek (`src/preprocessing.py::process_subject`):
```
Raw PLETH (125Hz) + anotasi napas (ann1, ann2)
   -> interpolasi NaN
   -> downsampling 5x -> 25Hz
   -> [opsional] Butterworth bandpass 0.1-0.4 Hz
   -> [opsional] Z-score normalization
   -> segmentasi window 30 detik (750 sample)
   -> hitung RR label dari rata-rata jumlah napas ann1 & ann2 pada window tsb
   -> [opsional] quality filtering (buang window flat/NaN/RR di luar 5-50 bpm/
                 disagreement 2 annotator > 2 bpm)
```
Tiap tanda `[opsional]` bisa dinyalakan/dimatikan lewat `config.py` — inilah
yang dipakai untuk **Skenario Pengujian #1 (ablation preprocessing)**.

### Tahap 3 — Training Teacher
```bash
python train_teacher.py --preset full_preprocessing --tag teacher_full
```
Ini akan:
- Membangun Teacher Transformer (`d_model=128, heads=8, layers=4`)
- Training standar (loss = MAE) dengan validation tiap epoch, early stopping,
  dan LR scheduler `ReduceLROnPlateau`
- Menyimpan checkpoint terbaik (berdasarkan val MAE) ke
  `checkpoints/teacher_full_best.pt`
- Mencetak metrik test set (MAE, RMSE, Acc@2bpm) & laporan efisiensi

### Tahap 4 — Training Student via Knowledge Distillation
```bash
python train_student.py --teacher_ckpt checkpoints/teacher_full_best.pt \
                         --preset full_preprocessing --tag student_full
```
Ini akan:
- Memuat Teacher yang sudah dilatih, membekukan (`freeze`) bobotnya
- Membangun Student Transformer (`d_model=32, heads=2, layers=2`)
- Setiap batch: hitung `y_teacher` (tanpa gradient) & `y_student`, lalu
  `L_total = alpha * MSE(y_student, y_teacher) + (1-alpha) * MAE(y_student, y_true)`
- Hanya bobot Student yang di-update (Teacher tetap beku)
- Simpan checkpoint terbaik ke `checkpoints/student_full_best.pt`

**PENTING:** preset preprocessing pada `train_student.py` harus SAMA dengan
yang dipakai saat melatih teacher-nya (`teacher_ckpt`), supaya soft-target
Teacher valid untuk data yang dilihat Student.

### Tahap 5 — Evaluasi / Testing manual (opsional)
```bash
python evaluate.py --model_type student --ckpt checkpoints/student_full_best.pt \
                    --preset full_preprocessing
```

---

## 3. Menjalankan Skenario-Skenario Pengujian Spesifik

### Skenario #1 — Pengaruh Preprocessing (ablation study)
```bash
python run_ablation_preprocessing.py --epochs 50
```
Otomatis melatih 4 varian preprocessing (`no_preprocessing`, `filter_only`,
`filter_norm`, `full_preprocessing`) memakai model Teacher, lalu menyimpan
tabel perbandingan MAE/RMSE/Acc di
`results/ablation_preprocessing_summary.csv`.

Kalau ingin custom kombinasi sendiri, edit fungsi `_apply_preset()` di
`config.py`, atau jalankan satu-satu manual:
```bash
python train_teacher.py --preset no_preprocessing --tag t_raw
python train_teacher.py --preset filter_only --tag t_filter
python train_teacher.py --preset filter_norm --tag t_filternorm
python train_teacher.py --preset full_preprocessing --tag t_full
```

### Skenario #2 & #3 — Teacher vs Student + Simulasi Wearable
```bash
python run_efficiency_test.py \
    --teacher_ckpt checkpoints/teacher_full_best.pt \
    --student_ckpt checkpoints/student_full_best.pt \
    --preset full_preprocessing
```
Default-nya sudah memakai simulasi wearable (CPU-only, `batch_size=1`,
`cpu_threads=1`, lihat `EfficiencyConfig` di `config.py`). Ini akan
mencetak & menyimpan (`results/efficiency_teacher_vs_student.csv`):
- Jumlah parameter, ukuran file model (KB), inference latency (ms) masing2
- Rasio kompresi (parameter, latency, ukuran file) Teacher -> Student
- Selisih MAE Teacher vs Student

Untuk mengukur di kondisi NORMAL (pakai GPU, batch besar seperti training):
```bash
python run_efficiency_test.py --teacher_ckpt ... --student_ckpt ... --no_wearable_sim
```

Untuk mensimulasikan device yang lebih terbatas lagi (mis. 1 core, batch=1)
vs yang agak longgar (mis. 2 core):
```bash
python run_efficiency_test.py --teacher_ckpt ... --student_ckpt ... --cpu_threads 1
python run_efficiency_test.py --teacher_ckpt ... --student_ckpt ... --cpu_threads 2
```

Semua parameter simulasi wearable (jumlah thread, batch size, jumlah run
pengukuran latency, dsb) ada di `EfficiencyConfig` pada `config.py` kalau
ingin diubah permanen.

### Skenario #4 — Benchmark dengan Arsitektur Lain
```bash
python run_benchmark_models.py \
    --preset full_preprocessing \
    --teacher_ckpt checkpoints/teacher_full_best.pt \
    --student_ckpt checkpoints/student_full_best.pt \
    --train_baselines --wearable_sim
```
Ini melatih (jika checkpoint belum ada) dan menguji CNN-LSTM & ConvMixer
dengan dataset, preprocessing, split, DAN simulasi wearable yang **sama
persis** dengan Teacher/Student, lalu merangkum semua ke satu tabel:
`results/benchmark_summary.csv` — cocok langsung dipakai sebagai tabel
"Comparative Analysis with Other Models" di Bab 4 tesis.

Untuk menambah model pembanding lain, tambahkan kelasnya di
`src/models/baselines.py` lalu daftarkan di `build_baseline_model()`.

### Menguji ukuran file model (mis. target "< 200 KB" seperti di literatur)
Sudah otomatis muncul di semua laporan efisiensi (`model_size_kb`), lewat
`src/efficiency.py::get_model_size_kb()` — dihitung dari hasil nyata
`torch.save(model.state_dict())`, bukan estimasi teoritis.

---

## 4. Cara Mengubah Parameter (Skenario Pengujian #6)

Ada 2 cara:

**A. Lewat `config.py` langsung** (untuk perubahan default permanen) — semua
parameter ada di satu tempat, dikelompokkan per kategori (`PathConfig`,
`PreprocessConfig`, `SplitConfig`, `TransformerModelConfig`, `TrainConfig`,
`KDConfig`, `EfficiencyConfig`).

**B. Lewat argumen CLI** (untuk eksperimen sekali-jalan tanpa mengubah file):
```bash
python train_teacher.py --epochs 50 --batch_size 32 --lr 5e-4 --device cuda
python train_student.py --teacher_ckpt ... --alpha 0.7   # lebih berat ke soft-target
```
Kalau ingin menambah parameter CLI baru, tambahkan `p.add_argument(...)` di
fungsi `parse_args()` pada script terkait — semua script sudah mengikuti
pola yang sama.

---

## 5. Menjalankan di VS Code dengan GPU Lokal

1. Buka folder `rr_estimation_kd/` di VS Code (`File > Open Folder`).
2. Pastikan interpreter Python yang dipakai VS Code adalah yang sudah
   ter-install PyTorch versi GPU (`Ctrl+Shift+P` -> "Python: Select
   Interpreter" -> pilih venv kamu).
3. Buka Terminal terintegrasi VS Code (`` Ctrl+` ``), pastikan berada di
   folder root `rr_estimation_kd/` (bukan di dalam `src/`), lalu jalankan
   perintah `python nama_script.py ...` seperti contoh-contoh di atas.
4. **Penting:** semua import di kode ini pakai path absolut dari root
   proyek (`from src.dataset import ...`, `from config import ...`), jadi
   script HARUS dijalankan dari folder root, bukan dari dalam folder `src/`.
5. Untuk debugging step-by-step, buat `launch.json` di VS Code dan set
   `"program"` ke script yang ingin di-debug, `"args"` untuk argumen CLI-nya.

Kalau terjadi error terkait `num_workers` di DataLoader saat di Windows
(kadang muncul `RuntimeError` terkait multiprocessing), set
`cfg.train.num_workers = 0` di `config.py`, atau jalankan dengan pastikan
kode dipanggil di dalam `if __name__ == "__main__":` (semua script di sini
sudah begitu).

---

## 6. Tuning Lanjutan untuk Menekan MAE

Kalau tuning dasar (window overlap, alpha KD, cutoff filter, batch size)
mentok, coba opsi-opsi berikut secara berurutan (dampaknya biasanya lebih
besar dari tuning permukaan):

**A. Cek MAE Teacher terlebih dahulu, sebelum menyalahkan Student/KD**
```bash
python train_teacher.py --preset full_preprocessing --tag teacher_check
```
Kalau MAE Teacher masih jauh (>1.5-2 bpm), perbaiki Teacher dulu — Student
tidak mungkin jauh lebih akurat dari Teacher yang mengajarinya.

**B. Aktifkan CNN frontend** (fitur lokal sebelum self-attention — mendekati
pendekatan Zhao et al. dengan inception block, sering membantu di dataset kecil):
```bash
python train_teacher.py --preset full_preprocessing --conv_frontend --tag teacher_conv

python train_student.py --teacher_ckpt checkpoints/teacher_conv_best.pt \
    --teacher_conv_frontend --conv_frontend \
    --preset full_preprocessing --tag student_conv
```
(`--teacher_conv_frontend` WAJIB disertakan di `train_student.py` kalau
teacher-nya dilatih pakai `--conv_frontend`, supaya arsitektur saat memuat
checkpoint cocok.)

**C. Ganti LR scheduler jadi cosine + warmup** (praktik standar melatih
Transformer, lebih stabil dari `ReduceLROnPlateau` terutama di data terbatas):
```bash
python train_teacher.py --lr_scheduler cosine_warmup --warmup_epochs 5 --tag teacher_cosine
```

**D. Ganti loss ke Huber** (lebih tahan terhadap label RR yang terkuantisasi
kasar per window, dibanding MAE murni):
```bash
python train_teacher.py --loss_type huber --tag teacher_huber
```

**E. Kombinasikan semuanya sekaligus:**
```bash
python train_teacher.py --preset full_preprocessing --conv_frontend \
    --lr_scheduler cosine_warmup --warmup_epochs 5 --loss_type huber \
    --tag teacher_tuned

python train_student.py --teacher_ckpt checkpoints/teacher_tuned_best.pt \
    --teacher_conv_frontend --conv_frontend \
    --lr_scheduler cosine_warmup --warmup_epochs 5 --loss_type huber \
    --preset full_preprocessing --tag student_tuned
```

**F. Hal lain yang perlu kamu sadari (bukan bug, tapi batasan riil):**
- MAE 0,77 bpm milik Chowdhury dkk dicapai lewat **VMD (dekomposisi sinyal)
  sebelum ConvMixer** — bukan murni end-to-end seperti Transformer+KD kamu.
  Membandingkan MAE mentah keduanya secara langsung agak "tidak apple-to-apple";
  kontribusi utama tesismu (sesuai H2/H3) adalah **efisiensi komputasi pada
  akurasi yang kompetitif**, bukan mengalahkan akurasi absolut metode dengan
  preprocessing sinyal yang jauh lebih berat.
- Dataset BIDMC cuma 53 subjek — split tunggal 70/15/15 (test ±8 subjek) bisa
  memberi angka MAE yang cukup bervariasi tergantung subjek mana yang jatuh
  ke test set. Kalau mau angka yang lebih stabil untuk dibandingkan ke
  literatur, pertimbangkan k-fold cross-validation atau Leave-One-Subject-Out
  (rata-ratakan MAE dari beberapa split, bukan cuma 1 split acak).


## 7. Urutan Perintah Contoh (Skenario Lengkap dari Nol)

```bash
# 1. Siapkan environment & dataset (lihat Bagian 1 & data/README.md)

# 2. Ablation study preprocessing -> menentukan skema terbaik
python run_ablation_preprocessing.py --epochs 50

# 3. Training final dengan skema preprocessing terbaik (misal full_preprocessing)
python train_teacher.py --preset full_preprocessing --tag teacher_final
python train_student.py --teacher_ckpt checkpoints/teacher_final_best.pt \
                         --preset full_preprocessing --tag student_final

# 4. Bandingkan performa & efisiensi Teacher vs Student + simulasi wearable
python run_efficiency_test.py \
    --teacher_ckpt checkpoints/teacher_final_best.pt \
    --student_ckpt checkpoints/student_final_best.pt \
    --preset full_preprocessing

# 5. Benchmark melawan arsitektur lain
python run_benchmark_models.py \
    --preset full_preprocessing \
    --teacher_ckpt checkpoints/teacher_final_best.pt \
    --student_ckpt checkpoints/student_final_best.pt \
    --train_baselines --wearable_sim
```

Semua hasil akhirnya ada di folder `results/` sebagai file `.csv` yang
tinggal kamu masukkan ke tabel-tabel di Bab 4 tesis (dan log per-epoch
di `results/logs/` bisa dipakai untuk membuat grafik learning curve).
