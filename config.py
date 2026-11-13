import os
import torch

"""Cara running program
Preprocessing dataset: python -m src.preprocessing
Training teacher: python training_teacher.py
Training student KD: python training_student.py --mode KD
Training student SCRATCH: python training_student.py --mode SCRATCH
Training model benchmark all: python run_benchmark_models.py --model all
Training model benchmark 1 model: python run_benchmark_models.py --model (model name)
Model name: TEACHER, STUDENT_KD, STUDENT_SCRATCH, TRANSRR, HYBRID_CNN_LSTM, CONVMIXER, RESPTRACK_1DCNN
Testing + tabel: python evaluate.py
Munculkan prediction plot: python plot_predictions.py --target (model name))


"""

class SplitConfig:
    random_seed = 42           #awalnya 42, 23, 
    n_train_subjects = 36
    n_val_subjects = 8
    n_test_subjects = 8

class DataConfig:
    sampling_rate = 125
    downsample_factor = 5        # 125 Hz -> 25 Hz
    fs_target = 25.0
    window_seconds = 30.0
    window_step_seconds = 2.0
    samples_per_window = int(window_seconds * fs_target) # 750 titik
    
    # Bandpass filter & Quality filtering
    use_bandpass_filter = True
    bandpass_low_hz = 0.083       #awalnya 0,08
    bandpass_high_hz = 0.57      #awalnya o,8 note: awalnya pakai ini karena menyesuaikan dengan rentang RR pada capnobase agar test robust bagus
    bandpass_order = 2          #awalnya 2
    use_zscore_normalization = True
    use_quality_filtering = True
    annotator_agreement_bpm = 3.0
    flat_signal_std_threshold = 1e-4
    rr_min_bpm = 5              #the best pakai rentang RR 4.8-48 bpm. RR normal 6-36 bpm
    rr_max_bpm = 34

class TeacherConfig:
    d_model = 128
    n_heads = 8
    num_layers = 4
    dim_feedforward = 256
    dropout = 0.1
    patch_size = 25

class StudentConfig:
    d_model = 32
    n_heads = 2
    num_layers = 2
    dim_feedforward = 64
    dropout = 0.1
    patch_size = 25

class TrainConfig:
    batch_size = 64
    epochs = 100
    learning_rate = lr = 1e-3       #1e-3
    weight_decay = 1e-4
    patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0

class KDConfig:
    alpha: float = 0.5
    hard_loss_type: str = "mae"
    huber_delta: float = 1.0
    teacher_checkpoint: str = "checkpoints/teacher_best.pth"

class Config:
    def __init__(self):
        self.split = SplitConfig()
        self.data = DataConfig()
        self.teacher = TeacherConfig()
        self.teacher_model = self.teacher
        self.student = StudentConfig()
        self.student_model = self.student
        self.train = TrainConfig()
        self.kd = KDConfig()

        # --- TOGGLES SKENARIO PENGUJIAN ---
        self.SIMULATE_WEARABLE = True
        self.TEST_RR_STRATIFICATION = True
        self.STUDENT_MODE = "KD"
        self.TEST_EXTERNAL_CAPNOBASE = True
        self.ACTIVE_DATASET = "BIDMC"
        self.USE_VMD = False  # Tambahan toggle VMD opsional

        # --- PATHS ---
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.RAW_BIDMC_DIR = os.path.join(self.DATA_DIR, "raw", "bidmc_csv")
        self.RAW_CAPNOBASE_DIR = os.path.join(self.DATA_DIR, "raw", "capnobase")
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, "processed")
        self.BIDMC_PROCESSED_PATH = os.path.join(self.PROCESSED_DATA_DIR, "bidmc_processed.pt")
        self.CAPNOBASE_PROCESSED_PATH = os.path.join(self.PROCESSED_DATA_DIR, "capnobase_processed.pt")
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "checkpoints")
        self.RESULTS_DIR = os.path.join(self.ROOT_DIR, "results")

        # Baseline Models
        self.ACTIVE_BENCHMARK_MODELS = ["TransRR", "Hybrid_CNN_LSTM", "ConvMixer", "Resptrack_1DCNN"]

def get_config(*args, **kwargs):
    return Config()

PreprocessConfig = Config
DataConfigClass = DataConfig
TrainConfigClass = TrainConfig
KDConfigClass = KDConfig