# BANAAG: Defeating Noise. Delivering Clarity

**CPE027 — Digital Signal Processing Final Project**  
**Group 1 (Section: CPE41S2)**

### 👥 Group Members:
* **Adia, James Russel**
* **Magistrado, Aira Pauleen**
* **Penas, John Patrick**
* **San Juan, Edson Ray**
* **Santos, Ma. Kassandra Nicole**

---

## 📌 Project Summary & Objectives

This project implements **BANAAG**, an audio speech enhancement and automatic speech recognition (ASR) evaluation web backend designed for low-resource Philippine language audio (**Cebuano / Bisaya**).

Based on the research and experimentation in `documentation.ipynb`, the project satisfies the following core objectives:

1. **Develop an Audio Speech Enhancement Pipeline that:**
   * **a. Digital Signal Processing (DSP):** Cascades a **Butterworth Band-Pass Filter (BPF)**, **STFT Spectral Subtraction (SS)**, and an adaptive **Wiener Filter (WF)** to suppress environmental noises (train rumble, street traffic, and background chatter).
   * **b. Objective Evaluation:** Assesses audio quality through **Signal-to-Noise Ratio (SNR)** and speech intelligibility through **Word Error Rate (WER)**.
2. **Test and Evaluate Pipeline Accuracy:**
   * Uses fine-tuned **Whisper AI** models for Cebuano to transcribe both the corrupted raw audio and the enhanced audio side-by-side, quantitatively demonstrating accuracy and intelligibility gains ($\Delta\text{WER}$).

---

## 🛠️ What Was Built & Accomplished

### 1. DSP Signal Processing Engine (`dsp.py`)
* **Standardization:** Resamples any input audio to **16 kHz mono** and normalizes amplitude to prevent clipping.
* **Stage 1 — Band-Pass Filter (BPF):** 4th-order Butterworth filter ($150\text{--}6000\text{ Hz}$) executed with zero-phase forward-backward filtering (`sosfiltfilt`).
* **Stage 2 — Spectral Subtraction (SS):** STFT frame-energy percentile noise estimation, over-subtraction factor $\alpha = 1.0$, and spectral floor $\beta = 0.05$.
* **Stage 3 — Adaptive Wiener Filter (WF):** Decision-directed a priori / a posteriori SNR estimation with smoothing factor $0.98$ and minimum gain floor $0.05$.
* **Waveform Visualization:** Extracts 150 normalized peak values (`-1.0` to `1.0`) so the frontend can instantly draw raw and processed waveforms without heavy client-side audio decoding.
* **Metric Calculation:** Computes output SNR in dB and formats the human-readable ratio (e.g. `75:1`) matching the UI mockup.

### 2. Cebuano Whisper AI & WER Evaluation Engine (`asr.py`)
* **Dual-Model Smart Hardware Detection:**
  * **GPU Host (CUDA):** Automatically runs [`sapinsapin/whisper-large-v3-pld-ceb-norm`](https://huggingface.co/sapinsapin/whisper-large-v3-pld-ceb-norm) (1.5B parameters) in FP16 for top accuracy and sub-second inference.
  * **CPU Laptop (No GPU):** Automatically switches to [`sapinsapin/whisper-small-pld-ceb`](https://huggingface.co/sapinsapin/whisper-small-pld-ceb) (244M parameters) with restricted thread count to prevent laptop overheating and memory exhaustion.
* **Comparative Evaluation:** Transcribes both **Raw Audio** and **Enhanced Audio**, computing `baseline_wer_percent`, `enhanced_wer_percent`, and the net improvement (`wer_improvement_percentage_points`) via `jiwer`.
* **Timestamped Segments:** Splits transcripts into timestamped intervals (`00:00 - 00:06`) ready for the frontend transcript box.

### 3. RESTful API Server (`main.py`)
* Built with **FastAPI** and **Uvicorn** with cross-origin resource sharing (**CORS**) enabled for any frontend framework (React, Vue, Vite, Next.js, or plain HTML).
* In-memory session audio caching for direct playback streaming and file downloads.
* Auto-generated interactive Swagger UI available at `/docs`.

### 4. Sample Audio Dataset (`samples/`)
Extracted real Cebuano audio from the dataset (`SilencioNetwork/cebuano-speech`):
* `cebuano_noisy_short_5s.wav`: 5-second test sample for rapid API testing.
* `cebuano_speech_noisy_15dB.wav`: Full speech with simulated 15 dB environmental noise.
* `cebuano_speech_noisy_0dB.wav`: Full speech with heavy 0 dB noise (1:1 SNR).
* `cebuano_speech_clean.wav`: Reference clean recording.

---

## 📂 Project Structure

```text
CPE027_DSP_Final_Project_Web_App/
│
├── dsp.py                  # Core DSP algorithms (BPF, Spectral Subtraction, Wiener, Waveforms)
├── asr.py                  # Whisper AI engine (Large v3 on GPU / Small on CPU, WER evaluation)
├── main.py                 # FastAPI application & REST endpoints
├── requirements.txt        # Python library dependencies
├── documentation.ipynb     # Research, experimental trials, and baseline documentation
│
├── samples/                # Ready-to-test Cebuano speech files
│   ├── cebuano_noisy_short_5s.wav
│   ├── cebuano_speech_noisy_15dB.wav
│   ├── cebuano_speech_noisy_0dB.wav
│   └── cebuano_speech_clean.wav
│
└── README.md               # Project documentation & API guide
```

---

## 🚀 Quickstart Guide

### Prerequisites
* **Python Version:** **Python 3.10 – 3.13** *(Tested and verified on Python 3.13.7 / 3.13.9)*
* **OS:** Windows, Linux, or macOS
* **Hardware:** Runs on any CPU (Intel/AMD/Apple Silicon). NVIDIA GPU with CUDA recommended for accelerated Whisper Large v3 inference.

### 1. Setup Environment & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server
```bash
python main.py
```
*(or run `uvicorn main:app --reload`)*

The server will be accessible at:
* **Base URL:** `http://127.0.0.1:8000`
* **Swagger API Documentation:** `http://127.0.0.1:8000/docs`

---

## 📡 API Contract for Frontend Teammates

### `POST /api/process`
Main endpoint powering the entire **BANAAG** interface.

* **Content-Type:** `multipart/form-data`
* **Parameters:**
  * `file` *(UploadFile, required)*: The audio file (`.wav`, `.mp3`, etc.).
  * `run_whisper` *(boolean, default: `true`)*: Whether to run Whisper ASR and calculate WER.
  * `model_choice` *(string, default: `"auto"`)*: `"auto"`, `"large"`, or `"small"`.
  * `reference_transcript` *(string, optional)*: Ground truth text. Defaults to CEB_006 reference.
  * `lowcut`, `highcut`, `bpf_order`: Tunable BPF cutoff parameters.
  * `ss_alpha`, `ss_beta`: Tunable Spectral Subtraction parameters.
  * `wf_smoothing`, `wf_gain_floor`: Tunable Wiener filter parameters.

* **Response (JSON):**
```json
{
  "audio_id": "8a7c2b1e",
  "duration_seconds": 5.0,
  "sample_rate": 16000,
  "audio_urls": {
    "raw_audio": "/api/audio/8a7c2b1e/raw",
    "processed_audio": "/api/audio/8a7c2b1e/processed",
    "download_processed": "/api/audio/8a7c2b1e/processed?download=true",
    "download_transcript": "/api/transcript/8a7c2b1e/download"
  },
  "waveforms": {
    "raw": [-0.12, 0.45, -0.68, "...150 points for grey waveform..."],
    "processed": [-0.05, 0.32, -0.51, "...150 points for green waveform..."]
  },
  "results": {
    "signal_to_noise_ratio": "75:1",
    "snr_db": 18.75,
    "word_error_rate": "15%",
    "wer_raw": 15.0
  },
  "evaluations": {
    "snr": {
      "output_snr_db": 18.75,
      "ratio": "75:1"
    },
    "wer": {
      "baseline_wer_percent": 82.43,
      "enhanced_wer_percent": 15.00,
      "wer_improvement_percentage_points": 67.43,
      "raw_transcription": "...",
      "enhanced_transcription": "...",
      "ground_truth_reference": "..."
    }
  },
  "transcript": {
    "full_text": "...",
    "segments": [
      {
        "timestamp": "00:00 - 00:06",
        "text": "..."
      }
    ]
  }
}
```

---

### Additional Streaming & Utility Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `GET /api/audio/{id}/raw` | `GET` | Stream raw WAV audio for the **Raw Audio Input** player |
| `GET /api/audio/{id}/processed` | `GET` | Stream enhanced WAV audio for the **Processed Audio Output** player |
| `GET /api/audio/{id}/processed?download=true` | `GET` | Download processed WAV file (**"Download Processed Audio"** button) |
| `GET /api/transcript/{id}/download` | `GET` | Download transcript as `.txt` (**"Download Transcript"** button) |
| `GET /api/asr/status` | `GET` | Check host hardware (GPU vs. CPU) and active Whisper model |
| `POST /api/asr/preload` | `POST` | Preload Whisper model into memory ahead of time |
