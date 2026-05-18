# 🎙️ VoiceIQ — Speech & NLP Intelligence Platform

> **Record → Denoise → Extract → Transcribe → Analyze**
> An end-to-end pipeline that takes raw Arabic audio and delivers English translation + sentiment analysis — all through a clean Streamlit interface.

---

## 🧠 What Does It Do?

VoiceIQ is a full-stack AI pipeline with 4 stages:

| Stage | What Happens |
|-------|-------------|
| 🎤 **Audio Acquisition** | Record mic input or upload a `.wav` file, then visualize the raw waveform & frequency spectrum |
| 🔇 **DSP & Noise Reduction** | Normalize → Pre-emphasis → VAD (silence removal) → Noise reduction → Framing → Hamming window |
| 📊 **Feature Extraction** | Extract MFCC, Delta-MFCC, Mel Spectrogram, Pitch, Spectral Centroid & Bandwidth |
| 🤖 **AI Translation + Sentiment** | Faster-Whisper transcribes/translates Arabic → English, then DistilBERT + Naive Bayes + SVM classify the sentiment |

---

## 🖥️ App Structure

```
app.py               ← Streamlit GUI (4 tabs)
audio_pipeline.py    ← Audio processing & Whisper functions
nlp_pipeline.py      ← NLP preprocessing & ML models
```

### Tabs inside the app:
- **Tab 1 — Main Pipeline**: Record or upload → runs the full pipeline automatically
- **Tab 2 — DSP Details**: Cleaned waveform, spectrogram, step-by-step DSP log
- **Tab 3 — Feature Extraction**: MFCC, Mel Spectrogram, Scaled Heatmap
- **Tab 4 — NLP / Sentiment**: BERT + NB + SVM predictions, confidence scores, token breakdown, JSON export

---

## ⚙️ Models Used

| Model | Purpose |
|-------|---------|
| `faster-whisper` (small) | Arabic ASR + English translation |
| `distilbert-base-uncased` (fine-tuned) | Sentiment classification |
| `Naive Bayes` + `TF-IDF` | Baseline sentiment classifier |
| `SVM` + `TF-IDF` | Baseline sentiment classifier |

> ⚠️ The fine-tuned BERT model (`my_best_model/`) is **not included** in this repo due to size. Train it using `nlp_pipeline.py` or load your own checkpoint.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/voiceiq.git
cd voiceiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
sounddevice
scipy
librosa
matplotlib
numpy
soundfile
noisereduce
scikit-learn
faster-whisper
transformers
torch
datasets
textblob
nltk
pandas
seaborn
emoji
```

---

## 📁 Files NOT Included (add to `.gitignore`)

```
*.wav                  # recorded/cleaned audio files
translation_output.json
my_best_model/         # fine-tuned BERT weights
nb_model.pkl
svm_model.pkl
tfidf.pkl
__pycache__/
.env
venv/
```

---

## 📤 Output

After running the pipeline, the app generates:
- 🔊 Cleaned audio preview
- 📊 Waveform, Spectrogram, MFCC, Mel plots
- 📝 English translation with confidence score & inference time
- 💬 Sentiment label (Positive / Negative) from 3 models
- 💾 Downloadable JSON report (`voiceiq_report.json`)

---

## 🔧 Pipeline Config

| Parameter | Default |
|-----------|---------|
| Sample Rate | 16,000 Hz |
| Recording Duration | 3 seconds |
| VAD Threshold | 35 dB |
| MFCC Coefficients | 13 |
| Mel Bands | 128 |
| Whisper Model | `small` |
| Whisper Beam Size | 5 |

---

## 👤 Author

Built as a graduation / portfolio project combining DSP, ASR, and NLP into one unified Streamlit app.
