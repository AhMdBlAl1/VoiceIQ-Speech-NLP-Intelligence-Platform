"""
Audio Processing Pipeline - GUI-Ready Functions
================================================
Step 1: Audio Acquisition & Analysis
Step 2: DSP & Noise Reduction
Step 3: Acoustic Feature Extraction
Step 4: AI & Translation
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import lfilter
from sklearn.preprocessing import StandardScaler
from faster_whisper import WhisperModel
import time
import json


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Audio Acquisition & Analysis
# ══════════════════════════════════════════════════════════════════

def record_audio(duration=3, sample_rate=16000, channels=1,
                 output_file="recorded_audio.wav"):
    print("Recording Started...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='float32'
    )
    sd.wait()
    print("Recording Finished!")

    audio = audio.flatten()
    write(output_file, sample_rate, audio)

    return audio, sample_rate


def load_audio(file_path="recorded_audio.wav", sample_rate=16000):
    signal, sr = librosa.load(file_path, sr=sample_rate)
    print("Signal Shape:", signal.shape)
    print("Sample Rate:", sr)
    return signal, sr


def plot_waveform(signal, sr):
   
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig


def plot_fft(signal, sr):
    
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frequency[:len(frequency) // 2],
            magnitude[:len(magnitude) // 2])
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    plt.tight_layout()

    return fig, frequency, magnitude


def get_zero_crossing_rate(signal):
  
    zcr = librosa.feature.zero_crossing_rate(signal)
    zcr_mean = np.mean(zcr)
    print("Zero Crossing Rate:", zcr_mean)
    return zcr_mean


def get_rms_energy(signal):
   
    rms = librosa.feature.rms(y=signal)
    rms_mean = np.mean(rms)
    print("RMS Energy:", rms_mean)
    return rms_mean


def get_signal_power(signal):
    
    power = np.mean(signal ** 2)
    print("Signal Power:", power)
    return power


# ══════════════════════════════════════════════════════════════════
# STEP 2 — DSP & Noise Reduction
# ══════════════════════════════════════════════════════════════════

def normalize_signal(signal):
    normalized = signal / np.max(np.abs(signal))
    print("Normalization Done")
    return normalized


def plot_normalization(signal, normalized_signal):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, label="Original")
    ax.plot(normalized_signal, label="Normalized", alpha=0.7)
    ax.set_title("Signal Normalization")
    ax.legend()
    plt.tight_layout()
    return fig


def apply_pre_emphasis(signal, pre_emphasis=0.97):
    
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1]
    )
    print("Pre-emphasis Applied")
    return emphasized_signal


def apply_vad(signal, top_db=35, hop_length=512):
    
    intervals = librosa.effects.split(
        signal,
        top_db=top_db,
        hop_length=hop_length
    )
    speech_signal = np.concatenate(
        [signal[start:end] for start, end in intervals]
    )
    print("Silence Removed")
    return speech_signal


def reduce_noise(speech_signal, sr):
    
    reduced = nr.reduce_noise(y=speech_signal, sr=sr)
    print("Noise Reduction Done")
    return reduced


def save_cleaned_audio(speech_signal, sr,
                       output_path="cleaned_voice.wav"):
    
    sf.write(output_path, speech_signal, sr)
    print(f"Cleaned audio saved as '{output_path}'")


def apply_framing(speech_signal, sr,
                  frame_size=0.025, frame_stride=0.01):
    
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)
    signal_length = len(speech_signal)

    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step
    ))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(speech_signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )

    frames = pad_signal[indices.astype(np.int32, copy=False)]
    print("Framing Done")
    print("Number of Frames:", len(frames))
    return frames


def estimate_noise_profile(frames):
    noise_profile = np.mean(np.abs(frames), axis=0)
    print("Noise Profile Estimated")
    return noise_profile


def apply_hamming_window(frames):
    
    frame_length = frames.shape[1]
    frames *= np.hamming(frame_length)
    print("Hamming Window Applied")
    return frames


def plot_spectrogram(speech_signal, sr):
    
    fig, ax = plt.subplots(figsize=(12, 5))
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(speech_signal)),
        ref=np.max
    )
    img = librosa.display.specshow(D, sr=sr,
                                   x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Spectrogram")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Acoustic Feature Extraction
# ══════════════════════════════════════════════════════════════════

def extract_mfcc(signal, sr, n_mfcc=13):
 
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    print("MFCC Shape:", mfccs.shape)
    return mfccs


def plot_mfcc(mfccs):
  
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC Features")
    plt.tight_layout()
    return fig


def extract_delta_mfcc(mfccs):
  
    delta_mfcc = librosa.feature.delta(mfccs)
    print("Delta MFCC Shape:", delta_mfcc.shape)
    return delta_mfcc


def extract_delta2_mfcc(mfccs):
 
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    print("Delta-Delta Shape:", delta2_mfcc.shape)
    return delta2_mfcc


def extract_mel_spectrogram(signal, sr, n_mels=128):
    
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    print("Mel Spectrogram Created")
    return mel_spec_db


def plot_mel_spectrogram(mel_spec_db, sr):
    
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(
        mel_spec_db, sr=sr,
        x_axis='time', y_axis='mel', ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


def extract_pitch(signal, sr):
    
    pitch, magnitude = librosa.piptrack(y=signal, sr=sr)
    print("Pitch Extracted")
    return pitch, magnitude


def extract_spectral_centroid(signal, sr):
    
    spectral_centroids = librosa.feature.spectral_centroid(
        y=signal, sr=sr
    )
    print("Spectral Centroid Shape:", spectral_centroids.shape)
    return spectral_centroids


def extract_spectral_bandwidth(signal, sr):
   
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr
    )
    print("Spectral Bandwidth Shape:", spectral_bandwidth.shape)
    return spectral_bandwidth


def scale_features(mfccs):
    
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    print("Feature Scaling Done")
    return mfccs_scaled, scaler


def plot_mfcc_heatmap(mfccs_scaled):
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mfccs_scaled.T, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title("MFCC Heatmap")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Coefficients")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# STEP 4 — AI & Translation
# ══════════════════════════════════════════════════════════════════

def load_whisper_model(model_size="small", device="cpu",
                       compute_type="int8"):

    model = WhisperModel(model_size, device=device,
                         compute_type=compute_type)
    print("Model Loaded Successfully")
    return model



def transcribe_and_translate(model, speech_signal,
                              language="ar",
                              initial_prompt="كلمات مصرية عامية",
                              beam_size=5):
    start_time = time.time()

    segments, info = model.transcribe(
        speech_signal,
        task="translate",
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        vad_filter=True
    )

    segments = list(segments)
    translated_text = " ".join([seg.text for seg in segments])

    end_time = time.time()
    inference_time = round(end_time - start_time, 2)

    print("Detected Language:", info.language)
    print("Inference Time:", inference_time, "seconds")

    return translated_text, info, segments, inference_time



def get_confidence_score(segments):

    probs = [np.exp(s.avg_logprob) for s in segments]
    if probs:
        confidence = (sum(probs) / len(probs)) * 100
    else:
        confidence = 0.0

    print(f"Confidence: {round(confidence, 2)}%")
    return round(confidence, 2)


def save_translation_json(translated_text, info,
                           inference_time, confidence,
                           model_used="small",
                           output_path="translation_output.json"):
   
    output_data = {
        "translated_text": translated_text.strip(),
        "language": info.language,
        "inference_time_seconds": inference_time,
        "confidence_score": confidence,
        "model_used": model_used
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"JSON File Saved Successfully → {output_path}")
    return output_data


# ══════════════════════════════════════════════════════════════════
# Full Pipeline (helper للـ GUI)
# ══════════════════════════════════════════════════════════════════

def run_full_pipeline(duration=3, sample_rate=16000,
                      audio_file="recorded_audio.wav",
                      cleaned_file="cleaned_voice.wav",
                      json_output="translation_output.json"):
  
    result = {}

    # ── Step 1: Acquisition ──────────────────
    audio, sr = record_audio(duration, sample_rate,
                              output_file=audio_file)
    signal, sr = load_audio(audio_file, sample_rate)

    result["zcr"]          = get_zero_crossing_rate(signal)
    result["rms"]          = get_rms_energy(signal)
    result["signal_power"] = get_signal_power(signal)

    # ── Step 2: DSP ──────────────────────────
    normalized   = normalize_signal(signal)
    emphasized   = apply_pre_emphasis(normalized)
    speech       = apply_vad(emphasized)
    speech       = reduce_noise(speech, sr)
    save_cleaned_audio(speech, sr, cleaned_file)

    frames       = apply_framing(speech, sr)
    noise_prof   = estimate_noise_profile(frames)
    frames       = apply_hamming_window(frames)

    result["speech_signal"] = speech

    # ── Step 3: Features ─────────────────────
    mfccs              = extract_mfcc(speech, sr)
    delta              = extract_delta_mfcc(mfccs)
    delta2             = extract_delta2_mfcc(mfccs)
    mel_db             = extract_mel_spectrogram(speech, sr)
    pitch, pitch_mag   = extract_pitch(speech, sr)
    centroid           = extract_spectral_centroid(speech, sr)
    bandwidth          = extract_spectral_bandwidth(speech, sr)
    mfccs_scaled, _    = scale_features(mfccs)

    result["mfccs"]         = mfccs
    result["mel_spec_db"]   = mel_db
    result["mfccs_scaled"]  = mfccs_scaled

    # ── Step 4: AI Translation ───────────────
    model          = load_whisper_model()
    trans, info, segs, inf_time = transcribe_and_translate(
        model, speech
    )
    confidence     = get_confidence_score(segs)
    output_data    = save_translation_json(
        trans, info, inf_time, confidence,
        output_path=json_output
    )

    result["translation"] = output_data
    print("\n✅ Pipeline Completed Successfully!")
    print("Translation:", trans.strip())

    return result
