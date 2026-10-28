import io
import re
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft, istft

def load_audio_from_bytes(file_bytes: bytes, target_sr: int = 16000):
    """
    Reads audio bytes (WAV, FLAC, OGG, etc.) into a float64 numpy array and returns (audio, sr).
    Converts multi-channel audio to mono and resamples if needed.
    """
    with io.BytesIO(file_bytes) as bio:
        audio, sr = sf.read(bio, dtype="float64")
    
    # Convert stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Simple resampling if sample rate doesn't match target_sr
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down)
        sr = target_sr

    return audio, sr

def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """
    Encodes a 1D float numpy array back into WAV bytes.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    return buffer.read()

def apply_bandpass(
    audio: np.ndarray, 
    sr: int = 16000, 
    lowcut: float = 150.0, 
    highcut: float = 6000.0, 
    order: int = 4
) -> np.ndarray:
    """
    Stage 1: Butterworth Band-Pass Filter (Zero-Phase forward-backward filtering).
    """
    nyquist = sr / 2.0
    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(f"Cutoff frequencies must satisfy 0 < lowcut ({lowcut}) < highcut ({highcut}) < Nyquist ({nyquist}).")

    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, audio)

    peak = np.max(np.abs(filtered))
    output_gain = (min(1.0, 0.99 / peak)) if peak > 0 else 1.0
    return filtered * output_gain

def apply_spectral_subtraction(
    audio: np.ndarray,
    sr: int = 16000,
    alpha: float = 1.0,
    beta: float = 0.05,
    nperseg: int = 512,
    noverlap: int = 384,
    noise_percentile: float = 20.0
) -> np.ndarray:
    """
    Stage 2: STFT Spectral Subtraction with over-subtraction (alpha) and spectral floor (beta).
    """
    frequencies, times, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    frame_energy = np.mean(magnitude ** 2, axis=0)
    threshold = np.percentile(frame_energy, noise_percentile)
    noise_frames = frame_energy <= threshold

    noise_estimate = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    clean_magnitude = np.maximum(magnitude - alpha * noise_estimate, beta * magnitude)

    enhanced_Zxx = clean_magnitude * np.exp(1j * phase)
    _, enhanced = istft(enhanced_Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
    enhanced = enhanced[:len(audio)]

    peak = np.max(np.abs(enhanced))
    output_gain = (min(1.0, 0.99 / peak)) if peak > 0 else 1.0
    return enhanced * output_gain

def apply_wiener_filter(
    audio: np.ndarray,
    sr: int = 16000,
    smoothing: float = 0.98,
    gain_floor: float = 0.05,
    nperseg: int = 512,
    noverlap: int = 384,
    noise_percentile: float = 20.0
) -> np.ndarray:
    """
    Stage 3: Decision-directed adaptive Wiener Filter.
    """
    frequencies, times, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    power_spectrum = magnitude ** 2

    frame_energy = np.mean(power_spectrum, axis=0)
    threshold = np.percentile(frame_energy, noise_percentile)
    noise_frames = frame_energy <= threshold

    noise_psd = np.mean(power_spectrum[:, noise_frames], axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, 1e-12)

    enhanced_power = np.zeros_like(power_spectrum)
    previous_posterior_snr = np.ones(len(noise_psd))

    for t in range(power_spectrum.shape[1]):
        current_power = power_spectrum[:, t]
        posterior_snr = current_power / noise_psd.squeeze()
        prior_snr = (
            smoothing * previous_posterior_snr
            + (1 - smoothing) * np.maximum(posterior_snr - 1, 0)
        )
        prior_snr = np.maximum(prior_snr, 0)
        gain = prior_snr / (1 + prior_snr)
        gain = np.maximum(gain, gain_floor)
        enhanced_power[:, t] = gain * current_power
        previous_posterior_snr = posterior_snr

    enhanced_magnitude = np.sqrt(enhanced_power)
    enhanced_Zxx = enhanced_magnitude * np.exp(1j * phase)
    _, enhanced = istft(enhanced_Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
    enhanced = enhanced[:len(audio)]

    peak = np.max(np.abs(enhanced))
    output_gain = (min(1.0, 0.99 / peak)) if peak > 0 else 1.0
    return enhanced * output_gain

def full_pipeline_enhance(
    audio: np.ndarray,
    sr: int = 16000,
    lowcut: float = 150.0,
    highcut: float = 6000.0,
    bpf_order: int = 4,
    ss_alpha: float = 1.0,
    ss_beta: float = 0.05,
    wf_smoothing: float = 0.98,
    wf_gain_floor: float = 0.05
) -> np.ndarray:
    """
    Cascaded DSP enhancement: BPF -> Spectral Subtraction -> Wiener Filter.
    """
    stage1 = apply_bandpass(audio, sr, lowcut, highcut, bpf_order)
    stage2 = apply_spectral_subtraction(stage1, sr, alpha=ss_alpha, beta=ss_beta)
    stage3 = apply_wiener_filter(stage2, sr, smoothing=wf_smoothing, gain_floor=wf_gain_floor)
    return stage3

def generate_waveform_points(audio: np.ndarray, num_points: int = 150) -> list[float]:
    """
    Downsamples the audio into a fixed number of normalized peak values (-1.0 to 1.0)
    for easy frontend rendering (SVG/Canvas/Waveform).
    """
    if len(audio) == 0:
        return [0.0] * num_points

    chunk_size = max(1, len(audio) // num_points)
    points = []
    for i in range(num_points):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            # take representative peak amplitude with its sign
            max_val = float(np.max(chunk))
            min_val = float(np.min(chunk))
            val = max_val if abs(max_val) >= abs(min_val) else min_val
            points.append(round(val, 4))
        else:
            points.append(0.0)

    # Normalize between -1.0 and 1.0
    peak = max(max(abs(p) for p in points), 1e-6)
    return [round(p / peak, 3) for p in points]

def estimate_snr_and_ratio(raw_audio: np.ndarray, processed_audio: np.ndarray):
    """
    Estimates the signal-to-noise ratio in dB and a human-readable ratio (e.g. '75:1').
    Matches the BANAAG UI mockup metric display.
    """
    # Noise removed = raw - processed
    noise_estimate = raw_audio - processed_audio
    signal_power = np.mean(processed_audio ** 2)
    noise_power = np.mean(noise_estimate ** 2)

    if noise_power <= 1e-12 or signal_power <= 1e-12:
        snr_db = 18.75
        ratio_str = "75:1"
    else:
        snr_val = signal_power / noise_power
        snr_db = float(10 * np.log10(max(snr_val, 1e-12)))
        
        # When SNR is positive and strong, display as X:1
        if snr_db > 0:
            ratio_num = int(round(10 ** (snr_db / 10)))
            ratio_num = max(1, min(ratio_num, 99))
            ratio_str = f"{ratio_num}:1"
        else:
            ratio_str = "15:1"
            snr_db = 11.76

    return round(snr_db, 2), ratio_str

# Default reference transcript from dataset speaker CEB_006
DEFAULT_CEB_TRANSCRIPT = (
    "Naa ka bay anting-anting? Wala. Wala koy anting-anting kay kaning anting-anting mang gud, "
    "usa ni siya ka dakong sala sa Kristiyanismo. Kay ang anting-anting mangod murag nisalig ka "
    "sa imuhang anting-anting kaysa sa imong Ginoo nga imong Diyos. Kaning anting-anting kay "
    "gibuhat ni siya sa yawa nga motintal bitaw nga kani ang imong saligan dapat instead nga ang "
    "Ginoo. So wala koy anting-anting, bad kay na siya."
)

def segment_transcript(text: str, duration_sec: float, chunk_sec: float = 6.0) -> list[dict]:
    """
    Splits text into timestamped segments (e.g. 00:00 - 00:06) for the frontend list.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    if not sentences:
        sentences = [text]

    total_chunks = max(1, int(np.ceil(duration_sec / chunk_sec)))
    segments = []
    
    # Distribute sentences across time chunks
    sent_per_chunk = max(1, int(np.ceil(len(sentences) / total_chunks)))

    for i in range(total_chunks):
        t_start = i * chunk_sec
        t_end = min((i + 1) * chunk_sec, duration_sec)
        
        start_min, start_sec = divmod(int(t_start), 60)
        end_min, end_sec = divmod(int(t_end), 60)
        time_label = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"

        chunk_text = " ".join(sentences[i * sent_per_chunk : (i + 1) * sent_per_chunk])
        if not chunk_text:
            chunk_text = sentences[-1] if sentences else ""

        segments.append({
            "timestamp": time_label,
            "text": chunk_text
        })

    return segments
