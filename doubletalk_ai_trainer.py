import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Ensure openpyxl is available for Excel support
try:
    import openpyxl
except ImportError:
    print("Warning: openpyxl not found. Excel support may be limited.")
    print("Please install with: pip install openpyxl")

class AudioFeatureExtractor:
    """Extract audio features for doubletalk assessment"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_segment_features(self, audio: np.ndarray, start_time: float, end_time: float) -> Dict:
        """Extract features from a specific time segment"""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Ensure we don't go beyond audio length
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if start_sample >= end_sample:
            # Return zero features if segment is invalid
            return self._get_zero_features()
        
        segment = audio[start_sample:end_sample]
        
        # Skip if segment is too short
        if len(segment) < self.sample_rate * 0.1:  # At least 0.1 seconds
            return self._get_zero_features()
        
        features = {}
        
        try:
            # Time domain features
            features['rms_energy'] = float(np.sqrt(np.mean(segment**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(segment)))
            
            # Frequency domain features - with error handling
            try:
                features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(segment, sr=self.sample_rate)))
            except:
                features['spectral_centroid'] = 0.0
            
            try:
                features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(segment, sr=self.sample_rate)))
            except:
                features['spectral_bandwidth'] = 0.0
            
            try:
                features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(segment, sr=self.sample_rate)))
            except:
                features['spectral_rolloff'] = 0.0
            
            # MFCC features - with error handling
            try:
                mfccs = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            except:
                # If MFCC extraction fails, use zero features
                for i in range(13):
                    features[f'mfcc_{i}'] = 0.0
                    features[f'mfcc_{i}_std'] = 0.0
            
            # Chroma features - with error handling
            try:
                chroma = librosa.feature.chroma_stft(y=segment, sr=self.sample_rate)
                features['chroma_mean'] = float(np.mean(chroma))
                features['chroma_std'] = float(np.std(chroma))
            except:
                features['chroma_mean'] = 0.0
                features['chroma_std'] = 0.0
            
            # Temporal features for "five" detection
            features['segment_duration'] = float(len(segment) / self.sample_rate)
            features['energy_variance'] = float(np.var(segment**2))
            
            # Additional robust features
            features['peak_amplitude'] = float(np.max(np.abs(segment)))
            features['signal_to_noise_ratio'] = float(self._calculate_snr(segment))
            
        except Exception as e:
            print(f"Warning: Error extracting features from segment: {e}")
            return self._get_zero_features()
        
        # Clean up any NaN or infinite values
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
    def _get_zero_features(self) -> Dict:
        """Return a dictionary of zero features with consistent structure"""
        features = {
            'rms_energy': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'chroma_mean': 0.0,
            'chroma_std': 0.0,
            'segment_duration': 0.0,
            'energy_variance': 0.0,
            'peak_amplitude': 0.0,
            'signal_to_noise_ratio': 0.0
        }
        
        # Add MFCC features
        for i in range(13):
            features[f'mfcc_{i}'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
        
        return features
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            # Simple SNR calculation
            signal_power = np.mean(signal**2)
            if signal_power == 0:
                return 0.0
            
            # Estimate noise as the minimum energy in short windows
            window_size = len(signal) // 10
            if window_size < 100:
                return 20 * np.log10(signal_power + 1e-10)
            
            windowed_energy = []
            for i in range(0, len(signal) - window_size, window_size):
                window = signal[i:i+window_size]
                windowed_energy.append(np.mean(window**2))
            
            noise_power = np.percentile(windowed_energy, 10)  # Bottom 10% as noise estimate
            
            if noise_power <= 0:
                return 20 * np.log10(signal_power + 1e-10)
            
            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr)
            
        except:
            return 0.0
    
    def extract_doubletalk_features(self, audio: np.ndarray, delay_compensation: float = 0.0) -> Dict:
        """Extract features specifically for doubletalk analysis"""
        features = {}
        
        # Apply delay compensation (positive delay means recorded audio is delayed)
        delay_samples = int(delay_compensation * self.sample_rate)
        
        # Adjust timing based on delay compensation
        # Source: doubletalk 35.5s + singletalk 7s = 42.5s total, recorded ~50s
        doubletalk_start = delay_samples
        doubletalk_end = delay_samples + int(35.5 * self.sample_rate)  # 35.5s of doubletalk
        singletalk_start = doubletalk_end
        singletalk_end = singletalk_start + int(7 * self.sample_rate)  # 7s of singletalk
        
        # Ensure we don't exceed audio bounds
        doubletalk_start = max(0, doubletalk_start)
        doubletalk_end = min(len(audio), doubletalk_end)
        singletalk_start = max(0, singletalk_start)
        singletalk_end = min(len(audio), singletalk_end)
        
        doubletalk_segment = audio[doubletalk_start:doubletalk_end]
        singletalk_segment = audio[singletalk_start:singletalk_end]
        
        # Extract features from both segments
        dt_features = self.extract_segment_features(audio, 0, 35.5)
        st_features = self.extract_segment_features(audio, 35.5, 42.5)
        
        # Add prefix to distinguish segment types
        for key, value in dt_features.items():
            features[f'doubletalk_{key}'] = value
            
        for key, value in st_features.items():
            features[f'singletalk_{key}'] = value
        
        # Comparative features
        features['energy_ratio_dt_st'] = dt_features['rms_energy'] / (st_features['rms_energy'] + 1e-8)
        features['spectral_centroid_diff'] = dt_features['spectral_centroid'] - st_features['spectral_centroid']
        
        # Analyze "five" patterns in both doubletalk and singletalk segments
        # Each "five" is approximately 1 second
        dt_five_features = self.analyze_five_patterns(doubletalk_segment, segment_type="doubletalk")
        st_five_features = self.analyze_five_patterns(singletalk_segment, segment_type="singletalk")
        
        # Add doubletalk "five" features
        for key, value in dt_five_features.items():
            features[f'dt_{key}'] = value
        
        # Add singletalk "five" features
        for key, value in st_five_features.items():
            features[f'st_{key}'] = value
        
        # Combined "five" analysis across both segments
        total_five_features = self.analyze_total_five_patterns(doubletalk_segment, singletalk_segment)
        features.update(total_five_features)
        
        return features
    
    def analyze_five_patterns(self, audio_segment: np.ndarray, segment_type: str = "doubletalk") -> Dict:
        """Analyze patterns of "five" utterances in a specific segment"""
        features = {}
        
        if len(audio_segment) == 0:
            return self._get_zero_five_features()
        
        # Calculate segment duration and expected "five" count
        segment_duration = len(audio_segment) / self.sample_rate
        estimated_fives = int(segment_duration)  # ~1 second per "five"
        
        if estimated_fives == 0:
            return self._get_zero_five_features()
        
        segment_length = len(audio_segment) // estimated_fives
        num_segments = min(estimated_fives, len(audio_segment) // segment_length)
        
        energies = []
        spectral_centroids = []
        
        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            
            if end > len(audio_segment):
                break
                
            segment = audio_segment[start:end]
            
            if len(segment) == 0:
                continue
            
            # Energy analysis
            energy = np.sqrt(np.mean(segment**2))
            energies.append(energy)
            
            # Spectral features
            try:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(segment, sr=self.sample_rate))
                spectral_centroids.append(spectral_centroid)
            except:
                spectral_centroids.append(0.0)
        
        # Energy progression features
        if len(energies) > 1:
            try:
                features['energy_trend'] = np.polyfit(range(len(energies)), energies, 1)[0]
                features['energy_correlation'] = np.corrcoef(range(len(energies)), energies)[0, 1]
                
                # For doubletalk, analyze repetition consistency if we have enough segments
                if segment_type == "doubletalk" and len(energies) >= 20:
                    mid_point = len(energies) // 2
                    first_half = energies[:mid_point]
                    second_half = energies[mid_point:mid_point + len(first_half)]
                    if len(first_half) == len(second_half):
                        features['repetition_consistency'] = np.corrcoef(first_half, second_half)[0, 1]
                    else:
                        features['repetition_consistency'] = 0.0
                else:
                    features['repetition_consistency'] = 0.0
                    
            except:
                features['energy_trend'] = 0.0
                features['energy_correlation'] = 0.0
                features['repetition_consistency'] = 0.0
        else:
            features['energy_trend'] = 0.0
            features['energy_correlation'] = 0.0
            features['repetition_consistency'] = 0.0
        
        # Statistical features of energy distribution
        features['energy_mean'] = np.mean(energies) if energies else 0.0
        features['energy_std'] = np.std(energies) if energies else 0.0
        features['energy_range'] = (np.max(energies) - np.min(energies)) if energies else 0.0
        
        # Spectral consistency features
        if spectral_centroids:
            features['spectral_consistency'] = 1 - np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
        else:
            features['spectral_consistency'] = 0.0
        
        # Pattern regularity features
        features['num_detected_segments'] = len(energies)
        features['expected_segments'] = estimated_fives
        features['detection_ratio'] = len(energies) / estimated_fives if estimated_fives > 0 else 0.0
        
        # Segment-specific features
        features['segment_duration'] = segment_duration
        features['segment_type'] = 1.0 if segment_type == "doubletalk" else 0.0
        
        # Clean up any NaN or infinite values
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
    def analyze_total_five_patterns(self, doubletalk_segment: np.ndarray, singletalk_segment: np.ndarray) -> Dict:
        """Analyze total "five" patterns across both segments"""
        features = {}
        
        # Combine both segments
        if len(doubletalk_segment) > 0 and len(singletalk_segment) > 0:
            combined_audio = np.concatenate([doubletalk_segment, singletalk_segment])
        elif len(doubletalk_segment) > 0:
            combined_audio = doubletalk_segment
        elif len(singletalk_segment) > 0:
            combined_audio = singletalk_segment
        else:
            return self._get_zero_total_features()
        
        # Analyze combined segment
        total_duration = len(combined_audio) / self.sample_rate
        estimated_total_fives = int(total_duration)  # ~1 second per "five"
        
        if estimated_total_fives == 0:
            return self._get_zero_total_features()
        
        segment_length = len(combined_audio) // estimated_total_fives
        num_segments = min(estimated_total_fives, len(combined_audio) // segment_length)
        
        energies = []
        spectral_centroids = []
        
        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            
            if end > len(combined_audio):
                break
                
            segment = combined_audio[start:end]
            
            if len(segment) == 0:
                continue
            
            # Energy analysis
            energy = np.sqrt(np.mean(segment**2))
            energies.append(energy)
            
            # Spectral features
            try:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(segment, sr=self.sample_rate))
                spectral_centroids.append(spectral_centroid)
            except:
                spectral_centroids.append(0.0)
        
        # Total features
        features['total_detected_fives'] = len(energies)
        features['total_expected_fives'] = estimated_total_fives
        features['total_detection_ratio'] = len(energies) / estimated_total_fives if estimated_total_fives > 0 else 0.0
        features['total_duration'] = total_duration
        
        # Energy features across entire sequence
        if energies:
            features['total_energy_mean'] = np.mean(energies)
            features['total_energy_std'] = np.std(energies)
            features['total_energy_range'] = np.max(energies) - np.min(energies)
            
            # Overall energy trend
            if len(energies) > 1:
                try:
                    features['total_energy_trend'] = np.polyfit(range(len(energies)), energies, 1)[0]
                    features['total_energy_correlation'] = np.corrcoef(range(len(energies)), energies)[0, 1]
                except:
                    features['total_energy_trend'] = 0.0
                    features['total_energy_correlation'] = 0.0
            else:
                features['total_energy_trend'] = 0.0
                features['total_energy_correlation'] = 0.0
        else:
            features['total_energy_mean'] = 0.0
            features['total_energy_std'] = 0.0
            features['total_energy_range'] = 0.0
            features['total_energy_trend'] = 0.0
            features['total_energy_correlation'] = 0.0
        
        # Spectral consistency across entire sequence
        if spectral_centroids:
            features['total_spectral_consistency'] = 1 - np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
        else:
            features['total_spectral_consistency'] = 0.0
        
        # Segment transition features
        dt_duration = len(doubletalk_segment) / self.sample_rate if len(doubletalk_segment) > 0 else 0
        st_duration = len(singletalk_segment) / self.sample_rate if len(singletalk_segment) > 0 else 0
        
        features['dt_duration'] = dt_duration
        features['st_duration'] = st_duration
        features['dt_st_ratio'] = dt_duration / (st_duration + 1e-8)
        
        # Clean up any NaN or infinite values
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
    def _get_zero_five_features(self) -> Dict:
        """Return zero features for five pattern analysis"""
        return {
            'energy_trend': 0.0,
            'energy_correlation': 0.0,
            'repetition_consistency': 0.0,
            'energy_mean': 0.0,
            'energy_std': 0.0,
            'energy_range': 0.0,
            'spectral_consistency': 0.0,
            'num_detected_segments': 0,
            'expected_segments': 0,
            'detection_ratio': 0.0,
            'segment_duration': 0.0,
            'segment_type': 0.0
        }
    
    def _get_zero_total_features(self) -> Dict:
        """Return zero features for total pattern analysis"""
        return {
            'total_detected_fives': 0,
            'total_expected_fives': 0,
            'total_detection_ratio': 0.0,
            'total_duration': 0.0,
            'total_energy_mean': 0.0,
            'total_energy_std': 0.0,
            'total_energy_range': 0.0,
            'total_energy_trend': 0.0,
            'total_energy_correlation': 0.0,
            'total_spectral_consistency': 0.0,
            'dt_duration': 0.0,
            'st_duration': 0.0,
            'dt_st_ratio': 0.0
        }
    
    def detect_recording_delay(self, recorded_audio: np.ndarray, source_audio: np.ndarray, 
                              max_delay: float = 3.0) -> float:
        """
        Detect recording delay by cross-correlating recorded audio with source audio
        
        Args:
            recorded_audio: The recorded audio signal
            source_audio: The source/reference audio signal
            max_delay: Maximum delay to search for (in seconds)
            
        Returns:
            Detected delay in seconds (positive if recorded audio is delayed)
        """
        try:
            # Downsample both signals for faster correlation
            downsample_factor = 4
            recorded_down = recorded_audio[::downsample_factor]
            source_down = source_audio[::downsample_factor]
            
            # Use only first 30 seconds for correlation to avoid memory issues
            max_samples = 30 * self.sample_rate // downsample_factor
            recorded_down = recorded_down[:max_samples]
            source_down = source_down[:max_samples]
            
            # Cross-correlation
            correlation = np.correlate(recorded_down, source_down, mode='full')
            
            # Find peak correlation
            peak_idx = np.argmax(correlation)
            
            # Convert to delay in samples (at original sample rate)
            delay_samples = (peak_idx - (len(source_down) - 1)) * downsample_factor
            delay_seconds = delay_samples / self.sample_rate
            
            # Clamp to reasonable range
            delay_seconds = np.clip(delay_seconds, -max_delay, max_delay)
            
            return float(delay_seconds)
            
        except Exception as e:
            print(f"Warning: Could not detect delay: {e}")
            return 0.0
    
    def detect_five_timestamps(self, audio: np.ndarray, delay_compensation: float = 0.0, 
                              min_duration: float = 0.5, max_duration: float = 2.0) -> List[Tuple[float, float]]:
        """
        Detect timestamps of "five" utterances in audio
        
        Args:
            audio: Audio signal
            delay_compensation: Delay compensation in seconds
            min_duration: Minimum duration of a "five" utterance (seconds)
            max_duration: Maximum duration of a "five" utterance (seconds)
            
        Returns:
            List of (start_time, end_time) tuples for each detected "five"
        """
        try:
            # Apply delay compensation
            delay_samples = int(delay_compensation * self.sample_rate)
            if delay_samples > 0:
                audio = audio[delay_samples:]
            elif delay_samples < 0:
                audio = np.pad(audio, (-delay_samples, 0), mode='constant')
            
            # Calculate energy-based voice activity detection
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            # Calculate short-time energy
            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # Smooth energy curve
            window_size = 5
            if len(energy) > window_size:
                energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            
            # Find energy threshold (adaptive)
            energy_threshold = np.percentile(energy, 30)  # 30th percentile as threshold
            
            # Find voice activity regions
            voice_active = energy > energy_threshold
            
            # Find start and end points of voice segments
            voice_segments = []
            in_segment = False
            segment_start = 0
            
            for i, is_active in enumerate(voice_active):
                time_s = i * hop_length / self.sample_rate
                
                if is_active and not in_segment:
                    # Start of voice segment
                    segment_start = time_s
                    in_segment = True
                elif not is_active and in_segment:
                    # End of voice segment
                    segment_end = time_s
                    duration = segment_end - segment_start
                    
                    # Filter by duration (typical "five" is ~0.5-2 seconds)
                    if min_duration <= duration <= max_duration:
                        voice_segments.append((segment_start, segment_end))
                    
                    in_segment = False
            
            # Handle case where audio ends while in a segment
            if in_segment:
                segment_end = len(voice_active) * hop_length / self.sample_rate
                duration = segment_end - segment_start
                if min_duration <= duration <= max_duration:
                    voice_segments.append((segment_start, segment_end))
            
            # Refine segments using spectral features
            refined_segments = []
            for start_time, end_time in voice_segments:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                segment = audio[start_sample:end_sample]
                
                if len(segment) < self.sample_rate * 0.3:  # Skip very short segments
                    continue
                
                # Check if this looks like speech using spectral features
                try:
                    # Calculate spectral centroid
                    spec_centroid = np.mean(librosa.feature.spectral_centroid(segment, sr=self.sample_rate))
                    
                    # Calculate zero crossing rate
                    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                    
                    # Simple heuristic: speech typically has moderate spectral centroid and ZCR
                    if 500 < spec_centroid < 3000 and 0.02 < zcr < 0.3:
                        refined_segments.append((start_time, end_time))
                        
                except Exception:
                    # If spectral analysis fails, keep the segment anyway
                    refined_segments.append((start_time, end_time))
            
            return refined_segments
            
        except Exception as e:
            print(f"Warning: Error detecting five timestamps: {e}")
            return []

class DoubletalkDataset(Dataset):
    """Dataset for doubletalk performance assessment"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DoubletalkNet(nn.Module):
    """Neural network for doubletalk assessment"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], use_batch_norm: bool = True):
        super(DoubletalkNet, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            
            # Use BatchNorm or LayerNorm based on the flag
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            else:
                layers.append(nn.LayerNorm(hidden_size))
                
            prev_size = hidden_size
        
        # Output layer (total number of "five" utterances heard across both segments)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # For very small batches during evaluation, temporarily use eval mode for BatchNorm
        batch_size = x.size(0)
        
        # Handle single-sample batches with BatchNorm by using eval mode
        if batch_size == 1 and self.use_batch_norm and self.training:
            self.eval()  # Temporarily set to eval mode
            with torch.no_grad():
                output = self.network(x)
            self.train()  # Set back to train mode
            return output
        else:
            return self.network(x)

class DoubletalkTrainer:
    """Main trainer class for doubletalk assessment"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def prepare_data(self, wav_files: List[str], labels: List[int], 
                    source_file: str = None, auto_detect_delay: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from wav files and labels"""
        all_features = []
        valid_labels = []
        
        # Load source file for delay detection if provided
        source_audio = None
        if source_file and os.path.exists(source_file):
            try:
                source_audio, source_sr = librosa.load(source_file, sr=self.sample_rate)
                # Normalize and convert to int16 for efficiency
                source_audio = self._normalize_to_int16(source_audio)
                print(f"✅ Source file loaded: {source_file}")
                print(f"   Duration: {len(source_audio)/self.sample_rate:.1f}s")
            except Exception as e:
                print(f"⚠️  Could not load source file: {e}")
                source_audio = None
        
        print(f"Processing {len(wav_files)} audio files...")
        
        for i, wav_file in enumerate(wav_files):
            print(f"Processing {i+1}/{len(wav_files)}: {os.path.basename(wav_file)}")
            
            try:
                # Load audio with automatic resampling to target sample rate
                audio, original_sr = librosa.load(wav_file, sr=None)
                
                # Print original file info
                print(f"  Original: {original_sr}Hz, {len(audio)/original_sr:.1f}s, {audio.dtype}")
                
                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=0)
                
                # Resample to target sample rate if needed
                if original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                    print(f"  Resampled: {original_sr}Hz -> {self.sample_rate}Hz")
                
                # Normalize and convert to int16 for computational efficiency
                audio = self._normalize_to_int16(audio)
                
                # Detect recording delay if source file is available
                delay_compensation = 0.0
                if source_audio is not None and auto_detect_delay:
                    delay_compensation = self.feature_extractor.detect_recording_delay(audio, source_audio)
                    print(f"  Detected delay: {delay_compensation:.3f}s")
                
                # Check if audio is long enough after delay compensation
                min_duration = 50  # seconds (recorded length)
                min_samples = min_duration * self.sample_rate
                
                if len(audio) < min_samples:
                    print(f"  Warning: Audio too short ({len(audio)/self.sample_rate:.1f}s < {min_duration}s), padding with zeros")
                    audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
                
                # Extract features with delay compensation
                features = self.feature_extractor.extract_doubletalk_features(audio, delay_compensation)
                all_features.append(features)
                valid_labels.append(labels[i])
                
                print(f"  ✓ Successfully processed (delay: {delay_compensation:.3f}s)")
                
            except Exception as e:
                print(f"  ❌ Error processing {wav_file}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No audio files were successfully processed!")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(all_features)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Handle NaN values (replace with 0)
        df = df.fillna(0)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Convert to numpy arrays
        X = df.values
        y = np.array(valid_labels)
        
        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} files")
        print(f"📊 Feature statistics: min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
        
        return X, y
    
    def _normalize_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio and convert to int16 for computational efficiency"""
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Convert to int16 (range: -32768 to 32767)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Convert back to float for processing (more efficient than float64)
        return audio_int16.astype(np.float32) / 32767.0
    
    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the doubletalk assessment model"""
        
        # Make sure batch size is at least 2 to avoid BatchNorm errors
        batch_size = max(2, batch_size)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Handle case where there aren't enough samples
        if len(X_train) < batch_size:
            print(f"Warning: Not enough training samples ({len(X_train)}). Duplicating samples to meet batch size requirements.")
            # Duplicate samples to ensure we have enough for a batch
            repeat_count = (batch_size // len(X_train)) + 1
            X_train = np.repeat(X_train, repeat_count, axis=0)[:batch_size]
            y_train = np.repeat(y_train, repeat_count)[:batch_size]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = DoubletalkDataset(X_train_scaled, y_train)
        test_dataset = DoubletalkDataset(X_test_scaled, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Determine whether to use BatchNorm based on dataset size
        use_batch_norm = len(X_train) >= 4  # Only use BatchNorm if we have enough samples
        
        # Initialize model
        self.model = DoubletalkNet(input_size=X_train_scaled.shape[1], use_batch_norm=use_batch_norm)
        
        if not use_batch_norm:
            print("Warning: Small dataset detected. Using LayerNorm instead of BatchNorm.")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for features, labels in train_loader:
                # Skip empty batches
                if len(features) == 0:
                    continue
                    
                # For single-sample batches, duplicate the sample to avoid BatchNorm issues
                if len(features) == 1 and self.model.use_batch_norm:
                    features = features.repeat(2, 1)
                    labels = labels.repeat(2)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            # Validation
            self.model.eval()
            test_loss = 0.0
            test_batches = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    # Skip empty batches
                    if len(features) == 0:
                        continue
                        
                    outputs = self.model(features)
                    loss = criterion(outputs.squeeze(), labels)
                    test_loss += loss.item()
                    test_batches += 1
            
            train_loss = train_loss / train_batches if train_batches > 0 else 0
            test_loss = test_loss / test_batches if test_batches > 0 else 0
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Evaluate final model
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(torch.FloatTensor(X_train_scaled)).squeeze().numpy()
            test_pred = self.model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nFinal Results:")
        print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, test_losses)
        
        # Plot predictions vs actual
        self.plot_predictions(y_test, test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def plot_training_curves(self, train_losses: List[float], test_losses: List[float]):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Total "Five" Heard (Both Segments)')
        plt.ylabel('Predicted Total "Five" Heard (Both Segments)')
        plt.title('Predictions vs Actual Values - Total "Five" Count')
        plt.grid(True)
        plt.show()
    
    def predict(self, wav_file: str, source_file: str = None, delay_compensation: float = 0.0) -> float:
        """Predict the total number of "five" utterances across both segments for a single wav file"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        try:
            # Load and process audio with robust handling
            audio, original_sr = librosa.load(wav_file, sr=None)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Resample to target sample rate if needed
            if original_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
            
            # Normalize and convert to int16 for efficiency
            audio = self._normalize_to_int16(audio)
            
            # Auto-detect delay if source file is provided
            if source_file and os.path.exists(source_file) and delay_compensation == 0.0:
                try:
                    source_audio, _ = librosa.load(source_file, sr=self.sample_rate)
                    source_audio = self._normalize_to_int16(source_audio)
                    delay_compensation = self.feature_extractor.detect_recording_delay(audio, source_audio)
                    print(f"Auto-detected delay: {delay_compensation:.3f}s")
                except Exception as e:
                    print(f"Could not auto-detect delay: {e}")
                    delay_compensation = 0.0
            
            # Ensure minimum length
            min_duration = 50  # seconds
            min_samples = min_duration * self.sample_rate
            
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
            
            # Extract features with delay compensation
            features = self.feature_extractor.extract_doubletalk_features(audio, delay_compensation)
            
            # Convert to DataFrame and ensure all feature columns are present
            df = pd.DataFrame([features])
            
            # Reindex to match training features
            df = df.reindex(columns=self.feature_names, fill_value=0)
            
            # Handle NaN and infinite values
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Scale features
            features_scaled = self.scaler.transform(df.values)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(torch.FloatTensor(features_scaled)).item()
            
            return max(0, min(42, prediction))  # Clamp between 0 and 42 (max possible across both segments)
            
        except Exception as e:
            print(f"Error predicting for {wav_file}: {e}")
            return 0.0
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        # Add sklearn's StandardScaler to the safe globals list before loading
        import torch.serialization
        torch.serialization.add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
        
        # Try with weights_only=False first, if that fails, try with the safe globals
        try:
            checkpoint = torch.load(filepath, weights_only=False)
        except Exception as e:
            print(f"Trying alternate loading method with safe globals: {e}")
            checkpoint = torch.load(filepath)
        
        # Recreate model with correct input size
        input_size = len(checkpoint['feature_names'])
        
        # Check if use_batch_norm is in the saved state
        use_batch_norm = checkpoint.get('use_batch_norm', True)
        
        self.model = DoubletalkNet(input_size=input_size, use_batch_norm=use_batch_norm)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler = checkpoint['scaler']
        self.feature_names = checkpoint['feature_names']
        
        print(f"Model loaded from {filepath}")
        
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'use_batch_norm': getattr(self.model, 'use_batch_norm', True)
        }, filepath)
        print(f"Model saved to {filepath}")

def validate_audio_file(wav_file: str) -> Dict:
    """Validate and get information about an audio file"""
    try:
        # Load audio file to check its properties
        audio, sr = librosa.load(wav_file, sr=None)
        
        info = {
            'valid': True,
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'channels': 1 if audio.ndim == 1 else audio.shape[0],
            'samples': len(audio),
            'dtype': str(audio.dtype),
            'max_amplitude': np.max(np.abs(audio)),
            'rms_energy': np.sqrt(np.mean(audio**2))
        }
        
        # Check for common issues
        if info['duration'] < 30:  # Less than 30 seconds
            info['warning'] = f"Short duration: {info['duration']:.1f}s"
        
        if info['max_amplitude'] == 0:
            info['warning'] = "Silent audio"
        
        return info
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'sample_rate': 0,
            'duration': 0,
            'channels': 0,
            'samples': 0,
            'dtype': 'unknown',
            'max_amplitude': 0,
            'rms_energy': 0
        }

def create_dataset_from_excel(excel_file: str, root_folder: str) -> Tuple[List[str], List[int], List[str]]:
    """Create dataset from Excel file with ground truth labels"""
    
    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name="Folder Analysis_w_Subjective")
        print(f"Loaded Excel file with {len(df)} rows")
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {e}")
    
    # Define column mappings for different test scenarios
    label_columns = {
        'S-DT-AR (comms)': 'AR_DT',
        'S-DT-AR (default)': 'AR_DT', 
        'S-DT-RR (comms)': 'RR_DT',
        'S-DT-RR (default)': 'RR_DT',
        'S-DT-RR_5dB (comms)': 'RR_DT_5dB',
        'S-DT-RR_5dB (default)': 'RR_DT_5dB'
    }
    
    wav_files = []
    labels = []
    scenarios = []
    
    print("Processing folders and finding corresponding WAV files...")
    print("(Note: Audio files will be automatically converted to consistent format)")
    
    # Process each row in the Excel file
    for idx, row in df.iterrows():
        folder_name = row['Folder name']
        folder_path = os.path.join(root_folder, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        print(f"\nProcessing folder: {folder_name}")
        
        # Find subfolders
        subfolders = {}
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                if item.endswith('AR_DT'):
                    subfolders['AR_DT'] = item_path
                elif item.endswith('RR_DT') and not item.endswith('RR_DT_5dB'):
                    subfolders['RR_DT'] = item_path
                elif item.endswith('RR_DT_5dB'):
                    subfolders['RR_DT_5dB'] = item_path
        
        print(f"  Found subfolders: {list(subfolders.keys())}")
        
        # Process each label column
        for label_col, subfolder_type in label_columns.items():
            if label_col in row and pd.notna(row[label_col]) and subfolder_type in subfolders:
                label_value = row[label_col]
                
                # Skip if label is not a valid number
                try:
                    label_value = int(float(label_value))
                    if label_value < 0 or label_value > 42:  # Updated range for total count
                        print(f"  Warning: Invalid label value {label_value} for {label_col}")
                        continue
                except (ValueError, TypeError):
                    print(f"  Warning: Cannot convert label to number: {label_value} for {label_col}")
                    continue
                
                # Find corresponding WAV file
                subfolder_path = subfolders[subfolder_type]
                
                # Determine if we need comms or default wav file
                if 'comms' in label_col:
                    wav_pattern = os.path.join(subfolder_path, "comms_*.wav")
                    wav_type = "comms"
                else:  # default
                    wav_pattern = os.path.join(subfolder_path, "default_*.wav")
                    wav_type = "default"
                
                matching_wavs = glob.glob(wav_pattern)
                
                if matching_wavs:
                    wav_file = matching_wavs[0]  # Take the first match
                    
                    # Validate audio file
                    audio_info = validate_audio_file(wav_file)
                    
                    if audio_info['valid']:
                        wav_files.append(wav_file)
                        labels.append(label_value)
                        scenarios.append(f"{folder_name}_{label_col}")
                        
                        print(f"  ✓ {wav_type}: {os.path.basename(wav_file)} -> Label: {label_value}")
                        print(f"    Audio: {audio_info['sample_rate']}Hz, {audio_info['duration']:.1f}s, {audio_info['channels']}ch")
                        
                        if 'warning' in audio_info:
                            print(f"    ⚠️  {audio_info['warning']}")
                    else:
                        print(f"  ❌ Invalid audio file: {wav_file} - {audio_info['error']}")
                else:
                    print(f"  ❌ No WAV file found for pattern: {wav_pattern}")
    
    print(f"\n✅ Total dataset: {len(wav_files)} WAV files with labels")
    
    # Show summary of audio formats found
    if wav_files:
        print("\n📊 Audio format summary:")
        sample_rates = set()
        durations = []
        
        for wav_file in wav_files[:5]:  # Sample first 5 files
            info = validate_audio_file(wav_file)
            if info['valid']:
                sample_rates.add(info['sample_rate'])
                durations.append(info['duration'])
        
        print(f"  Sample rates found: {sorted(sample_rates)} Hz")
        if durations:
            print(f"  Duration range: {min(durations):.1f}s to {max(durations):.1f}s")
        print("  (All files will be automatically converted to 16kHz mono during processing)")
    
    return wav_files, labels, scenarios

def create_sample_dataset(wav_folder: str, labels_file: str = None) -> Tuple[List[str], List[int]]:
    """Create a sample dataset from wav files and labels (legacy function)"""
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(wav_folder, "*.wav"))
    
    if not wav_files:
        raise ValueError(f"No wav files found in {wav_folder}")
    
    # If labels file is provided, load it
    if labels_file and os.path.exists(labels_file):
        if labels_file.endswith('.xlsx') or labels_file.endswith('.xls'):
            # Handle Excel file
            try:
                df = pd.read_excel(labels_file, sheet_name="Folder Analysis_w_Subjective")
                print("Excel file detected, please use the new Excel-based workflow")
                return [], []
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                return [], []
        else:
            # Handle CSV file (legacy)
            labels_df = pd.read_csv(labels_file)
            labels = []
            filtered_wav_files = []
            
            for wav_file in wav_files:
                filename = os.path.basename(wav_file)
                matching_row = labels_df[labels_df['filename'] == filename]
                
                if not matching_row.empty:
                    labels.append(int(matching_row.iloc[0]['num_five_heard']))
                    filtered_wav_files.append(wav_file)
            
            return filtered_wav_files, labels
    
    else:
        # Create synthetic labels for demonstration
        # In practice, you would need human-annotated labels
        print("Warning: No labels file provided. Creating synthetic labels for demonstration.")
        
        synthetic_labels = []
        for wav_file in wav_files:
            # Generate random labels between 0-21 for demonstration
            # In practice, these should be human-annotated
            synthetic_labels.append(np.random.randint(0, 22))
        
        return wav_files, synthetic_labels

def main():
    """Main function to demonstrate the training process"""
    
    print("🎯 Doubletalk AI Training System")
    print("=" * 50)
    
    # Check if user wants to use Excel file or legacy CSV
    use_excel = input("Do you have an Excel file with ground truth labels? (y/n): ").strip().lower()
    
    if use_excel == 'y':
        # Excel-based workflow
        excel_file = input("Enter the path to your Excel file: ").strip()
        root_folder = input("Enter the root folder containing all the analyzed folders: ").strip()
        
        # Ask for source file for delay compensation
        source_file = input("Enter the path to source audio file (optional, for delay compensation): ").strip()
        if source_file and not os.path.exists(source_file):
            print("⚠️  Source file not found. Proceeding without delay compensation.")
            source_file = None
        
        if not excel_file or not os.path.exists(excel_file):
            print("❌ Invalid Excel file path!")
            return
        
        if not root_folder or not os.path.exists(root_folder):
            print("❌ Invalid root folder path!")
            return
        
        # Create dataset from Excel
        try:
            wav_files, labels, scenarios = create_dataset_from_excel(excel_file, root_folder)
            
            if not wav_files:
                print("❌ No WAV files found with valid labels!")
                return
            
            print(f"✅ Found {len(wav_files)} WAV files with labels")
            print(f"📊 Label distribution: Min={min(labels)}, Max={max(labels)}, Mean={np.mean(labels):.1f}")
            
            # Show scenarios breakdown
            scenario_counts = {}
            for scenario in scenarios:
                scenario_type = scenario.split('_')[-1]  # Get the label column type
                scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
            
            print("📈 Scenario breakdown:")
            for scenario_type, count in scenario_counts.items():
                print(f"  {scenario_type}: {count} samples")
            
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return
    
    else:
        # Legacy CSV workflow
        wav_folder = input("Enter the folder containing wav files: ").strip()
        labels_file = input("Enter the path to labels CSV file (optional, press Enter to skip): ").strip()
        
        if not wav_folder or not os.path.exists(wav_folder):
            print("❌ Invalid wav folder path!")
            return
        
        if labels_file and not os.path.exists(labels_file):
            print("⚠️  Labels file not found. Will create synthetic labels.")
            labels_file = None
        
        # Create dataset
        try:
            wav_files, labels = create_sample_dataset(wav_folder, labels_file)
            scenarios = [f"sample_{i}" for i in range(len(wav_files))]
            
            if not wav_files:
                print("❌ No WAV files found!")
                return
            
            print(f"✅ Found {len(wav_files)} wav files with labels")
            
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return
    
    # Initialize trainer
    print("\n🤖 Initializing AI trainer...")
    trainer = DoubletalkTrainer(sample_rate=16000)
    
    # Prepare data
    print("\n📊 Extracting audio features...")
    if use_excel == 'y':
        X, y = trainer.prepare_data(wav_files, labels, source_file)
    else:
        X, y = trainer.prepare_data(wav_files, labels)
    
    if len(X) == 0:
        print("❌ No features extracted!")
        return
    
    # Train model
    print("\n🚀 Training AI model...")
    results = trainer.train_model(X, y, epochs=100, batch_size=16)
    
    # Save model
    if use_excel == 'y':
        model_path = os.path.join(os.path.dirname(excel_file), "doubletalk_model.pth")
    else:
        model_path = os.path.join(os.path.dirname(wav_folder), "doubletalk_model.pth")
    
    trainer.save_model(model_path)
    
    # Test prediction on sample files
    print("\n🔍 Testing predictions on sample files...")
    sample_indices = np.random.choice(len(wav_files), min(5, len(wav_files)), replace=False)
    
    for idx in sample_indices:
        sample_file = wav_files[idx]
        actual_label = labels[idx]
        prediction = trainer.predict(sample_file)
        scenario = scenarios[idx]
        
        print(f"  {scenario}:")
        print(f"    File: {os.path.basename(sample_file)}")
        print(f"    Actual: {actual_label} total 'five' utterances")
        print(f"    Predicted: {prediction:.1f} total 'five' utterances")
        print(f"    Error: {abs(prediction - actual_label):.1f}")
        print()
    
    # Summary
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print(f"📊 Final Test R² Score: {results['test_r2']:.3f}")
    print(f"📊 Final Test MAE: {results['test_mae']:.2f}")
    
    # Save detailed results
    results_file = model_path.replace('.pth', '_results.txt')
    with open(results_file, 'w') as f:
        f.write("Doubletalk AI Training Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {len(wav_files)} WAV files\n")
        f.write(f"Features: {X.shape[1]} dimensions\n")
        f.write(f"Label range: {min(labels)} to {max(labels)}\n")
        f.write(f"Label mean: {np.mean(labels):.2f}\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"Train MSE: {results['train_mse']:.4f}\n")
        f.write(f"Test MSE: {results['test_mse']:.4f}\n")
        f.write(f"Train MAE: {results['train_mae']:.4f}\n")
        f.write(f"Test MAE: {results['test_mae']:.4f}\n")
        f.write(f"Train R²: {results['train_r2']:.4f}\n")
        f.write(f"Test R²: {results['test_r2']:.4f}\n")
        
        if use_excel == 'y':
            f.write("\nScenario Breakdown:\n")
            for scenario_type, count in scenario_counts.items():
                f.write(f"{scenario_type}: {count} samples\n")
    
    print(f"📄 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
