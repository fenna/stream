"""
Robust Heart Rate Anomaly Detector

This module provides a streaming change point detector specifically designed for
identifying persistent heart rate deviations from a fixed baseline established
during an initial warmup period.

Detection Method:
    The detector collects heart rate samples during a configurable warmup phase
    to establish a baseline mean and standard deviation. Subsequent samples are
    compared against this baseline using z-scores. A change point is confirmed
    only after a configurable streak of consecutive outliers, reducing false
    positives from transient noise.


Important Dependencies:
    - collections.deque: Efficient circular buffer for warmup data
    - streamsim.src.core.interfaces.StreamingChangePointDetector: Base interface

Author: F.Feenstra
"""

from collections import deque
import numpy as np
from streamsim.src.core.interfaces import StreamingChangePointDetector
from streamsim.src.core.logger import get_logger  # Import logger

logger = get_logger(__name__)

class HeartRateAnomalyDetector(StreamingChangePointDetector):
    """
    Detects heart rate deviations against a FIXED baseline established during warmup.
    
    Unlike rolling window detectors, this maintains the original baseline and only
    updates it slowly (or not at all), making it sensitive to sudden changes that
    persist over time.
    """
    
    def __init__(
        self,
        threshold_std: float = 3.0,      # How many std devs from baseline
        warmup_beats: int = 15,          # Beats to establish baseline
        confirmation_count: int = 3,     # Consecutive outliers to confirm
        recovery_count: int = 5,         # Consecutive normal values to clear alert
        baseline_adaptation: float = 0.0 # How fast baseline adapts (0 = fixed, 0.1 = slow)
    ):
        self.threshold_std = threshold_std
        self.warmup_beats = warmup_beats
        self.confirmation_count = confirmation_count
        self.recovery_count = recovery_count
        self.baseline_adaptation = baseline_adaptation
        
        # Warmup buffer
        self._warmup_buffer = deque(maxlen=warmup_beats)
        
        # Fixed baseline (established after warmup)
        self._baseline_mean = None
        self._baseline_std = None
        
        # State tracking
        self._beat_count = 0
        self._outlier_streak = 0
        self._normal_streak = 0
        self._drift_detected = False

        # Log initialization at DEBUG level
        logger.debug(
            f"HeartRateAnomalyDetector initialized: "
            f"threshold_std={threshold_std}, warmup_beats={warmup_beats}, "
            f"confirmation_count={confirmation_count}"
        )

    def update(self, feature_value: float, raw_sample=None) -> bool:
        """
        Update detector with new heart rate value.
        
        Args:
            feature_value (float): Current heart rate in BPM.
        
        Returns:
            bool: True if a significant deviation is confirmed.
        """
        if feature_value is None:
            return False

        self._beat_count += 1
        
        # Phase 1: Warmup - collect baseline data
        if self._beat_count <= self.warmup_beats:
            self._warmup_buffer.append(feature_value)
            return False
        
        # Phase 2: Establish baseline after warmup completes
        if self._baseline_mean is None:
            arr = np.array(self._warmup_buffer)
            self._baseline_mean = arr.mean()
            self._baseline_std = max(arr.std(), 5.0)  # Floor at 5 BPM std
            logger.info(
                f"Baseline established: {self._baseline_mean:.1f} ± {self._baseline_std:.1f} BPM"
            )
            return False
        
        # Phase 3: Compare against fixed baseline
        deviation = abs(feature_value - self._baseline_mean)
        z_score = deviation / self._baseline_std
        
        is_outlier = z_score > self.threshold_std
        
        if is_outlier:
            self._outlier_streak += 1
            self._normal_streak = 0
            
            # Confirm anomaly after consecutive outliers
            if self._outlier_streak >= self.confirmation_count:
                if not self._drift_detected:
                    logger.warning(
                        f"ANOMALY DETECTED at beat {self._beat_count}: "
                        f"HR={feature_value:.1f}, baseline={self._baseline_mean:.1f}"
                    )
                self._drift_detected = True
                return True
        else:
            self._normal_streak += 1
            
            # Clear alert after sustained return to normal
            if self._normal_streak >= self.recovery_count:
                if self._drift_detected:
                    logger.info(f"Anomaly CLEARED at beat {self._beat_count}")
                self._drift_detected = False
                self._outlier_streak = 0
        
        # Optional: Slowly adapt baseline (for long-term drift accommodation)
        if self.baseline_adaptation > 0 and not self._drift_detected:
            self._baseline_mean = (
                (1 - self.baseline_adaptation) * self._baseline_mean +
                self.baseline_adaptation * feature_value
            )
        
        return self._drift_detected

    @property
    def drift_detected(self) -> bool:
        """Check if a change point was recently detected."""
        return self._drift_detected

    def reset(self) -> None:
        """Reset internal state."""
        logger.debug("Detector state reset")
        self._warmup_buffer.clear()
        self._baseline_mean = None
        self._baseline_std = None
        self._beat_count = 0
        self._outlier_streak = 0
        self._normal_streak = 0
        self._drift_detected = False