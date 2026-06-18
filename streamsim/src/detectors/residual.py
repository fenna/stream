"""
Residual Threshold Detector

Detects when the difference between the raw signal and the model (smoothed)
value exceeds a configurable threshold.
"""

import logging
from typing import Optional
from streamsim.src.core.interfaces import StreamingChangePointDetector

# Get a logger specific to this module
logger = logging.getLogger(__name__)


class ResidualThresholdDetector(StreamingChangePointDetector):
    """
    Flags a detection when |raw_sample - model_value| > threshold.
    """
    
    def __init__(self, threshold: float = 2.0, min_samples: int = 20):
        self.threshold = threshold
        self.min_samples = min_samples
        self._sample_count = 0
        self._last_residual = 0.0
        self._drift_detected = False
        
        logger.info(f"[ResidualDetector] Initialized: threshold={threshold}, min_samples={min_samples}")
    
    def update(self, feature_value: Optional[float], raw_sample: Optional[float] = None) -> bool:
        self._sample_count += 1
        self._drift_detected = False
        
        # 1. Log Warm-up Phase
        if self._sample_count <= self.min_samples:
            # Optional: Log every Nth sample to avoid spam during warmup
            if self._sample_count % 50 == 0:
                logger.debug(
                    f"[ResidualDetector] Warm-up: {self._sample_count}/{self.min_samples} "
                    f"(raw={raw_sample:.3f}, model={feature_value})"
                )
            return False
        
        # 2. Validate Inputs
        if feature_value is None or raw_sample is None:
            logger.warning(f"[ResidualDetector] Missing data: raw={raw_sample}, model={feature_value}")
            return False
        
        # 3. Calculate Residual
        residual = abs(raw_sample - feature_value)
        self._last_residual = residual
        
        # 4. Log Every Sample (DEBUG level) or just Anomalies (INFO level)
        # Uncomment the line below if you want to see EVERY calculation (can be verbose!)
        # logger.debug(f"[ResidualDetector] Check: residual={residual:.4f} | raw={raw_sample:.4f} | model={feature_value:.4f}")
        
        # 5. Check Threshold
        if residual > self.threshold:
            self._drift_detected = True
            
            # Log the ANOMALY clearly
            logger.info(
                f"!! ANOMALY DETECTED: residual={residual:.4f} > threshold={self.threshold:.4f} | "
                f"Raw={raw_sample:.4f} | Model={feature_value:.4f} | Count={self._sample_count}"
            )
            return True
        
        return False
    
    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
    
    @property
    def last_residual(self) -> float:
        return self._last_residual
    
    def reset(self) -> None:
        logger.info("[ResidualDetector] Resetting state.")
        self._sample_count = 0
        self._last_residual = 0.0
        self._drift_detected = False