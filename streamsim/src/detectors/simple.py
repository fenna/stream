"""
Simple Statistical Change Point Detector

This module provides a lightweight streaming change point detector that identifies
significant deviations from recent signal behavior using a z-score-based approach.

Detection Method:
    The detector maintains a rolling history of recent samples and computes the
    mean and standard deviation. A new sample is flagged as a change point if it
    deviates from the recent mean by more than a configurable number of standard
    deviations (threshold).

Important Dependencies:
    - collections.deque: Efficient circular buffer
    - streamsim.src.core.interfaces.StreamingChangePointDetector: Base interface

Author: F.Feenstra

"""


from collections import deque
import numpy as np
from streamsim.src.core.interfaces import StreamingChangePointDetector


class SimpleDetector(StreamingChangePointDetector):
    """
    Very simple change point detector that flags points deviating from recent mean
    by more than a specified number of standard deviations.
    """
    
    def __init__(
            self, 
            threshold: float = 2.0,
            history: int = 50
        ):
        """
        Initialize the SimpleDetector with a configurable sensitivity threshold.
        
        Args:
            threshold (float): Number of standard deviations for change point detection.
                               Default: 2.0.
        """
        self.threshold = threshold
        self.history = deque(maxlen=history)


    def update(self, feature_value: float, raw_sample=None) -> bool:
        """
        Process a new sample and determine if it represents a change point.
        
        Adds the sample to the rolling history, then checks if it deviates
        significantly from the recent mean using a z-score-based approach.
        
        Args:
            x (float): The new sample value to evaluate.
        
        Returns:
            bool: True if the sample is flagged as a change point, False otherwise.
        
        """
        self.history.append(feature_value)

        if len(self.history) > 20:
            arr = np.array(self.history)
            # Exclude current sample from statistics (use all but last)
            mean = arr[:-1].mean()
            std = arr[:-1].std()

            if std > 0 and abs(feature_value - mean) > self.threshold * std:
                return True
        return False