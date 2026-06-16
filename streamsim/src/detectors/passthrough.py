"""
Change point detector that treats any non-None feature value as a change point.
(passthrough)

Detection Method:
    When a valid feature (like a detected peak) is passed in, it flags it as a change
    point by returning True and setting `drift_detected` to True. This lets downstream
    components respond to detected events without extra interpretation logic.

Important Dependencies:
    - streamsim.src.core.interfaces.StreamingChangePointDetector: Base interface

Author: F.Feenstra
"""

from typing import Optional
from streamsim.src.core.interfaces import StreamingChangePointDetector

class PeakPassThrough(StreamingChangePointDetector):
    """ 
    Simple pass-through detector for peak detection.
    Returns True (and sets drift_detected=True) 
    whenever a valid feature (for instance a peak) is provided.
    """
    
    def __init__(self):
        self._drift_detected = False

    def update(self, feature_value: Optional[float], raw_sample=None) -> bool:
        """
        Returns True if feature_value is not None.
        Crucially, this also updates the drift_detected property.
        """
        flag = feature_value is not None
        self._drift_detected = flag 
        return flag
    
    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
    
    def reset(self) -> None:
        self._drift_detected = False