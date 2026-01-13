"""
Signals Module
==============

This module contains classes for generating and representing signals
used in the delta-sigma DAC simulation. 
"""

from .digital_signal_generator import DigitalSignalGenerator
from .signal_container import SignalContainer

__all__ = ["DigitalSignalGenerator", "SignalContainer"]