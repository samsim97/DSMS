"""
Delta-Sigma DAC Simulation Package
==================================

This package provides a complete simulation environment for evaluating
Delta-Sigma Modulator Digital-to-Analog Converter (DSM-DAC) parameters
before FPGA implementation. 

The package is designed with FPGA implementation in mind, particularly
for low-power cryogenic environments.

Package Structure:
- signals/: Input signal generation and representation
- modulator/: Delta-Sigma Modulator core components
- reconstruction/: Output reconstruction filters
- metrics/: SNR, ENOB, and FPGA-relevant metrics
- visualization/: Plotting and analysis tools
- simulation/: Simulation orchestration

Author: Generated for FPGA DSM-DAC evaluation
"""

__version__ = "1.0.0"