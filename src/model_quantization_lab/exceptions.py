"""Custom exceptions for model quantization lab."""


class QuantizationError(Exception):
    """Base exception for all quantization-related errors."""


class QuantizationConfigError(QuantizationError):
    """Raised when quantization configuration is invalid."""


class ModelLoadError(QuantizationError):
    """Raised when a model fails to load."""


class QuantizationFailedError(QuantizationError):
    """Raised when the quantization process fails."""


class EvaluationError(QuantizationError):
    """Raised when quality evaluation fails."""


class ProfilingError(QuantizationError):
    """Raised when performance profiling fails."""
