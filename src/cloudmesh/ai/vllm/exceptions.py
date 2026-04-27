class VLLMError(Exception):
    """Base exception for all vLLM related errors."""
    pass

class VLLMConnectionError(VLLMError):
    """Raised when there is a failure connecting to the remote host or API."""
    pass

class VLLMConfigError(VLLMError):
    """Raised when there is a configuration error or missing required field."""
    pass

class VLLMRuntimeError(VLLMError):
    """Raised when a remote operation fails during execution."""
    pass