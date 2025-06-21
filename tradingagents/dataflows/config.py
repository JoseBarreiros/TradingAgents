import copy
import tradingagents.default_config as default_config
from typing import Dict, Optional

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = None


def initialize_config(default_cfg: Dict = None):
    """Initialize the configuration with default values."""
    global _config, DATA_DIR
    if default_cfg is None:
        default_cfg = default_config.DEFAULT_CONFIG
    _config = copy.deepcopy(default_cfg)
    DATA_DIR = _config["data_dir"]


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config, DATA_DIR
    if _config is None:
        initialize_config()
    _config.update(config)
    DATA_DIR = _config["data_dir"]


def get_config() -> Dict:
    """Get a deep copy of the current configuration."""
    if _config is None:
        initialize_config()
    return copy.deepcopy(_config)


# Initialize with default config
initialize_config()
