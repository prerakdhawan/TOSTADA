'''
Tostada: Toolkit for Spatially TAilored Disordered Arrangements.

This package provides:
- PointDistribution
- PhaseDistribution

'''  

# package version
from importlib.metadata import version as _pkg_version  
from importlib.metadata import PackageNotFoundError  
try: 
    __version__ = _pkg_version("tostada")
except PackageNotFoundError:  
    __version__ = "0.0.0+unknown" 

import os
import shutil
import subprocess

def _detect_nvidia_gpu():
    """Return True if nvidia-smi is runnable and reports devices."""
    try:
        if shutil.which("nvidia-smi") is None:
            return False
        cp = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
        return cp.returncode == 0 and cp.stdout.strip() != ""
    except Exception:
        return False

if os.environ.get("FORCE_JAX_CPU", "") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    if not _detect_nvidia_gpu():
        os.environ["JAX_PLATFORMS"] = "cpu"
    # else: leave JAX_PLATFORMS unset so jax can try to use GPU

from .PointDistribution import PointDistribution  # noqa: F401
from .PhaseDistribution import PhaseDistribution           # noqa: F401
from .phaseprocess import Phaseprocess #noqa : F401
from .pointprocess import Pointprocess #noqa : F401
from .util.materials import Material #noqa : F401
from .plot_util import Visualize #noqa : F401
from .Optimization import Optimization # noqa : F401
from .util.Utility import Spectrum # noqa : F401
from .util.Utility import read_file # noqa : F401
def device_info():                                              
    """Report the JAX devices in use. Replaces the old print-on-import."""  # 
    import jax                                                  
    return jax.devices()  