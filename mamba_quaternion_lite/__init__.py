"""
Mamba-Quaternion-Lite: Un Modèle SSD Sélectif Multi-États avec
Dynamique Scalaire et Projections Quaternioniques
"""

from .quaternion import QuaternionLinear, quaternion_multiply
from .mamba_quat_lite import MambaQuaternionLiteBlock
from .model import MambaQuaternionLiteModel

# Versions optimisées
try:
    from .mamba_quat_lite_fast import (
        MambaQuaternionLiteFastBlock,
        FastResidualBlock,
        enable_tf32,
        compile_model,
    )
    from .model_fast import (
        MambaQuaternionLiteFastModel,
        create_optimized_model,
    )
    from .optimized_kernels import (
        optimized_parallel_scan,
        TRITON_AVAILABLE,
    )
    FAST_VERSION_AVAILABLE = True
except ImportError as e:
    FAST_VERSION_AVAILABLE = False
    print(f"Warning: Fast version not available: {e}")

__all__ = [
    # Base version
    'QuaternionLinear',
    'quaternion_multiply',
    'MambaQuaternionLiteBlock',
    'MambaQuaternionLiteModel',
]

# Add optimized versions if available
if FAST_VERSION_AVAILABLE:
    __all__.extend([
        'MambaQuaternionLiteFastBlock',
        'MambaQuaternionLiteFastModel',
        'FastResidualBlock',
        'create_optimized_model',
        'enable_tf32',
        'compile_model',
        'optimized_parallel_scan',
        'TRITON_AVAILABLE',
    ])
