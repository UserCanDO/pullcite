"""
Pipeline module for Pullcite.

This module provides the extraction pipeline components:
- Strategy: Defines prompts for each role
- Roles: Extractor, Verifier, Corrector
- Patcher: Applies corrections to data
"""

from .strategy import (
    ExtractionStrategy,
    DefaultStrategy,
    StrategyContext,
)
from .roles import (
    ExtractorRole,
    VerifierRole,
    CorrectorRole,
    ExtractionError,
    VerificationError,
    CorrectionError,
)
from .patcher import (
    Patch,
    PatchResult,
    Patcher,
    PatchError,
    create_patches,
)

__all__ = [
    # Strategy
    "ExtractionStrategy",
    "DefaultStrategy",
    "StrategyContext",
    # Roles
    "ExtractorRole",
    "VerifierRole",
    "CorrectorRole",
    "ExtractionError",
    "VerificationError",
    "CorrectionError",
    # Patcher
    "Patch",
    "PatchResult",
    "Patcher",
    "PatchError",
    "create_patches",
]
