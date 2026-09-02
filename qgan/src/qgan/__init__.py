"""QGAN condivisa fra dominio IDS e dominio QKD.

Implementazione della Fase 2. L'architettura e' congelata da SPEC-FASE2:
ogni modifica alle scelte elencate in §12 come congelate invalida la misura
di O1 e va discussa fra le due parti prima di essere applicata.
"""

from .circuit import QuantumGenerator
from .discriminator import Discriminator
from .model import QGAN
from . import metrics, autoregressive

__all__ = [
    "QGAN",
    "QuantumGenerator",
    "Discriminator",
    "metrics",
    "autoregressive",
]
__version__ = "0.2.0"