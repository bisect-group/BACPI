"""Builds BACPIv2, the memory-efficient equivalent of the original BACPI model (code/model.py)."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

from emulator_bench.model_v2 import BACPIv2, convert_v1_state_dict


@dataclass
class BACPIModelConfig:
    gat_dim: int = 50
    num_head: int = 3
    dropout: float = 0.1
    alpha: float = 0.1
    comp_dim: int = 80
    prot_dim: int = 80
    latent_dim: int = 80
    window: int = 5
    layer_cnn: int = 3
    layer_out: int = 3


def build_model(n_atom_tokens: int, n_amino_tokens: int, config: BACPIModelConfig) -> BACPIv2:
    params = SimpleNamespace(**asdict(config))
    return BACPIv2("affinity", int(n_atom_tokens), int(n_amino_tokens), params)


__all__ = ["BACPIModelConfig", "build_model", "convert_v1_state_dict"]
