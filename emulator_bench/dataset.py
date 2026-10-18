from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from emulator_bench.feature_pipeline import load_npz_payload


class CacheStore:
    """Loads cached npz payloads on first use."""

    def __init__(self, path_fn):
        self._path_fn = path_fn
        self._cache: Dict[Path, Dict] = {}

    def get(self, value: str) -> Dict:
        path = self._path_fn(value)
        if path not in self._cache:
            self._cache[path] = load_npz_payload(path)
        return self._cache[path]


class BACPIDataset(Dataset):
    def __init__(self, frame, compound_store: CacheStore, protein_store: CacheStore, smiles_col: str, sequence_col: str, target_col: str):
        self.frame = frame.reset_index(drop=True)

        # Pull hot-path data into plain Python lists so __getitem__ never touches pandas
        self._smiles_list: List[str] = self.frame[smiles_col].astype(str).tolist()
        self._sequence_list: List[str] = self.frame[sequence_col].astype(str).tolist()
        self._targets: List[float] = self.frame[target_col].tolist()

        # Pre-resolve cache references per row; avoids a hash + dict lookup on every __getitem__
        # call. Duplicate SMILES/sequences share the same dict object (no extra memory cost).
        self._compound_items: List[Dict] = [compound_store.get(s) for s in self._smiles_list]
        self._protein_items: List[Dict] = [protein_store.get(s) for s in self._sequence_list]

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, index):
        ci = self._compound_items[index]
        pi = self._protein_items[index]
        return {
            "compounds": ci["compounds"],
            "adjacencies": ci["adjacencies"],
            "fingerprint": ci["fingerprint"],
            "proteins": pi["proteins"],
            "target": self._targets[index],
        }


def _pad_1d(arrays: List[np.ndarray]):
    max_length = max(array.shape[0] for array in arrays)
    padded = np.zeros((len(arrays), max_length), dtype=np.int64)
    mask = np.zeros((len(arrays), max_length), dtype=np.float32)
    for index, array in enumerate(arrays):
        length = array.shape[0]
        padded[index, :length] = array.astype(np.int64) + 1
        mask[index, :length] = 1.0
    return padded, mask


def _pad_2d(arrays: List[np.ndarray]):
    max_length = max(array.shape[0] for array in arrays)
    # bool_ instead of int64: 8x smaller, and the model only checks adj > 0 / adj.bool()
    padded = np.zeros((len(arrays), max_length, max_length), dtype=np.bool_)
    for index, array in enumerate(arrays):
        length = array.shape[0]
        padded[index, :length, :length] = array.astype(bool)
    return padded


def bacpi_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    compounds, compound_mask = _pad_1d([item["compounds"] for item in batch])
    proteins, protein_mask = _pad_1d([item["proteins"] for item in batch])
    adjacencies = _pad_2d([item["adjacencies"] for item in batch])
    fingerprints = np.stack([item["fingerprint"] for item in batch]).astype(np.float32)
    targets = np.asarray([item["target"] for item in batch], dtype=np.float32).reshape(-1, 1)
    return {
        "compounds": torch.as_tensor(compounds, dtype=torch.long),
        "compound_mask": torch.as_tensor(compound_mask, dtype=torch.float32),
        "adjacencies": torch.as_tensor(adjacencies, dtype=torch.bool),
        "fingerprint": torch.as_tensor(fingerprints, dtype=torch.float32),
        "proteins": torch.as_tensor(proteins, dtype=torch.long),
        "protein_mask": torch.as_tensor(protein_mask, dtype=torch.float32),
        "targets": torch.as_tensor(targets, dtype=torch.float32),
    }


def create_loader(dataset: BACPIDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=bacpi_collate_fn,
    )
