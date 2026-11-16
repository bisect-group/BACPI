import multiprocessing
import pickle
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

from emulator_bench.common import ensure_parent, stable_hash


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


DEFAULT_RADIUS = 2
DEFAULT_NGRAM = 3
DEFAULT_FP_RADIUS = 2
DEFAULT_NBITS = 1024
CACHE_VERSION = 2


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def normalize_smiles_text(smiles: str) -> str:
    return str(smiles).strip()


def parse_smiles(smiles: str):
    """RDKit mol from the SMILES exactly as given (whitespace stripped), or None if invalid."""
    return Chem.MolFromSmiles(normalize_smiles_text(smiles))


def filter_invalid_smiles_rows(frame: pd.DataFrame, smiles_col: str, source_name: str = "", packed: Optional["PackedCache"] = None):
    """Drop rows whose SMILES RDKit cannot parse. SMILES in the packed compound cache are valid, so they are not parsed."""
    cache = {}
    mask = []
    invalid_examples = []
    for smiles in frame[smiles_col].astype(str).tolist():
        if smiles not in cache:
            cache[smiles] = (packed is not None and compound_cache_key(smiles) in packed.index) or parse_smiles(smiles) is not None
        is_valid = cache[smiles]
        mask.append(is_valid)
        if not is_valid and len(invalid_examples) < 5:
            invalid_examples.append(smiles)

    filtered = frame.loc[mask].reset_index(drop=True)
    dropped = int(len(frame) - len(filtered))
    stats = {
        "source": str(source_name),
        "rows_in": int(len(frame)),
        "rows_out": int(len(filtered)),
        "rows_dropped": dropped,
        "invalid_examples": invalid_examples,
    }
    return filtered, stats


def compound_cache_key(smiles: str) -> str:
    settings = "radius=%s|fp_radius=%s|nbits=%s" % (DEFAULT_RADIUS, DEFAULT_FP_RADIUS, DEFAULT_NBITS)
    return stable_hash("%s|%s" % (settings, normalize_smiles_text(smiles)))


def protein_cache_key(sequence: str) -> str:
    return stable_hash("ngram=%s|%s" % (DEFAULT_NGRAM, normalize_sequence(sequence)))


def featurizer_path(embeddings_dir: Path) -> Path:
    return Path(embeddings_dir) / ("vocabs_v%s" % CACHE_VERSION) / "featurizer.pkl"


def packed_dir(embeddings_dir: Path) -> Path:
    return Path(embeddings_dir) / ("packed_v%s" % CACHE_VERSION)


class BACPIFeaturizer:
    def __init__(
        self,
        radius: int = DEFAULT_RADIUS,
        ngram: int = DEFAULT_NGRAM,
        fp_radius: int = DEFAULT_FP_RADIUS,
        nbits: int = DEFAULT_NBITS,
        atom_dict: Optional[Dict] = None,
        bond_dict: Optional[Dict] = None,
        fingerprint_dict: Optional[Dict] = None,
        edge_dict: Optional[Dict] = None,
        word_dict: Optional[Dict] = None,
    ):
        self.radius = int(radius)
        self.ngram = int(ngram)
        self.fp_radius = int(fp_radius)
        self.nbits = int(nbits)
        self.atom_dict = dict(atom_dict or {})
        self.bond_dict = dict(bond_dict or {})
        self.fingerprint_dict = dict(fingerprint_dict or {})
        self.edge_dict = dict(edge_dict or {})
        self.word_dict = dict(word_dict or {})

    def _lookup(self, mapping: Dict, key, build: bool) -> int:
        if key in mapping:
            return int(mapping[key])
        if not build:
            raise KeyError("Feature `%s` is not present in the frozen BACPI vocabulary." % (key,))
        index = len(mapping)
        mapping[key] = index
        return index

    @staticmethod
    def raw_keys(mol):
        """Vocabulary-free description of a molecule: atom keys and bonds as (begin, end, bond type)."""
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for atom in mol.GetAromaticAtoms():
            index = atom.GetIdx()
            atoms[index] = (atoms[index], "aromatic")
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())) for bond in mol.GetBonds()]
        return atoms, bonds

    def _create_ijbonddict(self, n_atoms: int, bonds, build: bool):
        i_jbond_dict = {}
        for begin_idx, end_idx, bond_type in bonds:
            bond_id = self._lookup(self.bond_dict, bond_type, build)
            i_jbond_dict.setdefault(begin_idx, []).append((end_idx, bond_id))
            i_jbond_dict.setdefault(end_idx, []).append((begin_idx, bond_id))

        isolated = set(range(n_atoms)) - set(i_jbond_dict.keys())
        nan_bond = self._lookup(self.bond_dict, "nan", build)
        for atom_idx in isolated:
            i_jbond_dict.setdefault(atom_idx, []).append((atom_idx, nan_bond))
        return i_jbond_dict

    def compound_tokens(self, atom_keys, bonds, build: bool):
        atoms = np.asarray([self._lookup(self.atom_dict, key, build) for key in atom_keys], dtype=np.int32)
        i_jbond_dict = self._create_ijbonddict(len(atom_keys), bonds, build)
        return self._atom_features(atoms, i_jbond_dict, build)

    def _atom_features(self, atoms, i_jbond_dict, build: bool):
        if len(atoms) == 1 or self.radius == 0:
            fingerprints = [self._lookup(self.fingerprint_dict, int(atom), build) for atom in atoms]
            return np.asarray(fingerprints, dtype=np.int32)

        nodes = list(atoms.tolist())
        i_jedge_dict = i_jbond_dict
        for _ in range(self.radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = tuple(sorted((nodes[j], edge) for j, edge in j_edge))
                fingerprint = (nodes[i], neighbors)
                fingerprints.append(self._lookup(self.fingerprint_dict, fingerprint, build))

            nodes = fingerprints
            next_edges = {}
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_sides = tuple(sorted((nodes[i], nodes[j])))
                    edge_key = (both_sides, edge)
                    edge_id = self._lookup(self.edge_dict, edge_key, build)
                    next_edges.setdefault(i, []).append((j, edge_id))
            i_jedge_dict = next_edges
        return np.asarray(nodes, dtype=np.int32)

    def _create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        adjacency = np.asarray(adjacency, dtype=np.uint8)
        adjacency += np.eye(adjacency.shape[0], dtype=np.uint8)
        return adjacency

    def _fingerprint_bits(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.nbits, useChirality=True)
        array = np.zeros((self.nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def _split_sequence(self, sequence: str, build: bool):
        normalized = "-" + normalize_sequence(sequence) + "="
        words = []
        for index in range(len(normalized) - self.ngram + 1):
            token = normalized[index : index + self.ngram]
            words.append(self._lookup(self.word_dict, token, build))
        return np.asarray(words, dtype=np.int32)

    def encode_compound(self, smiles: str, build: bool = False) -> Dict:
        mol = parse_smiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES: %s" % normalize_smiles_text(smiles))
        mol = Chem.AddHs(mol)
        tokens = self.compound_tokens(*self.raw_keys(mol), build=build)
        return {"compounds": tokens, "adjacencies": self._create_adjacency(mol), "fingerprint": self._fingerprint_bits(mol)}

    def encode_protein(self, sequence: str, build: bool = False) -> np.ndarray:
        return self._split_sequence(sequence, build=build)

    @property
    def n_atom_tokens(self) -> int:
        return len(self.fingerprint_dict)

    @property
    def n_amino_tokens(self) -> int:
        return len(self.word_dict)

    def to_payload(self) -> Dict:
        return {
            "radius": self.radius,
            "ngram": self.ngram,
            "fp_radius": self.fp_radius,
            "nbits": self.nbits,
            "atom_dict": self.atom_dict,
            "bond_dict": self.bond_dict,
            "fingerprint_dict": self.fingerprint_dict,
            "edge_dict": self.edge_dict,
            "word_dict": self.word_dict,
        }

    def save(self, path: Path) -> None:
        ensure_parent(path)
        with open(path, "wb") as handle:
            pickle.dump(self.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(**payload)


_worker_featurizer = None
_worker_needed = frozenset()


def _raw_compounds(chunk):
    """Raw keys of each SMILES, plus adjacency and fingerprint for the needed ones. None marks an invalid SMILES."""
    out = []
    for smiles in chunk:
        mol = parse_smiles(smiles)
        if mol is None:
            out.append(None)
            continue
        mol = Chem.AddHs(mol)
        keys = BACPIFeaturizer.raw_keys(mol)
        if smiles in _worker_needed:
            out.append((keys, _worker_featurizer._create_adjacency(mol), _worker_featurizer._fingerprint_bits(mol)))
        else:
            out.append((keys, None, None))
    return out


def _new_ngrams(chunk):
    """Distinct n-grams of each sequence in first-occurrence order."""
    ngram = _worker_featurizer.ngram
    padded = ["-%s=" % sequence for sequence in chunk]
    return [tuple(dict.fromkeys(text[i : i + ngram] for i in range(len(text) - ngram + 1))) for text in padded]


def encode_compounds_chunk(chunk):
    return [_worker_featurizer.encode_compound(smiles) if parse_smiles(smiles) is not None else None for smiles in chunk]


def encode_proteins_chunk(chunk):
    return [_worker_featurizer.encode_protein(sequence) for sequence in chunk]


def parallel_map(function, items, featurizer, workers: int, desc: str, needed=frozenset(), chunk_size: int = 64):
    """Yield function(chunk) results per item in input order from forked workers that share the featurizer."""
    global _worker_featurizer, _worker_needed
    _worker_featurizer, _worker_needed = featurizer, needed
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for part in progress(pool.imap(function, chunks), desc=desc, unit="chunk", total=len(chunks)):
            yield from part


def compound_entry(encoded: Dict) -> tuple:
    return encoded["compounds"], encoded["adjacencies"], encoded["fingerprint"]


def build_vocabulary(featurizer: BACPIFeaturizer, smiles_values, sequence_values, needed_smiles, workers: int) -> Dict[str, tuple]:
    """Assign vocabulary ids exactly as a serial pass over the sorted unique inputs.

    Workers compute the id-free RDKit keys. This process assigns ids in order.
    Returns the packed entries of `needed_smiles` that are valid SMILES in the vocabulary inputs.
    """
    unique_smiles = sorted({normalize_smiles_text(value) for value in smiles_values})
    needed = frozenset(needed_smiles)
    entries = {}
    for smiles, raw in zip(unique_smiles, parallel_map(_raw_compounds, unique_smiles, featurizer, workers, "Building compound vocabulary", needed)):
        if raw is not None:
            keys, adjacency, fingerprint = raw
            tokens = featurizer.compound_tokens(*keys, build=True)
            if smiles in needed:
                entries[smiles] = (tokens, adjacency, fingerprint)
    unique_sequences = sorted({normalize_sequence(value) for value in sequence_values})
    for ngrams in parallel_map(_new_ngrams, unique_sequences, featurizer, workers, "Building protein vocabulary"):
        for token in ngrams:
            featurizer._lookup(featurizer.word_dict, token, True)
    return entries


def write_packed(embeddings_dir: Path, new: Dict[str, Dict[str, tuple]]) -> None:
    """Write the packed store: the old entries plus `new` ({kind: {key: arrays}}), sorted by key.

    Compounds hold tokens (int32), adjacencies (uint8, n x n) and fingerprints (uint8). Proteins hold tokens.
    The new store is built beside the old one and swapped in by rename.
    """
    final = packed_dir(embeddings_dir)
    tmp, previous = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for kind, names in (("compounds", ("tokens", "adj", "fp")), ("proteins", ("tokens",))):
        old = PackedCache(embeddings_dir, kind) if (final / ("%s.index.npz" % kind)).exists() else None
        fresh = new[kind]
        keys = sorted(set(old.index if old else ()) | set(fresh))
        offsets = {name: [0] for name in names[:2]}
        with ExitStack() as stack:
            handles = [stack.enter_context(open(tmp / ("%s.%s.bin" % (kind, name)), "wb")) for name in names]
            for key in progress(keys, desc="Writing %s" % kind, unit="entry"):
                if key in fresh:
                    arrays = fresh[key]
                elif kind == "compounds":
                    arrays = compound_entry(old.compound(old.index[key]))
                else:
                    arrays = (old.tokens(old.index[key]),)
                for name, array, handle in zip(names, arrays, handles):
                    handle.write(np.ascontiguousarray(array).tobytes())
                    if name in offsets:
                        offsets[name].append(offsets[name][-1] + array.size)
        np.savez(
            tmp / ("%s.index.npz" % kind),
            keys=np.asarray(keys),
            **{"%s_offsets" % name: np.asarray(values, dtype=np.int64) for name, values in offsets.items()},
        )
    shutil.rmtree(previous, ignore_errors=True)
    if final.exists():
        final.rename(previous)
    tmp.rename(final)
    shutil.rmtree(previous, ignore_errors=True)


class PackedCache:
    """Read-only view of the packed cache of one kind. Entries come from `write_packed`."""

    def __init__(self, embeddings_dir: Path, kind: str):
        directory = packed_dir(embeddings_dir)
        if not (directory / ("%s.index.npz" % kind)).exists():
            raise FileNotFoundError("No packed %s cache in %s. Run emulator_bench/cache_embeddings.py." % (kind, directory))
        with np.load(directory / ("%s.index.npz" % kind)) as payload:
            self.index = {key: row for row, key in enumerate(payload["keys"].tolist())}
            self._token_offsets = payload["tokens_offsets"]
            self._adj_offsets = payload["adj_offsets"] if "adj_offsets" in payload.files else None
        self._tokens = np.memmap(directory / ("%s.tokens.bin" % kind), dtype=np.int32, mode="r")
        if self._adj_offsets is not None:
            self._adj = np.memmap(directory / ("%s.adj.bin" % kind), dtype=np.uint8, mode="r")
            self._fp = np.memmap(directory / ("%s.fp.bin" % kind), dtype=np.uint8, mode="r").reshape(-1, DEFAULT_NBITS)

    def rows(self, values, key_fn) -> list:
        unique = {value: self.index[key_fn(value)] for value in set(values)}
        return [unique[value] for value in values]

    def tokens(self, row: int) -> np.ndarray:
        return self._tokens[self._token_offsets[row] : self._token_offsets[row + 1]]

    def compound(self, row: int) -> Dict:
        tokens = self.tokens(row)
        size = tokens.shape[0]
        start = self._adj_offsets[row]
        return {
            "compounds": tokens,
            "adjacencies": self._adj[start : start + size * size].reshape(size, size),
            "fingerprint": self._fp[row],
        }
