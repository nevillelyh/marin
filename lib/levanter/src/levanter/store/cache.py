# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging as pylogging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import deepdiff
import jax
import jax.tree_util as jtu
from rigging.filesystem import open_url, url_to_fs
import numpy as np
import pyarrow as pa
from dataclasses_json import dataclass_json
from fray import ResourceConfig
from fsspec import AbstractFileSystem
from tqdm_loggable.tqdm_logging import tqdm_logging
from zephyr import Dataset, ZephyrContext
from zephyr import counters as zephyr_counters
from zephyr.writers import ThreadedBatchWriter, atomic_rename, batchify, ensure_parent_dir

from levanter.data.dataset import AsyncDataset
from levanter.utils.jax_utils import broadcast_one_to_all

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.sharded_datasource import ShardedDataSource
from .jagged_array import _no_cache_read_context
from .tree_store import TreeStore

T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"
CONSOLIDATE_DATA_SIZE_WORKERS = 32
CACHE_LAYOUT_CONSOLIDATED = "consolidated"
CACHE_LAYOUT_SHARDED = "sharded"

DEFAULT_LOG_LEVEL = pylogging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


@dataclass(frozen=True)
class CacheOptions:
    batch_size: int = 128

    @staticmethod
    def default():
        return CacheOptions()


def build_or_load_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions = CacheOptions.default(),
) -> "TreeCache[U]":
    """
    Build or load a TreeCache from a sharded data source using a Zephyr backend.
    """
    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    try:
        return TreeCache.load(cache_dir, processor.output_exemplar, metadata)
    except FileNotFoundError:
        logger.info(f"Cache not found at {cache_dir}. Building with zephyr pipeline.")

    # Distributed coordination: only process 0 builds; others wait and then load.
    if jax.distributed.is_initialized() and jax.process_count() > 1:
        _distributed_build_cache(cache_dir, source, processor, options, metadata, is_leader=jax.process_index() == 0)
    else:
        build_cache(cache_dir, source, processor, options, metadata)

    return TreeCache.load(cache_dir, processor.output_exemplar, metadata)


class TreeCache(AsyncDataset[T_co]):
    ledger: "CacheLedger"

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: "CacheLedger",
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._exemplar = exemplar

        if not ledger.is_finished:
            raise RuntimeError(f"Cache at {cache_dir} is not finished.")

        self._store = None
        if not self.is_sharded:
            self._store = TreeStore.open(self._exemplar, self.cache_dir, mode="r", cache_metadata=False)

    @property
    def store(self) -> TreeStore[T_co]:
        if self._store is None:
            raise RuntimeError(f"Cache at {self.cache_dir} is sharded and has no top-level TreeStore.")
        return self._store

    @property
    def is_sharded(self) -> bool:
        return self.ledger.layout == CACHE_LAYOUT_SHARDED

    def shard_path(self, shard_name: str) -> str:
        if "://" in shard_name:
            return shard_name
        return os.path.join(self.cache_dir, shard_name)

    async def async_len(self) -> int:
        if self.is_sharded:
            return self.ledger.total_num_rows
        return len(self.store)

    def __len__(self):
        if self.is_sharded:
            return self.ledger.total_num_rows
        return len(self.store)

    def is_finite(self) -> bool:
        return True

    def __getitem__(self, item):
        return self.store[item]

    async def get_batch(self, indices: Sequence[int] | slice):
        if isinstance(indices, slice):
            indices = range(indices.start or 0, indices.stop or len(self), indices.step or 1)
        return await self.store.get_batch(indices)

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        if isinstance(indices_or_slice, slice):
            indices_or_slice = range(
                indices_or_slice.start or 0,
                indices_or_slice.stop or len(self),
                indices_or_slice.step or 1,
            )
        return self.store.get_batch_sync(indices_or_slice)

    @staticmethod
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        logger.info(f"Loading cache from {cache_dir}")
        ledger = CacheLedger.load(cache_dir, options)

        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: Optional["CacheOptions"] = None,
    ) -> "TreeCache[U]":
        if options is None:
            options = CacheOptions.default()
        return build_or_load_cache(cache_dir, shard_source, processor, options=options)

    @property
    def is_finished(self):
        return True


@dataclass_json
@dataclass
class CacheLedger:
    total_num_rows: int
    shard_rows: Dict[str, int]
    is_finished: bool = False
    finished_shards: List[str] = dataclasses.field(default_factory=list)
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    field_counts_by_shard: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)
    layout: str = CACHE_LAYOUT_CONSOLIDATED
    metadata: "CacheMetadata" = dataclasses.field(default_factory=lambda: CacheMetadata({}))

    @staticmethod
    def load_or_initialize(cache_dir: str, source: ShardedDataSource, processor: BatchProcessor):
        metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
        try:
            return CacheLedger.load(cache_dir, metadata)
        except FileNotFoundError:
            return CacheLedger(
                total_num_rows=0,
                shard_rows={shard: 0 for shard in source.shard_names},
                is_finished=False,
                metadata=metadata,
            )

    @staticmethod
    def load(cache_dir: str, metadata: Optional["CacheMetadata"] = None) -> "CacheLedger":
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        try:
            logger.info(f"Attempting to load cache ledger from {ledger_path}")
            with open_url(ledger_path) as file:
                cache_ledger = CacheLedger.from_json(file.read())  # type: ignore[arg-type]
            if metadata:
                diff = cache_ledger.metadata.compare_to(metadata)
                if diff:
                    logger.warning(f"Metadata mismatch: {diff}")
            return cache_ledger
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Cache ledger not found at {ledger_path}") from exc

    def _serialize_and_commit(self, cache_dir):
        path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        return _serialize_json_and_commit(path, self)  # type: ignore[arg-type]


@dataclass_json
@dataclass(frozen=True)
class CacheMetadata:
    preprocessor_metadata: Optional[dict[str, Any]] = None

    def compare_to(self, other: "CacheMetadata") -> deepdiff.DeepDiff:
        if other.preprocessor_metadata is None:
            sorta_self = dataclasses.replace(self, preprocessor_metadata=None)
        else:
            sorta_self = self
        return deepdiff.DeepDiff(sorta_self, other)

    @staticmethod
    def empty():
        return CacheMetadata()


class SerialCacheWriter:
    """
    Writes TreeCache-compatible caches to disk directly. Mostly for scripts and debugging.
    """

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        metadata: Optional["CacheMetadata"] = None,
        shard_name: str = "",
        mode: str = "w",
    ):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._exemplar = exemplar
        self._shard_name = shard_name
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode=mode, cache_metadata=True)
        self._is_closed = False

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ledger = CacheLedger(
            total_num_rows=len(self._tree_store),
            is_finished=True,
            shard_rows={self._shard_name: len(self._tree_store)},
            finished_shards=[self._shard_name],
            field_counts=_field_counts_from_store(self._tree_store),
            metadata=self.metadata or CacheMetadata.empty(),
        )

        if exc_type is None:
            ledger._serialize_and_commit(self.cache_dir)
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self) -> "TreeCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until TreeCacheWriter is closed")
        return TreeCache.load(self.cache_dir, self._exemplar, self.metadata)

    def write_batch(self, batch: BatchResult):
        if isinstance(batch, pa.RecordBatch):
            batch = dict_from_record_batch(batch)

        cbatch = _canonicalize_batch(batch)  # type: ignore[arg-type]
        self._tree_store.extend(cbatch)


# Fixed batch size for Levanter cache writes (2^14).
_LEVANTER_BATCH_SIZE = 16384


def write_levanter_cache(
    records: Iterable[dict[str, Any]],
    output_path: str,
    *,
    metadata: dict[str, Any],
    batch_size: int = _LEVANTER_BATCH_SIZE,
) -> dict:
    """Write tokenized records to Levanter cache format.

    Args:
        records: Tokenized records (iterable of dicts with array values)
        output_path: Path to output cache directory
        metadata: Metadata for the cache
        batch_size: Number of records to accumulate before flushing to disk.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    ensure_parent_dir(output_path)
    record_iter = iter(records)

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 0
    logger.info("write_levanter_cache: starting write to %s (batch_size=%d)", output_path, batch_size)

    with atomic_rename(output_path) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata)) as writer:

            def _drain_batches(batches: Iterable) -> None:
                for batch in batches:
                    writer.write_batch(batch)

            with ThreadedBatchWriter(_drain_batches) as threaded:
                threaded.submit([exemplar])
                count += 1
                zephyr_counters.increment("zephyr/records_out")
                for batch in batchify(record_iter, n=batch_size):
                    threaded.submit(batch)
                    count += len(batch)
                    zephyr_counters.increment("zephyr/records_out", len(batch))
                    logger.info("write_levanter_cache: %s — %d records so far", output_path, count)

    logger.info("write_levanter_cache: finished %s — %d records", output_path, count)

    # write success sentinel
    with open_url(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def _serialize_json_and_commit(path: str, obj):
    fs: AbstractFileSystem = url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        fs.copy(path, f"{path}.bak")

    for _ in range(10):
        try:
            with open_url(path, "w") as file:
                file.write(obj.to_json())
            break
        except FileNotFoundError:
            logger.exception(f"Failed to write {path}")


def build_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions,
    metadata: CacheMetadata,
) -> CacheLedger:
    """
    Build a cache from a sharded data source using a Zephyr backend.
    """
    shard_names = list(source.shard_names)

    if len(shard_names) == 0:
        logger.info("No shards to process. Writing empty cache.")
        TreeStore.open(processor.output_exemplar, cache_dir, mode="w", cache_metadata=True)
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            finished_shards=[],
            field_counts={},
            metadata=metadata,
        )
        ledger._serialize_and_commit(cache_dir)
        return ledger

    temp_root = cache_dir
    shard_jobs = [{"shard_name": name, "index": idx} for idx, name in enumerate(shard_names)]

    def process_shard(job: dict):
        return _build_single_shard_cache(
            shard_name=job["shard_name"],
            shard_index=job["index"],
            temp_root=temp_root,
            source=source,
            processor=processor,
            options=options,
            metadata=metadata,
        )

    ctx = ZephyrContext(
        resources=ResourceConfig(ram="32g", disk="16g"),
        max_workers=min(128, len(shard_jobs)),
        name="levanter-cache-build",
    )
    shard_results = ctx.execute(Dataset.from_list(shard_jobs).map(process_shard), verbose=False).results
    shard_results = sorted(shard_results, key=lambda r: r["index"])

    shard_cache_paths = [s["path"] for s in shard_results]
    ledger = consolidate_shard_caches(
        shard_cache_paths=shard_cache_paths,
        output_path=cache_dir,
        exemplar=processor.output_exemplar,
        metadata=metadata,
    )
    return ledger


def _build_single_shard_cache(
    shard_name: str,
    shard_index: int,
    temp_root: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    options: CacheOptions,
    metadata: CacheMetadata,
):
    shard_path = os.path.join(temp_root, f"{shard_index:05d}_{_sanitize_shard_name(shard_name)}")
    existing = _try_load(shard_path, metadata)
    if existing is not None:
        logger.info(f"Found existing shard cache for {shard_name} at {shard_path}. Skipping build.")
        return {"shard_name": shard_name, "path": shard_path, "ledger": existing, "index": shard_index}

    logger.info(f"Building shard {shard_name} -> {shard_path}")

    def records():
        batch = []
        pbar = tqdm_logging(desc=f"Shard {shard_name}")
        for example in source.open_shard_at_row(shard_name, 0):
            batch.append(example)
            if len(batch) >= options.batch_size:
                processed = processor(batch)
                yield from _canonicalize_batch(processed)
                batch.clear()
            pbar.update(1)
        if batch:
            processed = processor(batch)
            yield from _canonicalize_batch(processed)

    result = write_levanter_cache(records(), shard_path, metadata=metadata.preprocessor_metadata or {})

    if result.get("count", 0) == 0:
        TreeStore.open(processor.output_exemplar, shard_path, mode="w", cache_metadata=True)
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={shard_name: 0},
            is_finished=True,
            finished_shards=[shard_name],
            field_counts={},
            metadata=metadata,
        )
        ledger._serialize_and_commit(shard_path)
    else:
        ledger = CacheLedger.load(shard_path, metadata)

    return {"shard_name": shard_name, "path": shard_path, "ledger": ledger, "index": shard_index}


def consolidate_shard_caches(
    shard_cache_paths: list[str],
    output_path: str,
    exemplar,
    metadata: CacheMetadata | None = None,
) -> CacheLedger:
    """
    Consolidate multiple shard cache ledgers into a single sharded cache ledger.

    Args:
        shard_cache_paths: List of shard cache directories.
        output_path: Destination cache directory.
        exemplar: Output exemplar structure.
        metadata: CacheMetadata to use for the final ledger.
    """
    if metadata is None:
        metadata = CacheMetadata.empty()

    if not shard_cache_paths:
        ledger = CacheLedger(
            total_num_rows=0,
            shard_rows={},
            is_finished=True,
            finished_shards=[],
            field_counts={},
            field_counts_by_shard={},
            layout=CACHE_LAYOUT_SHARDED,
            metadata=metadata,
        )
        ledger._serialize_and_commit(output_path)
        return ledger

    logger.info(f"Consolidating {len(shard_cache_paths)} shard cache ledgers into {output_path}")

    # Distributed: load ledger + read data_size for each shard in parallel.
    # Both operations are S3 I/O-bound; distributing across zephyr workers
    # avoids serializing thousands of S3 calls in the coordinator process.
    def _probe_shard(shard_path):
        ledger = CacheLedger.load(shard_path, metadata)
        if ledger.field_counts:
            field_counts = ledger.field_counts
        else:
            with _no_cache_read_context():
                store = TreeStore.open(exemplar, shard_path, mode="r", cache_metadata=True)
                field_counts = _field_counts_from_store(store)
        return (field_counts, ledger)

    probe_ctx = ZephyrContext(
        resources=ResourceConfig(ram="5g", cpu=2),
        max_workers=min(CONSOLIDATE_DATA_SIZE_WORKERS, len(shard_cache_paths)),
        name="levanter-cache-probe",
    )
    probe_results = probe_ctx.execute(
        Dataset.from_list(shard_cache_paths).map(_probe_shard),
    ).results
    per_shard_field_counts = [r[0] for r in probe_results]
    shard_ledgers = [r[1] for r in probe_results]

    return _merge_ledgers(output_path, shard_cache_paths, shard_ledgers, per_shard_field_counts, metadata)


def _merge_ledgers(
    output_path: str,
    shard_cache_paths: list[str],
    shard_ledgers: list[CacheLedger],
    per_shard_field_counts: list[dict[str, int]],
    metadata: CacheMetadata,
) -> CacheLedger:
    final_ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        finished_shards=[],
        field_counts={},
        field_counts_by_shard={},
        layout=CACHE_LAYOUT_SHARDED,
        metadata=metadata,
    )
    for shard_path, ledger, field_counts in zip(shard_cache_paths, shard_ledgers, per_shard_field_counts, strict=True):
        shard_name = _relative_shard_path(output_path, shard_path)
        final_ledger.shard_rows[shard_name] = ledger.total_num_rows
        final_ledger.finished_shards.append(shard_name)
        final_ledger.total_num_rows += ledger.total_num_rows
        final_ledger.field_counts_by_shard[shard_name] = field_counts
        for field, count in field_counts.items():
            final_ledger.field_counts[field] = final_ledger.field_counts.get(field, 0) + count

    final_ledger.is_finished = True
    final_ledger._serialize_and_commit(output_path)
    return final_ledger


def _distributed_build_cache(
    cache_dir: str,
    source: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    options: CacheOptions,
    metadata: CacheMetadata,
    is_leader: bool,
) -> CacheLedger:
    status = {"val": np.array(0, dtype=np.int32)}
    lock = threading.Lock()

    def broadcaster():
        while True:
            with lock:
                received = broadcast_one_to_all(status["val"], is_source=is_leader)
                status["val"] = received
                if received != 0:
                    break

            time.sleep(10)

        return status["val"]

    if is_leader:
        b_thread = threading.Thread(target=broadcaster, daemon=True)
        b_thread.start()

        try:
            ledger = build_cache(cache_dir, source, processor, options, metadata)
            with lock:
                status["val"] = np.array(1, dtype=np.int32)
        except Exception:
            with lock:
                status["val"] = np.array(-1, dtype=np.int32)
            raise
        finally:
            b_thread.join()
        return ledger
    else:
        status_out = broadcaster()
        if status_out == 1:
            return CacheLedger.load(cache_dir, metadata)
        elif status_out == -1:
            raise RuntimeError("Cache build failed on leader process.")
        else:
            raise RuntimeError("Unexpected status received during distributed cache build.")


def _expose_cache_rows(cache_path: str, exemplar: T, num_rows: int) -> None:
    cache = TreeStore.open(exemplar, cache_path, mode="a", cache_metadata=False)
    futures = jax.tree.leaves(jax.tree.map(lambda x: x.offsets[0].write(num_rows), cache.tree))
    for future in futures:
        future.result()


def _relative_shard_path(output_path: str, shard_path: str) -> str:
    if "://" in shard_path:
        prefix = output_path.rstrip("/") + "/"
        if shard_path.startswith(prefix):
            return shard_path[len(prefix) :]
        return shard_path
    try:
        return os.path.relpath(shard_path, output_path)
    except ValueError:
        return shard_path


def _field_counts_from_store(store: TreeStore) -> dict[str, int]:
    return _field_counts_from_data_sizes(jax.tree.map(lambda array: array.data_size, store.tree))


def _field_counts_from_data_sizes(data_sizes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path, value in jtu.tree_leaves_with_path(data_sizes):
        field = "/".join(_render_path_elem(part) for part in path)
        counts[field] = int(value)
    return counts


def _render_path_elem(path_elem) -> str:
    match path_elem:
        case jtu.DictKey(key):
            return str(key)
        case jtu.GetAttrKey(key):
            return str(key)
        case jtu.SequenceKey(i):
            return str(i)
        case jtu.FlattenedIndexKey(i):
            return str(i)
        case _:
            return str(path_elem)


def _sanitize_shard_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return safe or "shard"


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        keys = list(batch.keys())
        values = list(batch.values())
        num_rows = len(values[0]) if values else 0
        return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]
    else:
        return list(batch)


def _try_load(path, metadata):
    try:
        ledger = CacheLedger.load(path, metadata)
        if ledger.is_finished:
            return ledger
        logger.debug(f"Cache exists but is not finished at {path}.")
        return None
    except FileNotFoundError:
        return None


__all__ = [
    "TreeCache",
    "build_or_load_cache",
    "SerialCacheWriter",
    "CacheLedger",
    "CacheMetadata",
    "CacheOptions",
    "consolidate_shard_caches",
    "write_levanter_cache",
]
