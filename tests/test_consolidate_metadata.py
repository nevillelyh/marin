# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ledger-only shard cache consolidation."""

import os
import tempfile

import numpy as np
import pytest
from levanter.data.text.datasets import TokenSeqDataset
from levanter.store.cache import (
    CACHE_LAYOUT_SHARDED,
    CacheLedger,
    TreeCache,
    _expose_cache_rows,
    _relative_shard_path,
    consolidate_shard_caches,
)
from levanter.store.tree_store import TreeStore

NUM_SHARDS = 4
ROWS_PER_SHARD = 3
ROW_WIDTH = 5
SEQ_LEN = 4

EXEMPLAR_FLAT = {"input_ids": np.array([0], dtype=np.int32)}


def _build_shard_cache(shard_path: str, shard_index: int) -> None:
    store = TreeStore.open(EXEMPLAR_FLAT, shard_path, mode="w", cache_metadata=True)
    rows = [
        {"input_ids": np.arange(ROW_WIDTH, dtype=np.int32) + shard_index * 100 + row_index * 10}
        for row_index in range(ROWS_PER_SHARD)
    ]
    store.extend(rows)
    _expose_cache_rows(shard_path, EXEMPLAR_FLAT, len(rows))
    ledger = CacheLedger(
        total_num_rows=len(rows),
        shard_rows={os.path.basename(shard_path): len(rows)},
        is_finished=True,
        finished_shards=[os.path.basename(shard_path)],
        field_counts={"input_ids": ROWS_PER_SHARD * ROW_WIDTH},
    )
    ledger._serialize_and_commit(shard_path)


def test_consolidate_shard_caches_writes_only_ledger():
    with tempfile.TemporaryDirectory(prefix="levanter-test-consolidate-") as tmpdir:
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)

        ledger = consolidate_shard_caches(shard_paths, tmpdir, EXEMPLAR_FLAT)

        assert ledger.layout == CACHE_LAYOUT_SHARDED
        assert ledger.total_num_rows == NUM_SHARDS * ROWS_PER_SHARD
        assert ledger.field_counts == {"input_ids": NUM_SHARDS * ROWS_PER_SHARD * ROW_WIDTH}
        assert ledger.finished_shards == [os.path.basename(path) for path in shard_paths]

        with pytest.raises(FileNotFoundError):
            TreeStore.open(EXEMPLAR_FLAT, tmpdir, mode="r", cache_metadata=True)


def test_relative_shard_path_keeps_external_local_paths_absolute():
    with tempfile.TemporaryDirectory(prefix="levanter-test-shard-paths-") as tmpdir:
        output_path = os.path.join(tmpdir, "output")
        internal_shard = os.path.join(output_path, "part-00000")
        external_shard = os.path.join(tmpdir, "source", "part-00000")

        assert _relative_shard_path(output_path, internal_shard) == "part-00000"
        assert _relative_shard_path(output_path, external_shard) == external_shard


@pytest.mark.asyncio
async def test_consolidate_external_shards_uses_absolute_paths():
    with tempfile.TemporaryDirectory(prefix="levanter-test-external-shards-") as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        output_path = os.path.join(tmpdir, "output")
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(source_dir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)

        ledger = consolidate_shard_caches(shard_paths, output_path, EXEMPLAR_FLAT)

        assert ledger.finished_shards == shard_paths
        cache = TreeCache.load(output_path, EXEMPLAR_FLAT)
        dataset = TokenSeqDataset(cache, SEQ_LEN)

        batch = await dataset.get_batch([0])

        np.testing.assert_array_equal(batch[0], np.arange(SEQ_LEN, dtype=np.int32))


def test_sharded_cache_rejects_duplicate_shards():
    ledger = CacheLedger(
        total_num_rows=2,
        shard_rows={"part-00000": 1},
        is_finished=True,
        finished_shards=["part-00000", "part-00000"],
        layout=CACHE_LAYOUT_SHARDED,
    )

    with pytest.raises(ValueError, match="duplicate shard"):
        TreeCache("unused", EXEMPLAR_FLAT, ledger)


def test_sharded_cache_requires_row_counts_for_finished_shards():
    ledger = CacheLedger(
        total_num_rows=1,
        shard_rows={},
        is_finished=True,
        finished_shards=["part-00000"],
        layout=CACHE_LAYOUT_SHARDED,
    )

    with pytest.raises(ValueError, match="missing row count"):
        TreeCache("unused", EXEMPLAR_FLAT, ledger)


def test_sharded_cache_rejects_total_row_mismatch():
    ledger = CacheLedger(
        total_num_rows=2,
        shard_rows={"part-00000": 1},
        is_finished=True,
        finished_shards=["part-00000"],
        layout=CACHE_LAYOUT_SHARDED,
    )

    with pytest.raises(ValueError, match="row count mismatch"):
        TreeCache("unused", EXEMPLAR_FLAT, ledger)


@pytest.mark.asyncio
async def test_empty_sharded_token_seq_len_is_zero():
    ledger = CacheLedger(
        total_num_rows=0,
        shard_rows={},
        is_finished=True,
        finished_shards=[],
        layout=CACHE_LAYOUT_SHARDED,
    )
    cache = TreeCache("unused", EXEMPLAR_FLAT, ledger)
    dataset = TokenSeqDataset(cache, SEQ_LEN)

    assert await dataset.async_len() == 0


@pytest.mark.asyncio
async def test_token_seq_dataset_reads_sharded_cache():
    with tempfile.TemporaryDirectory(prefix="levanter-test-sharded-read-") as tmpdir:
        shard_paths = []
        all_tokens = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)
            for row_index in range(ROWS_PER_SHARD):
                all_tokens.extend(np.arange(ROW_WIDTH, dtype=np.int32) + i * 100 + row_index * 10)

        consolidate_shard_caches(shard_paths, tmpdir, EXEMPLAR_FLAT)
        cache = TreeCache.load(tmpdir, EXEMPLAR_FLAT)
        dataset = TokenSeqDataset(cache, SEQ_LEN)

        assert await dataset.async_len() == len(all_tokens) // SEQ_LEN

        batch = await dataset.get_batch([0, 3, 4])

        np.testing.assert_array_equal(batch[0], np.array(all_tokens[0:4], dtype=np.int32))
        np.testing.assert_array_equal(batch[1], np.array(all_tokens[12:16], dtype=np.int32))
        np.testing.assert_array_equal(batch[2], np.array(all_tokens[16:20], dtype=np.int32))


@pytest.mark.asyncio
async def test_tree_cache_get_batch_reads_sharded_rows():
    with tempfile.TemporaryDirectory(prefix="levanter-test-sharded-tree-cache-") as tmpdir:
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)

        consolidate_shard_caches(shard_paths, tmpdir, EXEMPLAR_FLAT)
        cache = TreeCache.load(tmpdir, EXEMPLAR_FLAT)

        batch = await cache.get_batch([0, ROWS_PER_SHARD, NUM_SHARDS * ROWS_PER_SHARD - 1])

        np.testing.assert_array_equal(batch[0]["input_ids"], np.arange(ROW_WIDTH, dtype=np.int32))
        np.testing.assert_array_equal(batch[1]["input_ids"], np.arange(ROW_WIDTH, dtype=np.int32) + 100)
        np.testing.assert_array_equal(
            batch[2]["input_ids"],
            np.arange(ROW_WIDTH, dtype=np.int32) + (NUM_SHARDS - 1) * 100 + (ROWS_PER_SHARD - 1) * 10,
        )


@pytest.mark.asyncio
async def test_tree_cache_get_batch_slice_uses_python_slice_semantics():
    with tempfile.TemporaryDirectory(prefix="levanter-test-sharded-get-batch-slice-") as tmpdir:
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)

        consolidate_shard_caches(shard_paths, tmpdir, EXEMPLAR_FLAT)
        cache = TreeCache.load(tmpdir, EXEMPLAR_FLAT)

        assert await cache.get_batch(slice(0, 0)) == []
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            await cache.get_batch(slice(None, None, 0))


def test_tree_cache_get_batch_sync_slice_uses_python_slice_semantics():
    with tempfile.TemporaryDirectory(prefix="levanter-test-sharded-get-batch-sync-slice-") as tmpdir:
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"part-{i:05d}")
            _build_shard_cache(shard_path, i)
            shard_paths.append(shard_path)

        consolidate_shard_caches(shard_paths, tmpdir, EXEMPLAR_FLAT)
        cache = TreeCache.load(tmpdir, EXEMPLAR_FLAT)

        assert cache.get_batch_sync(slice(0, 0)) == []
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            cache.get_batch_sync(slice(None, None, 0))
