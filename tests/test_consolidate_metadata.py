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
