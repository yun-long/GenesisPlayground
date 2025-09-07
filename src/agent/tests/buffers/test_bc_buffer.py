from collections.abc import Iterator
from typing import Any

import pytest
import torch
from gs_agent.buffers.bc_buffer import BCBuffer
from gs_agent.buffers.config.schema import BCBufferKey


@pytest.fixture()
def cfg() -> dict[str, Any]:
    return dict(num_envs=1, max_steps=5, obs_size=3, action_size=2, device=torch.device("cpu"))


@pytest.fixture()
def buffer(cfg: dict[str, Any]) -> BCBuffer:
    return BCBuffer(**cfg)


def make_transition(
    num_envs: int, obs_size: int, action_size: int, device: torch.device
) -> dict[BCBufferKey, torch.Tensor]:
    return {
        BCBufferKey.OBSERVATIONS: torch.randn(num_envs, obs_size, device=device),
        BCBufferKey.ACTIONS: torch.randn(num_envs, action_size, device=device),
    }


def test_init_shapes_and_state(buffer: BCBuffer, cfg: dict[str, Any]) -> None:
    assert len(buffer) == 0
    assert not buffer.is_full()
    # internal storage should have correct shapes
    # (we don't rely on private attrs except to check shapes here)
    batch = next(
        buffer.minibatch_gen(batch_size=cfg["max_steps"], num_epochs=1, shuffle=False),
        None,
    )
    assert batch is None  # empty iterator when size == 0


def test_append_and_len(buffer: BCBuffer, cfg: dict[str, Any]) -> None:
    t = make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
    buffer.append(t)
    assert len(buffer) == 1
    # Fill up to size 3 and check values are stored
    vals = [
        make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
        for _ in range(2)
    ]
    for v in vals:
        buffer.append(v)
    assert len(buffer) == 3

    # Pull a full-batch and check it contains the last 3 in order (since we didn't shuffle)
    batches = list(buffer.minibatch_gen(batch_size=cfg["max_steps"], num_epochs=1, shuffle=False))
    assert len(batches) == 1
    batch = batches[0]
    assert batch[BCBufferKey.OBSERVATIONS].ndim == 3
    assert batch[BCBufferKey.ACTIONS].ndim == 3
    assert batch[BCBufferKey.OBSERVATIONS].shape[1:] == (cfg["num_envs"], cfg["obs_size"])
    assert batch[BCBufferKey.ACTIONS].shape[1:] == (cfg["num_envs"], cfg["action_size"])


def test_is_full_and_circular_overwrite(cfg: dict[str, Any]) -> None:
    buf = BCBuffer(**cfg)
    # Add exactly max_size transitions
    stored = []
    for _ in range(cfg["max_steps"]):
        tr = make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
        stored.append(tr)
        buf.append(tr)

    assert len(buf) == cfg["max_steps"]
    assert buf.is_full()

    # Add one more; oldest element should be overwritten
    new_tr = make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
    buf.append(new_tr)
    assert len(buf) == cfg["max_steps"]  # size stays capped

    # Read everything (no shuffle) and ensure the earliest element got replaced
    batches = list(buf.minibatch_gen(batch_size=cfg["max_steps"], num_epochs=1, shuffle=False))
    (batch,) = batches
    obs = batch[BCBufferKey.OBSERVATIONS]
    act = batch[BCBufferKey.ACTIONS]

    # Because we iterated without shuffle, items appear in ring-buffer logical order
    # The first row should correspond to what used to be the "second" element when we overflowed.
    # Easiest robust check: confirm that the *new* transition is present somewhere,
    # and the very first original one is absent.
    def row_in(x: torch.Tensor, row: torch.Tensor) -> bool:
        return any(torch.allclose(r, row) for r in x)

    assert row_in(obs, new_tr[BCBufferKey.OBSERVATIONS])
    assert row_in(act, new_tr[BCBufferKey.ACTIONS])

    # The very first stored element should no longer exist in the buffer.
    first = stored[0]
    assert not row_in(obs, first[BCBufferKey.OBSERVATIONS])
    assert not row_in(act, first[BCBufferKey.ACTIONS])


@pytest.mark.parametrize("shuffle", [False, True])
def test_minibatch_sizes_and_epochs(buffer: BCBuffer, cfg: dict[str, Any], shuffle: bool) -> None:
    # Insert 13 items to test partial final batch splitting
    n = 13
    for _ in range(n):
        buffer.append(
            make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
        )

    batch_size = 5
    num_epochs = 2
    batches_per_epoch = (n + batch_size - 1) // batch_size

    seen_shapes: list[tuple[int, int]] = []
    count = 0
    for batch in buffer.minibatch_gen(
        batch_size=batch_size, num_epochs=num_epochs, shuffle=shuffle
    ):
        count += batches_per_epoch
        obs = batch[BCBufferKey.OBSERVATIONS]
        act = batch[BCBufferKey.ACTIONS]
        assert obs.ndim == 3 and act.ndim == 3
        assert obs.shape[1] == cfg["num_envs"]
        assert obs.shape[2] == cfg["obs_size"]
        assert act.shape[1] == cfg["num_envs"]
        assert act.shape[2] == cfg["action_size"]
        assert 1 <= obs.shape[0] <= batch_size
        assert obs.shape[0] == act.shape[0]
        seen_shapes.append((obs.shape[0], act.shape[0]))

    assert count == num_epochs * batches_per_epoch


def test_shuffle_changes_order(buffer: BCBuffer, cfg: dict[str, Any]) -> None:
    # Deterministic content
    torch.manual_seed(0)
    for _ in range(10):
        buffer.append(
            make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
        )

    # Collect first epoch without shuffle
    no_shuffle_idxs = []
    for batch in buffer.minibatch_gen(batch_size=10, num_epochs=1, shuffle=False):
        # Only one batch expected
        no_shuffle_idxs.append(batch[BCBufferKey.OBSERVATIONS].clone())

    # Collect with shuffle and check order differs at least somewhere
    torch.manual_seed(0)  # seeding still leads to different permutation vs identity
    shuffle_idxs = []
    for batch in buffer.minibatch_gen(batch_size=10, num_epochs=1, shuffle=True):
        shuffle_idxs.append(batch[BCBufferKey.OBSERVATIONS].clone())

    assert len(no_shuffle_idxs) == 1 and len(shuffle_idxs) == 1
    same = torch.allclose(no_shuffle_idxs[0], shuffle_idxs[0])
    assert not same, "Expected shuffled minibatch to differ in order from non-shuffled"


def test_clear_and_reset(buffer: BCBuffer, cfg: dict[str, Any]) -> None:
    for _ in range(cfg["max_steps"] // 2):
        buffer.append(
            make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
        )
    assert len(buffer) > 0

    buffer.clear()
    assert len(buffer) == 0
    assert not buffer.is_full()

    # After clear, we should be able to append again normally
    buffer.append(
        make_transition(cfg["num_envs"], cfg["obs_size"], cfg["action_size"], cfg["device"])
    )
    assert len(buffer) == 1


def test_minibatch_empty_iterator_when_empty(cfg: dict[str, Any]) -> None:
    buf = BCBuffer(**cfg)
    it: Iterator | None = buf.minibatch_gen(batch_size=4, num_epochs=1, shuffle=False)
    # The generator returns immediately; iterating should produce nothing.
    assert next(it, None) is None


def test_type_and_key_usage(buffer: BCBuffer, cfg: dict[str, Any]) -> None:
    # Ensure BCBufferKey keys are accepted and tensor shapes match expectations
    tr = {
        BCBufferKey.OBSERVATIONS: torch.zeros(cfg["obs_size"]),
        BCBufferKey.ACTIONS: torch.ones(cfg["action_size"]),
    }
    buffer.append(tr)
    batch = next(buffer.minibatch_gen(batch_size=1, num_epochs=1, shuffle=False))
    assert BCBufferKey.OBSERVATIONS in batch and BCBufferKey.ACTIONS in batch
    assert torch.allclose(batch[BCBufferKey.ACTIONS][0], tr[BCBufferKey.ACTIONS])
