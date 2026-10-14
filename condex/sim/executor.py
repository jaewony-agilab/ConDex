"""Frozen RLDX-1 behind its own ZeroMQ inference server (rldx/eval/run_rldx_server.py).

The server runs in RLDX-1's venv, a separate process that never has condex installed, so the
executor cannot import the recorder; this client must not import condex.privileged either
(tests/test_isolation.py).
"""

from __future__ import annotations

import numpy as np


class RLDXExecutor:
    def __init__(self, host: str, port: int, n_action_steps: int = 16, timeout_ms: int = 120000):
        from rldx.policy.server_client import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms)
        self.n_action_steps = n_action_steps
        self.session_id = ""
        self._reset_memory = True

    def modality_delta_indices(self) -> tuple[np.ndarray, np.ndarray | None]:
        mc = self.client.get_modality_config()
        video = np.array(mc["video"].delta_indices)
        state = np.array(mc["state"].delta_indices) if "state" in mc else None
        return video, state

    def reset(self, session_id: str) -> None:
        self.session_id = session_id
        self._reset_memory = True

    def interrupt(self, reset_memory: bool) -> None:
        """Called after a recovery or replan: the next query starts a fresh chunk; optionally the
        policy's memory is cleared too (its history no longer matches what it did)."""
        self._reset_memory = self._reset_memory or reset_memory

    def act(self, obs_batched: dict) -> list[dict]:
        """One policy query -> per-step actions (the chunk, truncated to n_action_steps)."""
        options = {"reset_memory": [self._reset_memory], "session_ids": [self.session_id]}
        actions, _ = self.client.get_action(obs_batched, options=options)
        self._reset_memory = False
        horizon = min(self.n_action_steps, min(np.asarray(v).shape[1] for v in actions.values()))
        return [{k: np.asarray(v)[0, t] for k, v in actions.items()} for t in range(horizon)]
