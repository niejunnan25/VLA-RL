from __future__ import annotations

import logging
from typing import Any, Mapping

from .controller import ManualEpisodeController


class CriticalPhaseController(ManualEpisodeController):
    """Manual episode controller with one-way VLA -> RL handover."""

    def __init__(
        self,
        *,
        enabled: bool,
        interface: str = "terminal",
        poll_interval_sec: float = 0.05,
        terminal_grace_sec: float = 0.15,
        keys: Mapping[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        key_cfg = dict(keys or {})
        self._critical_key = str(key_cfg.get("critical", "c")).strip().lower()
        self._critical_phase_active = False
        super().__init__(
            enabled=enabled,
            interface=interface,
            poll_interval_sec=poll_interval_sec,
            terminal_grace_sec=terminal_grace_sec,
            keys=key_cfg,
            logger=logger,
        )

    @property
    def is_critical_phase(self) -> bool:
        with self._lock:
            return bool(self._critical_phase_active)

    def request_critical(self) -> dict[str, Any]:
        with self._lock:
            self._critical_phase_active = True
        return self.get_meta()

    def start_episode(self) -> None:
        with self._lock:
            self._critical_phase_active = False
        super().start_episode()

    def get_meta(self) -> dict[str, Any]:
        meta = super().get_meta()
        with self._lock:
            meta["critical_phase_active"] = bool(self._critical_phase_active)
        return meta

    def _dispatch_key(self, ch: str) -> None:
        if str(ch).strip().lower() == self._critical_key:
            self.request_critical()
            self.logger.info("Controller event: critical handover")
            return
        super()._dispatch_key(ch)

    def _log_help(self) -> None:
        super()._log_help()
        self.logger.info("Critical phase key: critical=%s", self._critical_key)
