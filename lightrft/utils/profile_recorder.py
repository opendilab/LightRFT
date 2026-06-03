"""
Step-level profiling utilities for long-running training jobs.

This module provides a lightweight profiler that is suitable for production
training loops:

- Measures named sections in wall-clock seconds.
- Persists per-step summaries to JSONL with flush + fsync.
- Maintains a continuously refreshed latest snapshot so interrupted jobs still
  leave readable profiling state on disk.
- Optionally emits sampled ``torch.profiler`` traces on rank 0.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch


class _DummyTorchProfiler:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def step(self) -> None:
        pass


class StepProfileRecorder:
    """
    Persistent step profiler for distributed training.

    The recorder keeps local timing state on every rank, aggregates it at
    train-step boundaries, and writes the aggregated profile on rank 0.
    """

    TRACE_WAIT_STEPS = 1
    TRACE_WARMUP_STEPS = 1
    TRACE_ACTIVE_STEPS = 2
    TRACE_REPEAT = 2
    HEARTBEAT_INTERVAL_S = 1.0

    def __init__(self, enabled: bool, output_dir: str, print_fn=None) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.print_fn = print_fn
        self.rank = torch.distributed.get_rank() if self._dist_enabled() else 0
        self.world_size = torch.distributed.get_world_size() if self._dist_enabled() else 1
        self.is_rank_0 = self.rank == 0

        self.current_step: Optional[int] = None
        self.current_episode: Optional[int] = None
        self.current_step_start_wall: Optional[float] = None
        self.current_step_started_at: Optional[float] = None
        self.section_totals: Dict[str, float] = {}
        self.phase_stack: List[str] = []
        self.active_section_name: Optional[str] = None
        self.active_section_start_wall: Optional[float] = None
        self.last_section_name: Optional[str] = None
        self.last_section_elapsed_s: Optional[float] = None
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._snapshot_generation = 0

        self.rank_step_profile_path = self.output_dir / f"step_profile.rank{self.rank}.jsonl"
        self.rank_latest_profile_path = self.output_dir / f"step_profile.rank{self.rank}.latest.json"
        self.rank_current_profile_path = self.output_dir / f"step_profile.rank{self.rank}.current.json"
        self.step_profile_path = self.output_dir / "step_profile.global.jsonl"
        self.latest_profile_path = self.output_dir / "step_profile.latest.json"
        self.current_profile_path = self.output_dir / "step_profile.current.json"
        self.trace_dir = self.output_dir / "traces"

        self._torch_profiler = None
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.is_rank_0:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._torch_profiler = self._build_torch_profiler()
            self._torch_profiler.start()
            self._start_heartbeat()

    @staticmethod
    def _dist_enabled() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    def _cuda_sync_if_available() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _build_torch_profiler(self):
        if not self.is_rank_0:
            return _DummyTorchProfiler()

        from torch.profiler import ProfilerActivity

        return torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.TRACE_WAIT_STEPS,
                warmup=self.TRACE_WARMUP_STEPS,
                active=self.TRACE_ACTIVE_STEPS,
                repeat=self.TRACE_REPEAT,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.trace_dir)),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )

    def start_step(self, train_step: int, episode: int) -> None:
        if not self.enabled:
            return

        self._cuda_sync_if_available()
        with self._state_lock:
            self._snapshot_generation += 1
            self.current_step = int(train_step)
            self.current_episode = int(episode)
            self.current_step_start_wall = time.perf_counter()
            self.current_step_started_at = time.time()
            self.section_totals = {}
            self.phase_stack = []
            self.active_section_name = None
            self.active_section_start_wall = None
            self.last_section_name = None
            self.last_section_elapsed_s = None
        self._write_current_snapshot()

    @contextmanager
    def phase(self, phase_name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        cleaned = phase_name.strip("/")
        if not cleaned:
            yield
            return

        self.phase_stack.append(cleaned)
        try:
            yield
        finally:
            self.phase_stack.pop()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled or self.current_step is None:
            yield
            return

        full_name = self._qualify_name(name)
        record_ctx = torch.profiler.record_function(full_name) if self.is_rank_0 else nullcontext()

        self._cuda_sync_if_available()
        start = time.perf_counter()
        with self._state_lock:
            self.active_section_name = full_name
            self.active_section_start_wall = start
        self._write_current_snapshot()
        with record_ctx:
            try:
                yield
            finally:
                self._cuda_sync_if_available()
                elapsed = time.perf_counter() - start
                with self._state_lock:
                    self.section_totals[full_name] = self.section_totals.get(full_name, 0.0) + elapsed
                    self.active_section_name = None
                    self.active_section_start_wall = None
                    self.last_section_name = full_name
                    self.last_section_elapsed_s = elapsed
                self._write_current_snapshot()

    def _qualify_name(self, name: str) -> str:
        cleaned = name.strip("/")
        if not self.phase_stack:
            return cleaned
        return "/".join([*self.phase_stack, cleaned])

    def finish_step(self, extra: Optional[Dict] = None) -> Optional[Dict]:
        if not self.enabled or self.current_step is None or self.current_step_start_wall is None:
            return None

        self._cuda_sync_if_available()
        with self._state_lock:
            train_step = int(self.current_step)
            episode = int(self.current_episode) if self.current_episode is not None else None
            started_at = self.current_step_started_at
            total_elapsed = time.perf_counter() - self.current_step_start_wall
            local_sections = dict(self.section_totals)
            local_sections["step/total"] = total_elapsed
            self._snapshot_generation += 1
            self.current_step = None
            self.current_episode = None
            self.current_step_start_wall = None
            self.current_step_started_at = None
            self.section_totals = {}
            self.phase_stack = []
            self.active_section_name = None
            self.active_section_start_wall = None
            self.last_section_name = None
            self.last_section_elapsed_s = None

        local_step_total_s = local_sections.get("step/total", total_elapsed)
        local_ratios = {
            name: (value / local_step_total_s if local_step_total_s > 0 else 0.0)
            for name, value in local_sections.items()
        }
        local_record = {
            "train_step": train_step,
            "episode": episode,
            "rank": self.rank,
            "world_size": self.world_size,
            "started_at": started_at,
            "finished_at": time.time(),
            "sections_local_s": local_sections,
            "sections_local_ratio": local_ratios,
        }
        if extra:
            local_record["extra"] = extra
        self._append_jsonl(self.rank_step_profile_path, local_record)
        self._write_atomic_json(self.rank_latest_profile_path, local_record)
        self._write_atomic_json(self.rank_current_profile_path, local_record)

        gathered_sections = self._gather_sections(local_sections)
        self._torch_profiler.step()

        result = None
        if self.is_rank_0:
            aggregated = self._aggregate_sections(gathered_sections)
            step_total_s = aggregated["max_s"].get("step/total", total_elapsed)
            ratios = {
                name: (value / step_total_s if step_total_s > 0 else 0.0)
                for name, value in aggregated["max_s"].items()
            }
            mean_ratios = {
                name: (
                    value / aggregated["mean_s"].get("step/total", step_total_s)
                    if aggregated["mean_s"].get("step/total", step_total_s) > 0 else 0.0
                )
                for name, value in aggregated["mean_s"].items()
            }
            record = {
                "train_step": train_step,
                "episode": episode,
                "world_size": self.world_size,
                "available_ranks": list(range(self.world_size)),
                "started_at": started_at,
                "finished_at": time.time(),
                "sections_max_s": aggregated["max_s"],
                "sections_mean_s": aggregated["mean_s"],
                "sections_max_ratio": ratios,
                "sections_mean_ratio": mean_ratios,
            }
            if extra:
                record["extra"] = extra
            self._append_jsonl(self.step_profile_path, record)
            self._write_atomic_json(self.latest_profile_path, record)
            self._write_atomic_json(self.current_profile_path, record)
            result = {
                "record": record,
                "wandb_logs": self._build_wandb_logs(train_step, aggregated["max_s"], ratios),
                "summary": self._build_summary(aggregated["max_s"], ratios),
            }
        return result

    def close(self) -> None:
        if not self.enabled:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(self.HEARTBEAT_INTERVAL_S * 2, 2.0))
        if self._torch_profiler is not None:
            self._torch_profiler.stop()

    def _gather_sections(self, local_sections: Dict[str, float]) -> List[Dict[str, float]]:
        if not self._dist_enabled():
            return [local_sections]

        gathered_sections = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered_sections, local_sections)
        return [item or {} for item in gathered_sections]

    @staticmethod
    def _aggregate_sections(section_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        section_names = sorted({name for section_dict in section_list for name in section_dict})
        max_s: Dict[str, float] = {}
        mean_s: Dict[str, float] = {}
        world_size = max(len(section_list), 1)

        for name in section_names:
            values = [float(section_dict.get(name, 0.0)) for section_dict in section_list]
            max_s[name] = max(values)
            mean_s[name] = sum(values) / world_size

        return {"max_s": max_s, "mean_s": mean_s}

    @staticmethod
    def _flatten_section_name(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_")

    def _build_wandb_logs(self, train_step: int, max_s: Dict[str, float], ratios: Dict[str, float]) -> Dict[str, float]:
        logs = {"profile/train_step": train_step}
        for name, value in max_s.items():
            flat_name = self._flatten_section_name(name)
            logs[f"profile/{flat_name}_s"] = value
            logs[f"profile/{flat_name}_ratio"] = ratios.get(name, 0.0)
        return logs

    @staticmethod
    def _build_summary(max_s: Dict[str, float], ratios: Dict[str, float]) -> str:
        interesting_sections = [
            "collect/total",
            "collect/generate",
            "learn/total",
            "learn/update_engine_weights",
            "eval/total",
            "checkpoint/total",
        ]
        parts = []
        step_total = max_s.get("step/total", 0.0)
        parts.append(f"step_total={step_total:.2f}s")
        for name in interesting_sections:
            if name not in max_s:
                continue
            parts.append(f"{name}={max_s[name]:.2f}s ({ratios.get(name, 0.0):.1%})")
        return "profile: " + ", ".join(parts)

    def _write_current_snapshot(self) -> None:
        snapshot = self._build_current_snapshot()
        if snapshot is None:
            return
        if not self._write_snapshot_if_current(self.rank_current_profile_path, snapshot):
            return
        if not self.is_rank_0:
            return

        global_snapshot = self._build_global_current_snapshot(snapshot)
        if global_snapshot is None:
            return
        self._write_atomic_json(self.current_profile_path, global_snapshot)

    def _build_current_snapshot(self) -> Optional[Dict]:
        if not self.enabled:
            return None

        with self._state_lock:
            if self.current_step is None or self.current_step_start_wall is None:
                return None

            current_elapsed_s = max(time.perf_counter() - self.current_step_start_wall, 0.0)
            sections_local_s = dict(self.section_totals)
            active_section_elapsed_s = None
            if self.active_section_name is not None and self.active_section_start_wall is not None:
                active_section_elapsed_s = max(time.perf_counter() - self.active_section_start_wall, 0.0)
                sections_local_s[self.active_section_name
                                 ] = (sections_local_s.get(self.active_section_name, 0.0) + active_section_elapsed_s)
            current_ratios = {
                name: (value / current_elapsed_s if current_elapsed_s > 0 else 0.0)
                for name, value in sections_local_s.items()
            }
            snapshot = {
                "train_step": self.current_step,
                "episode": self.current_episode,
                "rank": self.rank,
                "world_size": self.world_size,
                "started_at": self.current_step_started_at,
                "partial": True,
                "current_elapsed_s": current_elapsed_s,
                "sections_local_s": sections_local_s,
                "sections_local_ratio": current_ratios,
                "_snapshot_generation": self._snapshot_generation,
            }
            if self.last_section_name is not None:
                snapshot["last_section"] = self.last_section_name
            if self.last_section_elapsed_s is not None:
                snapshot["last_elapsed_s"] = self.last_section_elapsed_s
            if self.active_section_name is not None:
                snapshot["active_section"] = self.active_section_name
            if active_section_elapsed_s is not None:
                snapshot["active_section_elapsed_s"] = active_section_elapsed_s
            return snapshot

    def _build_global_current_snapshot(self, rank0_snapshot: Dict) -> Optional[Dict]:
        current_step = rank0_snapshot.get("train_step")
        current_episode = rank0_snapshot.get("episode")
        if current_step is None:
            return None

        snapshots = []
        available_ranks = []
        active_sections = {}
        for rank in range(self.world_size):
            if rank == self.rank:
                candidate = dict(rank0_snapshot)
            else:
                candidate = self._read_json(self.output_dir / f"step_profile.rank{rank}.current.json")
            if not candidate:
                continue
            if candidate.get("train_step") != current_step or candidate.get("episode") != current_episode:
                continue
            snapshots.append(candidate)
            available_ranks.append(rank)
            active_section = candidate.get("active_section")
            if active_section:
                active_sections[f"rank{rank}"] = active_section

        if not snapshots:
            return None

        aggregated = self._aggregate_sections([snapshot.get("sections_local_s", {}) for snapshot in snapshots])
        elapsed_values = [float(snapshot.get("current_elapsed_s", 0.0)) for snapshot in snapshots]
        max_elapsed = max(elapsed_values) if elapsed_values else 0.0
        mean_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0
        max_ratios = {
            name: (value / max_elapsed if max_elapsed > 0 else 0.0)
            for name, value in aggregated["max_s"].items()
        }
        mean_ratios = {
            name: (value / mean_elapsed if mean_elapsed > 0 else 0.0)
            for name, value in aggregated["mean_s"].items()
        }
        started_at_candidates = [
            snapshot.get("started_at") for snapshot in snapshots if snapshot.get("started_at") is not None
        ]

        global_snapshot = {
            "train_step": current_step,
            "episode": current_episode,
            "world_size": self.world_size,
            "available_ranks": available_ranks,
            "num_rank_snapshots": len(snapshots),
            "started_at": min(started_at_candidates) if started_at_candidates else None,
            "partial": True,
            "current_elapsed_max_s": max_elapsed,
            "current_elapsed_mean_s": mean_elapsed,
            "sections_max_s": aggregated["max_s"],
            "sections_mean_s": aggregated["mean_s"],
            "sections_max_ratio": max_ratios,
            "sections_mean_ratio": mean_ratios,
        }
        if active_sections:
            global_snapshot["active_sections"] = active_sections
        return global_snapshot

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="step-profile-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.HEARTBEAT_INTERVAL_S):
            self._write_current_snapshot()

    def _write_snapshot_if_current(self, path: Path, payload: Dict) -> bool:
        snapshot_generation = payload.get("_snapshot_generation")
        if snapshot_generation is None:
            self._write_atomic_json(path, payload)
            return True

        sanitized_payload = dict(payload)
        sanitized_payload.pop("_snapshot_generation", None)
        with self._write_lock:
            with self._state_lock:
                if snapshot_generation != self._snapshot_generation:
                    return False
            self._write_atomic_json_unlocked(path, sanitized_payload)
        return True

    def _append_jsonl(self, path: Path, payload: Dict) -> None:
        with self._write_lock:
            self._append_jsonl_unlocked(path, payload)

    @staticmethod
    def _append_jsonl_unlocked(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_atomic_json(self, path: Path, payload: Dict) -> None:
        with self._write_lock:
            self._write_atomic_json_unlocked(path, payload)

    @staticmethod
    def _write_atomic_json_unlocked(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
