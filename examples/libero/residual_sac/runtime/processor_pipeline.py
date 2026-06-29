from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
from typing import Protocol
from typing import Sequence

from .processor_protocol import NormalizedChunkResult
from .processor_protocol import normalize_chunk_result
from .processor_protocol import reconstruct_chunk_execution_record_from_normalized
from .transition_assembly import AssemblyResult
from .transition_assembly import BatchAwareLiberoTransitionAssembler
from .transition_assembly import ChunkExecutionRecord


@dataclass
class ProcessorChunkContext:
    payload: dict[str, Any]
    normalized_chunk: NormalizedChunkResult | None = None
    raw_chunk: ChunkExecutionRecord | None = None
    assembled_chunk: AssemblyResult | None = None


class ProcessorChunkStage(Protocol):
    name: str

    def run(self, context: ProcessorChunkContext) -> None: ...


class ProcessorTransitionStage(Protocol):
    name: str

    def run(
        self,
        assembled_chunk: AssemblyResult,
        *,
        context: ProcessorChunkContext,
    ) -> AssemblyResult: ...


@dataclass(slots=True)
class RewardRelabelChunkStage:
    reward_relabeler: Any
    task_id: int
    name: str = "reward_relabel"
    _active_episode_id: int | None = None

    def run(self, context: ProcessorChunkContext) -> None:
        payload = context.payload
        chunk_result = dict(payload["chunk_result"])
        episode_id = int(payload["episode_id"])
        if self._active_episode_id != int(episode_id):
            if self._active_episode_id is not None:
                self.reward_relabeler.finish_episode()
            self.reward_relabeler.start_episode(
                episode_id=int(episode_id),
                task_prompt=str(payload["task_prompt"]),
                initial_obs=self._initial_obs_from_chunk(chunk_result),
                init_episode_idx=int(payload.get("init_episode_idx", int(episode_id) - 1)),
                task_id=int(self.task_id),
            )
            self._active_episode_id = int(episode_id)

        relabeled_chunk = self.reward_relabeler.relabel_chunk(
            chunk_result,
            episode_step_start=int(payload["episode_step_start"]),
        )
        payload["chunk_result"] = relabeled_chunk

        if self._chunk_ended_episode(relabeled_chunk):
            self.reward_relabeler.finish_episode()
            self._active_episode_id = None

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        pending_contexts: list[ProcessorChunkContext] = []
        pending_specs: list[tuple[dict[str, Any], int]] = []

        def flush_pending() -> None:
            if not pending_contexts:
                return
            relabel_chunk_batch = getattr(
                self.reward_relabeler,
                "relabel_chunk_batch",
                None,
            )
            if callable(relabel_chunk_batch):
                relabeled_chunks = list(relabel_chunk_batch(pending_specs))
            else:
                relabeled_chunks = [
                    self.reward_relabeler.relabel_chunk(
                        chunk_result,
                        episode_step_start=int(episode_step_start),
                    )
                    for chunk_result, episode_step_start in pending_specs
                ]
            if len(relabeled_chunks) != len(pending_contexts):
                raise RuntimeError(
                    "reward relabeler returned a mismatched batch size: "
                    f"got {len(relabeled_chunks)} expected {len(pending_contexts)}"
                )
            for current_context, relabeled_chunk in zip(
                pending_contexts,
                relabeled_chunks,
            ):
                current_context.payload["chunk_result"] = relabeled_chunk
            pending_contexts.clear()
            pending_specs.clear()

        for context in contexts:
            payload = context.payload
            chunk_result = dict(payload["chunk_result"])
            episode_id = int(payload["episode_id"])
            if self._active_episode_id != int(episode_id):
                flush_pending()
                if self._active_episode_id is not None:
                    self.reward_relabeler.finish_episode()
                self.reward_relabeler.start_episode(
                    episode_id=int(episode_id),
                    task_prompt=str(payload["task_prompt"]),
                    initial_obs=self._initial_obs_from_chunk(chunk_result),
                    init_episode_idx=int(payload.get("init_episode_idx", int(episode_id) - 1)),
                    task_id=int(self.task_id),
                )
                self._active_episode_id = int(episode_id)

            pending_contexts.append(context)
            pending_specs.append(
                (
                    chunk_result,
                    int(payload["episode_step_start"]),
                )
            )

            if self._chunk_ended_episode(chunk_result):
                flush_pending()
                self.reward_relabeler.finish_episode()
                self._active_episode_id = None

        flush_pending()

    @staticmethod
    def _initial_obs_from_chunk(chunk_result: dict[str, Any]) -> dict[str, Any]:
        steps = list(chunk_result.get("steps", ()))
        if not steps:
            raise ValueError("reward relabel stage received empty chunk_result.steps")
        first_step = dict(steps[0])
        if "obs" not in first_step:
            raise ValueError("reward relabel stage requires step.obs in raw chunks")
        return dict(first_step["obs"])

    @staticmethod
    def _chunk_ended_episode(chunk_result: dict[str, Any]) -> bool:
        if bool(chunk_result.get("done", False)) or bool(
            chunk_result.get("truncated", False)
        ):
            return True
        steps = list(chunk_result.get("steps", ()))
        if not steps:
            return False
        last_step = dict(steps[-1])
        return bool(last_step.get("done", False)) or bool(
            last_step.get("truncated", False)
        )


@dataclass(frozen=True, slots=True)
class NormalizeChunkStage:
    name: str = "normalize_chunk"

    def run(self, context: ProcessorChunkContext) -> None:
        context.normalized_chunk = normalize_chunk_result(
            dict(context.payload["chunk_result"])
        )

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        for context in contexts:
            self.run(context)


@dataclass(frozen=True, slots=True)
class ReconstructChunkExecutionRecordStage:
    assembler: BatchAwareLiberoTransitionAssembler
    name: str = "reconstruct_chunk"

    def run(self, context: ProcessorChunkContext) -> None:
        normalized_chunk = context.normalized_chunk
        if normalized_chunk is None:
            raise RuntimeError("normalize_chunk stage must run before reconstruct")
        context.raw_chunk = reconstruct_chunk_execution_record_from_normalized(
            payload=context.payload,
            normalized_chunk=normalized_chunk,
            assembler=self.assembler,
        )

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        for context in contexts:
            self.run(context)


@dataclass(frozen=True, slots=True)
class AssembleTransitionsStage:
    assembler: BatchAwareLiberoTransitionAssembler
    name: str = "assemble_transitions"

    def run(self, context: ProcessorChunkContext) -> None:
        raw_chunk = context.raw_chunk
        if raw_chunk is None:
            raise RuntimeError("reconstruct_chunk stage must run before assemble")
        context.assembled_chunk = self.assembler.process_chunk(
            raw=raw_chunk,
            task_prompt=str(context.payload["task_prompt"]),
        )

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        raw_chunks: list[ChunkExecutionRecord] = []
        task_prompts: list[str] = []
        for context in contexts:
            raw_chunk = context.raw_chunk
            if raw_chunk is None:
                raise RuntimeError("reconstruct_chunk stage must run before assemble")
            raw_chunks.append(raw_chunk)
            task_prompts.append(str(context.payload["task_prompt"]))
        assembled_chunks = self.assembler.process_chunk_batch(
            raw_chunks=raw_chunks,
            task_prompts=task_prompts,
        )
        if len(assembled_chunks) != len(contexts):
            raise RuntimeError(
                "assembler returned a mismatched chunk batch size: "
                f"got {len(assembled_chunks)} expected {len(contexts)}"
            )
        for context, assembled_chunk in zip(contexts, assembled_chunks):
            context.assembled_chunk = assembled_chunk


class RolloutProcessorPipeline:
    def __init__(
        self,
        *,
        chunk_stages: Sequence[ProcessorChunkStage],
        transition_stages: Sequence[ProcessorTransitionStage] = (),
    ) -> None:
        self._chunk_stages = tuple(chunk_stages)
        self._transition_stages = tuple(transition_stages)

    @classmethod
    def for_libero(
        cls,
        *,
        assembler: BatchAwareLiberoTransitionAssembler,
        reward_relabeler: Any | None = None,
        task_id: int | None = None,
        transition_stages: Sequence[ProcessorTransitionStage] = (),
    ) -> "RolloutProcessorPipeline":
        reward_stages: tuple[ProcessorChunkStage, ...] = ()
        if reward_relabeler is not None and bool(
            getattr(reward_relabeler, "enabled", False)
        ):
            reward_stages = (
                RewardRelabelChunkStage(
                    reward_relabeler=reward_relabeler,
                    task_id=int(task_id or 0),
                ),
            )
        return cls(
            chunk_stages=(
                *reward_stages,
                NormalizeChunkStage(),
                ReconstructChunkExecutionRecordStage(assembler=assembler),
                AssembleTransitionsStage(assembler=assembler),
            ),
            transition_stages=transition_stages,
        )

    def process_payload(
        self,
        payload: dict[str, Any],
        *,
        timer: Any | None = None,
    ) -> AssemblyResult:
        assembled_chunks = self.process_payload_batch(
            payloads=(payload,),
            timer=timer,
        )
        if len(assembled_chunks) != 1:
            raise RuntimeError(
                "processor pipeline expected exactly one assembled chunk, "
                f"got {len(assembled_chunks)}"
            )
        return assembled_chunks[0]

    def process_payload_batch(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        timer: Any | None = None,
    ) -> list[AssemblyResult]:
        contexts = [ProcessorChunkContext(payload=dict(payload)) for payload in payloads]
        if not contexts:
            return []

        for stage in self._chunk_stages:
            stage_name = str(stage.name)
            self._run_stage_batch(
                stage_name=stage_name,
                timer=timer,
                batch_size=len(contexts),
                fn=lambda current_stage=stage: self._run_chunk_stage_batch(
                    current_stage,
                    contexts,
                ),
            )

        assembled_chunks: list[AssemblyResult] = []
        for context in contexts:
            assembled_chunk = context.assembled_chunk
            if assembled_chunk is None:
                raise RuntimeError("processor pipeline did not assemble transitions")
            assembled_chunks.append(assembled_chunk)

        for stage in self._transition_stages:
            stage_name = str(stage.name)

            def _run_transition_stage_batch(
                current_stage: ProcessorTransitionStage = stage,
            ) -> None:
                nonlocal assembled_chunks
                assembled_chunks = self._run_transition_stage_batch(
                    current_stage,
                    assembled_chunks,
                    contexts,
                )

            self._run_stage_batch(
                stage_name=stage_name,
                timer=timer,
                batch_size=len(contexts),
                fn=_run_transition_stage_batch,
            )

        return assembled_chunks

    @staticmethod
    def _run_stage(*, stage_name: str, timer: Any | None, fn: Any) -> None:
        if timer is None:
            fn()
            return
        with timer.context(str(stage_name)):
            fn()

    @staticmethod
    def _run_stage_batch(
        *,
        stage_name: str,
        timer: Any | None,
        batch_size: int,
        fn: Any,
    ) -> None:
        if timer is None:
            fn()
            return
        start_time = time.time()
        fn()
        duration = time.time() - start_time
        RolloutProcessorPipeline._record_timer_duration(
            timer=timer,
            stage_name=stage_name,
            duration_s=duration,
            count=max(1, int(batch_size)),
        )

    @staticmethod
    def _run_chunk_stage_batch(
        stage: ProcessorChunkStage,
        contexts: Sequence[ProcessorChunkContext],
    ) -> None:
        run_batch = getattr(stage, "run_batch", None)
        if callable(run_batch):
            run_batch(contexts)
            return
        for context in contexts:
            stage.run(context)

    @staticmethod
    def _run_transition_stage_batch(
        stage: ProcessorTransitionStage,
        assembled_chunks: Sequence[AssemblyResult],
        contexts: Sequence[ProcessorChunkContext],
    ) -> list[AssemblyResult]:
        run_batch = getattr(stage, "run_batch", None)
        if callable(run_batch):
            return list(run_batch(assembled_chunks, contexts=contexts))

        next_chunks: list[AssemblyResult] = []
        for assembled_chunk, context in zip(assembled_chunks, contexts):
            next_chunks.append(stage.run(assembled_chunk, context=context))
        return next_chunks

    @staticmethod
    def _record_timer_duration(
        *,
        timer: Any,
        stage_name: str,
        duration_s: float,
        count: int,
    ) -> None:
        timer.times[str(stage_name)] += float(duration_s)
        timer.counts[str(stage_name)] += max(1, int(count))


__all__ = [
    "AssembleTransitionsStage",
    "NormalizeChunkStage",
    "ProcessorChunkContext",
    "ProcessorChunkStage",
    "ProcessorTransitionStage",
    "ReconstructChunkExecutionRecordStage",
    "RewardRelabelChunkStage",
    "RolloutProcessorPipeline",
]
