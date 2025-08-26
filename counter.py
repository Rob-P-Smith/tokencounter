import time
import asyncio
from collections import deque
from typing import Optional, Deque, Dict, Tuple
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        sliding_window_sec: int = Field(5, description="Averaging window for TPS")
        update_interval_ms: int = Field(
            600, description="UI refresh cadence during stream"
        )
        decimals: int = Field(2, description="Numeric precision")
        label_prefix: str = Field("⏱ TPS", description="Prefix shown in the status row")
        show_intermediate: bool = Field(True, description="Update while streaming")
        show_final_card: bool = Field(
            False, description="Post a final message card too"
        )
        max_context_tokens: int = Field(
            49152, description="Total allowed context tokens"
        )
        warning_threshold_percent: float = Field(
            50.0,
            description="Trigger alert when usage reaches this % of max",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._sessions: Dict[str, Dict] = {}
        self._conversation_token_totals: Dict[str, int] = {}

    @staticmethod
    def _fmt(n: float, d: int) -> str:
        return f"{n:.{d}f}"

    def _render_bar(
        self,
        ttft: Optional[float],
        tps5: Optional[float],
        total_tokens: int,
        remaining_percent: float,
        total_time: Optional[float],
        current_response_tokens: int = None,
        is_final: bool = False,
    ) -> str:
        d = int(self.valves.decimals)
        parts = []

        if is_final:
            prefix = "✅ Final"
        else:
            prefix = self.valves.label_prefix

        if ttft is not None:
            parts.append(f"TTFT {self._fmt(ttft, d)}s")

        if tps5 is not None and not is_final:
            parts.append(f"TPS(5s) {self._fmt(tps5, d)}")

        if total_time is not None and total_time > 0:
            parts.append(f"Total {self._fmt(total_time, d)}s")

        parts.append(f"Tokens {total_tokens}")
        parts.append(f"Remaining {remaining_percent:.{d}f}%")

        response_tokens = (
            current_response_tokens
            if current_response_tokens is not None
            else total_tokens
        )

        if response_tokens > 0 and total_time is not None:
            tta = response_tokens / max(total_time or 1e-6, 1e-6)
            parts.append(f"TTA {self._fmt(tta, d)}/s")

        if ttft is not None and total_time is not None and response_tokens > 0:
            gen_time = max(total_time - ttft, 1e-6)
            tra = response_tokens / gen_time
            parts.append(f"TRA {self._fmt(tra, d)}/s")

        return f"{prefix}: " + "  |  ".join(parts)

    async def _emit_status(self, emitter, description: str, done: bool = False):
        if not emitter:
            return
        await emitter(
            {
                "type": "status",
                "data": {"description": description, "done": done, "hidden": False},
            }
        )

    async def _emit_message(self, emitter, content: str):
        if not emitter:
            return
        await emitter({"type": "message", "data": {"role": "tool", "content": content}})

    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        **user: Optional[dict],
    ) -> dict:
        conv_id = body.get("conversation_id") or "conv"
        rid = f"{conv_id}:{int(time.time() * 1000)}"
        body.setdefault("_token_counter_meta_", {})["rid"] = rid
        self._sessions[rid] = {
            "start_ts": time.perf_counter(),
            "first_token_ts": None,
            "last_ui": 0.0,
            "token_count": 0,
            "samples": deque(),
        }
        if conv_id not in self._conversation_token_totals:
            self._conversation_token_totals[conv_id] = 0

        await self._emit_status(
            __event_emitter__, f"{self.valves.label_prefix}: measuring…", done=False
        )
        return body

    def stream(
        self,
        event: dict,
        __event_emitter__=None,
        **user: Optional[dict],
    ) -> dict:
        rid = None
        req_meta = event.get("request") or {}
        rid = req_meta.get("rid") or (
            req_meta.get("_token_counter_meta_", {}) or {}
        ).get("rid")
        if not rid and self._sessions:
            rid = list(self._sessions.keys())[-1]

        S = self._sessions.get(rid)
        if not S:
            return event

        now = time.perf_counter()
        delta = None
        for choice in event.get("choices", []):
            delta = delta or choice.get("delta", {}).get("content")
        delta = delta or event.get("content") or event.get("text") or event.get("token")

        inc = (
            max(1, len(delta.strip().split()))
            if isinstance(delta, str) and delta
            else 1
        )
        S["token_count"] += inc

        samples: Deque[Tuple[float, int]] = S["samples"]
        samples.append((now, inc))

        cutoff = now - self.valves.sliding_window_sec
        while samples and samples[0][0] < cutoff:
            samples.popleft()

        if S["first_token_ts"] is None and S["token_count"] > 0:
            S["first_token_ts"] = now

        conv_id = rid.split(":", 1)[0]
        total_tokens_for_conv = (
            self._conversation_token_totals[conv_id] + S["token_count"]
        )

        max_tokens = self.valves.max_context_tokens
        remaining_percent = ((max_tokens - total_tokens_for_conv) / max_tokens) * 100.0

        threshold_pct = self.valves.warning_threshold_percent
        warning_triggered = remaining_percent <= threshold_pct and remaining_percent > 0

        refresh_interval = max(0.05, self.valves.update_interval_ms / 1000.0)
        if self.valves.show_intermediate and (now - S["last_ui"]) >= refresh_interval:
            tps5 = 0.0
            if len(samples) >= 2:
                span = samples[-1][0] - samples[0][0]
                total_in_window = sum(n for _, n in samples)
                tps5 = total_in_window / max(span, 1e-6)

            ttft = (
                (S["first_token_ts"] - S["start_ts"]) if S["first_token_ts"] else None
            )
            total_time = (now - S["start_ts"]) if S["token_count"] > 0 else None

            bar = self._render_bar(
                ttft,
                tps5,
                total_tokens_for_conv,
                remaining_percent,
                total_time,
                current_response_tokens=S["token_count"],
                is_final=False,
            )

            try:
                asyncio.create_task(
                    self._emit_status(__event_emitter__, bar, done=False)
                )
            except Exception:
                pass
            S["last_ui"] = now

        if warning_triggered and not S.get("warning_sent", False):
            warn_msg = f"⚠️ Warning: Conversation is {threshold_pct}% complete ({total_tokens_for_conv}/{max_tokens} tokens). Consider compacting."
            try:
                asyncio.create_task(self._emit_message(__event_emitter__, warn_msg))
            except Exception:
                pass
            S["warning_sent"] = True

        return event

    async def outlet(
        self,
        body: dict,
        __event_emitter__=None,
        **user: Optional[dict],
    ) -> dict:
        rid = None
        meta = body.get("_token_counter_meta_", {})
        rid = meta.get("rid")
        if not rid and self._sessions:
            rid = list(self._sessions.keys())[-1]

        S = self._sessions.get(rid)
        if not S:
            return body

        end_ts = time.perf_counter()
        total_time = end_ts - S["start_ts"]
        ttft = (S["first_token_ts"] - S["start_ts"]) if S["first_token_ts"] else None

        conv_id = rid.split(":", 1)[0]

        self._conversation_token_totals[conv_id] += S["token_count"]
        final_tokens = self._conversation_token_totals[conv_id]

        remaining_percent = (
            (self.valves.max_context_tokens - final_tokens)
            / self.valves.max_context_tokens
        ) * 100.0

        final_bar = self._render_bar(
            ttft,
            None,
            final_tokens,
            remaining_percent,
            total_time,
            current_response_tokens=S["token_count"],
            is_final=True,
        )

        await self._emit_status(__event_emitter__, final_bar, done=True)

        threshold_pct = self.valves.warning_threshold_percent
        if self.valves.show_final_card and remaining_percent <= threshold_pct:
            warning_msg = f"📊 Context usage at {final_tokens}/{self.valves.max_context_tokens} ({remaining_percent:.1f}% remaining). Compact history to avoid overflow."
            await self._emit_message(__event_emitter__, warning_msg)

        self._sessions.pop(rid, None)

        return body
