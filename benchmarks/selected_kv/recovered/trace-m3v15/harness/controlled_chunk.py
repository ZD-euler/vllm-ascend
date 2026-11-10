"""Test-only scheduler split for an actual unaligned H continuation.

The first execution of a `uc-h<H>-...` request is capped to H-65536 tokens.
The next execution sees the real H and is allowed to schedule the requested Q.
All native allocation and scheduling calls remain delegated.
"""
from __future__ import annotations

import re

from vllm.v1.core.sched.scheduler import Scheduler as _NativeScheduler

_ID = re.compile(r"^uc-h(\d+)-")


class ControlledScheduler(_NativeScheduler):
    def schedule(self, *args, **kwargs):
        requests = [r for key, r in self.requests.items() if _ID.match(key)]
        if not requests:
            return super().schedule(*args, **kwargs)
        histories = {int(_ID.match(r.request_id).group(1)) for r in requests}
        if len(histories) != 1:
            raise AssertionError("one continuation geometry per batch is required")
        history = next(iter(histories))
        remaining_prefix = history - 65536
        if remaining_prefix <= 0:
            return super().schedule(*args, **kwargs)
        seen = getattr(self, "_cmp_started", set())
        cap = remaining_prefix if all(r.request_id not in seen for r in requests) else 4096
        cfg = self.scheduler_config
        previous = cfg.long_prefill_token_threshold
        cfg.long_prefill_token_threshold = cap
        try:
            output = super().schedule(*args, **kwargs)
            seen.update(key for key in output.num_scheduled_tokens if _ID.match(key))
            self._cmp_started = seen
            return output
        finally:
            cfg.long_prefill_token_threshold = previous
