"""
Retry ONE model request on 429, instead of restarting the whole agent run.

A whole-run retry re-executes every tool call and re-spends every token,
which on an 8k tokens-per-minute tier makes the next 429 almost certain.
"""

import asyncio
import os
import re

from pydantic_ai.exceptions import ModelHTTPError, ModelRateLimitError
from pydantic_ai.models.wrapper import WrapperModel


def retry_after_seconds(error: Exception, attempt: int) -> float:
    """Provider hint ('try again in 1.33s') or exponential backoff, capped at 60 s."""

    hint = getattr(error, "retry_after", None)
    if hint is None:
        match = re.search(r"try again in ([\d.]+)s", str(error))
        hint = float(match.group(1)) if match else None
    backoff = min(5.0 * 2 ** (attempt - 1), 60.0)  # 5, 10, 20, 40, 60 ...
    return min(max(hint or 0.0, backoff) + 1.0, 61.0)

def is_parse_failure(error: Exception) -> bool:
    """Groq 400 'output_parse_failed': the model wrote prose instead of a tool call."""

    return (isinstance(error, ModelHTTPError) and error.status_code == 400
            and "output_parse_failed" in str(error.body))

class RateLimitRetryModel(WrapperModel):
    def __init__(self, wrapped, max_attempts: int | None = None) -> None:
        super().__init__(wrapped)
        self.max_attempts = max_attempts or int(os.getenv("RATE_LIMIT_RETRIES", "6"))
        self.waits: list[float] = []  # exposed for the report
        self.max_parse_retries = int(os.getenv("PARSE_FAILURE_RETRIES", "2"))
        self.parse_retries_used = 0

    async def request(self, messages, model_settings, model_request_parameters):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except ModelRateLimitError as error:
                if attempt == self.max_attempts:
                    raise
                wait = retry_after_seconds(error, attempt)
                self.waits.append(wait)
                print(f"  429 rate limit, waiting {wait:.0f}s (attempt {attempt}/{self.max_attempts})")
                await asyncio.sleep(wait)
            except ModelHTTPError as error:
                # Sampling glitch, not a bad request: re-ask once or twice, then give up visibly.
                if not is_parse_failure(error) or self.parse_retries_used >= self.max_parse_retries:
                    raise
                self.parse_retries_used += 1
                print(f"  400 output_parse_failed, re-asking ({self.parse_retries_used}/{self.max_parse_retries})")

# class RateLimitRetryModel(WrapperModel):
#     def __init__(self, wrapped, max_attempts: int | None = None) -> None:
#         super().__init__(wrapped)
#         self.max_attempts = max_attempts or int(os.getenv("RATE_LIMIT_RETRIES", "6"))
#         self.waits: list[float] = []  # exposed for the report

#     async def request(self, messages, model_settings, model_request_parameters):
#         for attempt in range(1, self.max_attempts + 1):
#             try:
#                 return await super().request(messages, model_settings, model_request_parameters)
#             except ModelRateLimitError as error:
#                 if attempt == self.max_attempts:
#                     raise
#                 wait = retry_after_seconds(error, attempt)
#                 self.waits.append(wait)
#                 print(f"  429 rate limit, waiting {wait:.0f}s (attempt {attempt}/{self.max_attempts})")
#                 await asyncio.sleep(wait)