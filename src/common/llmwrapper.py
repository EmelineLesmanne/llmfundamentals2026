import functools
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import anthropic
import openai
from dotenv import load_dotenv
from loguru import logger

Provider = Literal["anthropic", "openai"]

# Both providers go through OpenRouter: Claude with the Anthropic SDK, OpenAI with the OpenAI SDK.
OPENROUTER_ANTHROPIC_URL = "https://openrouter.ai/api"
OPENROUTER_OPENAI_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "anthropic/claude-haiku-4.5",
    "openai": "openai/gpt-4.1-mini",
}

# USD per million tokens. Check the OpenRouter models page before relying on these.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
# Response cache: a SQLite file at the project root (git-ignored), entries live one month.
CACHE_DB = Path(__file__).resolve().parents[2] / ".cache" / "llm_cache.sqlite"
CACHE_TTL_SECONDS = 30 * 24 * 3600


# Temporary errors worth retrying. BadRequestError (400) is not here: the same request
# would fail again, so it is raised immediately.
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function on RETRYABLE_ERRORS, waiting base_delay * 2**attempt seconds between tries."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as exc:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * 2**attempt
                    logger.warning(
                        "{} (retry {}/{} in {}s)", type(exc).__name__, attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model's price is unknown
    stop_reason: str | None = None  # e.g. "end_turn" or "max_tokens" (Claude), "stop" or "length" (OpenAI)
    latency_seconds: float | None = None  # wall-clock time of ask(), including retries


class LLMWrapper:
    """Handles API connection to the chosen LLM."""

    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        max_tokens: int = 1000,
        use_cache: bool = True,
    ):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider!r}")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (check your .env file)")

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.max_tokens = max_tokens
        self._anthropic = (
            anthropic.Anthropic(base_url=OPENROUTER_ANTHROPIC_URL, auth_token=api_key, max_retries=0)
            if provider == "anthropic"
            else None
        )
        self._openai = (
            openai.OpenAI(base_url=OPENROUTER_OPENAI_URL, api_key=api_key, max_retries=0)
            if provider == "openai"
            else None
        )
        # Without cache, an in-memory database is used: it starts empty and is thrown away.
        if use_cache:
            CACHE_DB.parent.mkdir(exist_ok=True)
        self._db = sqlite3.connect(CACHE_DB if use_cache else ":memory:")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
        )
        self._db.execute("DELETE FROM cache WHERE created_at < ?", (time.time() - CACHE_TTL_SECONDS,))
        self._db.commit()

    def ask(
        self,
        prompt: str,
        system: str | None = None,
        with_streaming: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a single prompt and return the text, token usage and cost.

        If with_streaming is given, the response is streamed and with_streaming is called with
        each text chunk as it arrives. The full LLMResponse is still returned.

        Answers are cached for a month, keyed by SHA-256 of provider, model, max_tokens, system
        and prompt. A cache hit costs nothing and, when streaming, is sent as a single chunk.
        """
        start = time.perf_counter()
        key = self._cache_key(prompt, system)
        cached = self._cache_get(key)
        if cached is not None:
            logger.debug("Cache hit for {}/{}", self.provider, self.model)
            if with_streaming is not None:
                with_streaming(cached.text)
            response = replace(cached, cost_usd=0.0)
        else:
            response = self._request(prompt, system, with_streaming)
            self._cache_set(key, response)  # stored without latency: it depends on each call
        return replace(response, latency_seconds=round(time.perf_counter() - start, 3))

    @retry_with_backoff()
    def _request(
        self,
        prompt: str,
        system: str | None,
        with_streaming: Callable[[str], None] | None,
    ) -> LLMResponse:
        logger.debug("Sending request to {}/{}", self.provider, self.model)
        if self.provider == "anthropic":
            return self._ask_anthropic(prompt, system, with_streaming)
        return self._ask_openai(prompt, system, with_streaming)

    def _cache_key(self, prompt: str, system: str | None) -> str:
        payload = json.dumps([self.provider, self.model, self.max_tokens, system, prompt])
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMResponse | None:
        row =self._db.execute(
            "SELECT response FROM cache WHERE key = ? AND created_at >= ?",
            (key, time.time() - CACHE_TTL_SECONDS),
        ).fetchone()
        return LLMResponse(**json.loads(row[0])) if row else None

    def _cache_set(self, key: str, response: LLMResponse) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                (key, json.dumps(asdict(response)), time.time()),
            )

    def _cost(self, input_tokens: int, output_tokens: int) -> float | None:
        prices = PRICES_PER_MTOK.get(self.model)
        if prices is None:
            return None
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        return round(cost, 3)

    def _ask_anthropic(
        self,
        prompt: str,
        system: str | None,
        with_streaming: Callable[[str], None] | None,
    ) -> LLMResponse:
        kwargs = {"system": system} if system else {}
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if with_streaming is None:
            response = self._anthropic.messages.create(**request)
        else:
            with self._anthropic.messages.stream(**request) as stream:
                for text in stream.text_stream:
                    with_streaming(text)
                response = stream.get_final_message()
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude refused the request: {response.stop_details}")
        usage = response.usage
        return LLMResponse(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self._cost(usage.input_tokens, usage.output_tokens),
            stop_reason=response.stop_reason,
        )

    def _ask_openai(
        self,
        prompt: str,
        system: str | None,
        with_streaming: Callable[[str], None] | None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": messages,
        }
        if with_streaming is None:
            response = self._openai.chat.completions.create(**request)
            choice = response.choices[0]
            text, stop_reason, usage = choice.message.content or "", choice.finish_reason, response.usage
        else:
            text, stop_reason, usage = "", None, None
            # include_usage adds a last chunk with the token counts (and no choices)
            stream = self._openai.chat.completions.create(
                **request, stream=True, stream_options={"include_usage": True}
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        with_streaming(delta)
                    text += delta
                    stop_reason = chunk.choices[0].finish_reason or stop_reason
                if chunk.usage:
                    usage = chunk.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=self._cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            ),
            stop_reason=stop_reason,
        )


if __name__ == "__main__":
    # allows to quickly check that LLMWrapper works
    load_dotenv()  # reads OPENROUTER_API_KEY from the .env file
    for provider in ("anthropic", "openai"):
        llm = LLMWrapper(provider)
        logger.info("{}: {}", provider, llm.ask("Say hello in French."))

        logger.info(f"{provider} (streamed): ", end="")
        llm.ask("Say hello in Spanish.", with_streaming=lambda chunk: print(chunk, end="", flush=True))
        print()
