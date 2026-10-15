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
from anthropic import transform_schema
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ValidationError

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


class StructuredOutputError(Exception):
    """The answer of the model could not be used.

    answer: the raw text returned by the model (or the fake answer)
    problems: what was wrong, one line per problem
    """

    def __init__(self, answer: str, problems: str):
        super().__init__(problems)
        self.answer = answer
        self.problems = problems


def _describe_problems(error: ValidationError) -> str:
    """One line per problem: the field name, then the message (no URLs, no input dumps)."""
    lines = []
    for problem in error.errors():
        field = ".".join(str(part) for part in problem["loc"]) or "answer"
        lines.append(f"{field}: {problem['msg']}")
    return "\n".join(lines)


# Fake answers let us test validation and retries without calling the model.
_fake_answers: list[str] = []
_received_prompts: list[str] = []  # every prompt given to ask_structured, used by the checker


def fake_next(answer: str | Path) -> None:
    """Queue a fake answer for the next ask_structured() call: raw text, or the path of a .json file."""
    answer = str(answer)
    if not answer.lstrip().startswith(("{", "[")) and Path(answer).is_file():
        answer = Path(answer).read_text(encoding="utf-8")
    _fake_answers.append(answer)


def clear_fakes() -> None:
    _fake_answers.clear()


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
        use_cache: bool = False,
    ):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider!r}")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (check your .env file)")

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.max_tokens = max_tokens
        self.use_cache = use_cache
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
        CACHE_DB.parent.mkdir(exist_ok=True)
        self._db = sqlite3.connect(CACHE_DB)
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
        schema: dict | None = None,
        use_cache: bool | None = None,
    ) -> LLMResponse:
        """Send a single prompt and return the text, token usage and cost.

        If with_streaming is given, the response is streamed and with_streaming is called with
        each text chunk as it arrives. The full LLMResponse is still returned.

        If schema (a JSON Schema as a dict) is given, the model is asked to answer with JSON that
        follows it. See ask_structured() for the version that also validates the answer.

        use_cache decides whether the cache is used for this call. With None (the default), the
        value given to the constructor is used, and that one is False by default.
        When used, answers are cached for a month, keyed by SHA-256 of provider, model, max_tokens,
        system, schema and prompt. A cache hit costs nothing and, when streaming, is sent as a
        single chunk.
        """
        start = time.perf_counter()
        if use_cache is None:
            use_cache = self.use_cache
        key = self._cache_key(prompt, system, schema)
        cached = self._cache_get(key) if use_cache else None
        if cached is not None:
            logger.debug("Cache hit for {}/{}", self.provider, self.model)
            if with_streaming is not None:
                with_streaming(cached.text)
            response = replace(cached, cost_usd=0.0)
        else:
            response = self._request(prompt, system, with_streaming, schema)
            if use_cache:
                self._cache_set(key, response)  # stored without latency: it depends on each call
        return replace(response, latency_seconds=round(time.perf_counter() - start, 3))

    @retry_with_backoff()
    def _request(
        self,
        prompt: str,
        system: str | None,
        with_streaming: Callable[[str], None] | None,
        schema: dict | None,
    ) -> LLMResponse:
        logger.debug("Sending request to {}/{}", self.provider, self.model)
        if self.provider == "anthropic":
            return self._ask_anthropic(prompt, system, with_streaming, schema)
        return self._ask_openai(prompt, system, with_streaming, schema)

    def ask_structured(self, prompt: str, output_class: type[BaseModel], verbose: bool = False) -> BaseModel:
        """Ask for an answer that follows output_class (a Pydantic model) and return it as an object.

        Exactly one attempt: if the answer is not valid, StructuredOutputError is raised and
        the caller decides what to do. If fake answers are queued (see fake_next), the next one
        is used instead of calling the model.
        """
        _received_prompts.append(prompt)
        if verbose:
            logger.info(f"[ask_structured] provider: {self.provider} / model: {self.model}")
            logger.info(f"[ask_structured] schema: {output_class.__name__} ({len(output_class.model_fields)} fields)")

        if _fake_answers:
            raw_text = _fake_answers.pop(0)
            kind = "fake"
        else:
            kind = "real"
            # transform_schema gives a schema both providers accept
            # (every field required, additionalProperties false, unsupported constraints removed)
            try:
                response = self.ask(prompt, schema=transform_schema(output_class))
            except RuntimeError as error:  # raised when the model refuses to answer
                raise StructuredOutputError("", str(error)) from error
            raw_text = response.text
            if response.stop_reason in ("max_tokens", "length"):
                raise StructuredOutputError(raw_text, "The answer was cut off (max tokens reached)")
        if verbose:
            logger.info(f"[ask_structured] answer ({kind}): {raw_text}")

        # Pydantic reports invalid JSON as a ValidationError too, so one except covers everything
        try:
            result = output_class.model_validate_json(raw_text)
        except ValidationError as error:
            problems = _describe_problems(error)
            if verbose:
                logger.info("[ask_structured] validation: FAILED")
                logger.info(problems)
            raise StructuredOutputError(raw_text, problems) from error
        if verbose:
            logger.info("[ask_structured] validation: OK")
        return result

    def _cache_key(self, prompt: str, system: str | None, schema: dict | None) -> str:
        payload = json.dumps([self.provider, self.model, self.max_tokens, system, schema, prompt])
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
        schema: dict | None,
    ) -> LLMResponse:
        kwargs = {"system": system} if system else {}
        if schema:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
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
        schema: dict | None,
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
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            }
            request["extra_body"] = {"provider": {"require_parameters": True}}
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
