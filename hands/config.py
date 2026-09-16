"""Project paths, environment loading, and input parsing helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_RUNS = PROJECT_ROOT / "evidence" / "runs"
CAPABILITIES_DIR = PROJECT_ROOT / "capabilities"
DEFAULT_POLICY = PROJECT_ROOT / "policy.yaml"
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_MOCK_URL = "http://127.0.0.1:5050/"

_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_provider(explicit: str | None = None) -> str:
    """Pick "anthropic" or "openai".

    Precedence: an explicit argument (e.g. a --provider CLI flag) wins,
    then the HANDS_PROVIDER environment variable, then auto-detection from
    whichever API key is present (ANTHROPIC_API_KEY vs OPENAI_API_KEY; if
    both are set, anthropic wins). If neither key is present, this still
    returns the default "anthropic" - the caller (build_provider) is the
    one that reports the missing-key error, not this function.
    """
    if explicit:
        return explicit
    env_provider = os.environ.get("HANDS_PROVIDER")
    if env_provider:
        return env_provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


def model_name(explicit: str | None = None, provider: str | None = None) -> str:
    """Resolve the model to use.

    Precedence: an explicit argument wins, then the HANDS_MODEL environment
    variable (a global override regardless of provider - this preserves
    the original no-provider-argument behaviour exactly for any existing
    caller), then a provider-specific default: DEFAULT_OPENAI_MODEL for
    "openai", DEFAULT_MODEL (the Anthropic default) for anything else,
    including the no-provider-given case.
    """
    if explicit:
        return explicit
    env_model = os.environ.get("HANDS_MODEL")
    if env_model:
        return env_model
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_MODEL


def build_provider(
    provider: str | None = None, model: str | None = None, fake_script: list[dict[str, Any]] | None = None
) -> Any:
    """Factory returning a ready `LLMProvider`.

    `fake_script`, if given, always wins and returns a `FakeLLM` regardless
    of `provider`/`model` - the same convenience a --fake-script CLI flag
    or a test harness would want, with no network/key requirements at all.

    Otherwise resolves the provider (see `resolve_provider`) and the model
    (see `model_name`), and raises a clear `SystemExit` naming the exact
    missing environment variable when the selected provider has no API key
    configured, before constructing anything that could touch the network.
    """
    if fake_script is not None:
        from hands.agent.llm import FakeLLM

        return FakeLLM(fake_script)

    resolved_provider = resolve_provider(provider)
    resolved_model = model_name(model, resolved_provider)

    if resolved_provider not in _PROVIDER_KEY_ENV:
        raise SystemExit(f"hands: unknown provider {resolved_provider!r} (expected 'anthropic' or 'openai')")

    key_env = _PROVIDER_KEY_ENV[resolved_provider]
    if not os.environ.get(key_env):
        raise SystemExit(
            f"hands: no LLM provider credentials found for provider={resolved_provider!r}: "
            f"the {key_env} environment variable is not set. Set it (e.g. in .env), or choose a "
            f"different provider via --provider or the HANDS_PROVIDER environment variable."
        )

    if resolved_provider == "openai":
        from hands.agent.llm import OpenAIProvider

        return OpenAIProvider(model=resolved_model)

    from hands.agent.llm import AnthropicProvider

    return AnthropicProvider(model=resolved_model)


def parse_inputs(pairs: list[str]) -> dict[str, str]:
    """Parse --input name=value. A value of the form @env:VAR is read from the
    environment so secrets never appear on the command line or in shell history."""
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--input expects name=value, got {p!r}")
        k, v = p.split("=", 1)
        if v.startswith("@env:"):
            var = v[5:]
            if var not in os.environ:
                raise SystemExit(f"--input {k}: environment variable {var} is not set")
            v = os.environ[var]
        out[k.strip()] = v
    return out
