
from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_REMEDIATION = 5
# Typhoon2 8B looped one remediation line ~25 times at temperature 0.1 without this.
REPEAT_PENALTY = 1.15


class LlmDraft(BaseModel):

    model_config = ConfigDict(extra="forbid")

    explanation_th: str
    remediation: list[str] = Field(min_length=1, max_length=MAX_REMEDIATION)

    @field_validator("remediation", mode="before")
    @classmethod
    def _dedupe_and_cap(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        unique = list(dict.fromkeys(item.strip() if isinstance(item, str) else item for item in value))
        return unique[:MAX_REMEDIATION]


def build_chat_model(config: dict[str, Any]) -> ChatOllama | ChatOpenAI:
    common = {
        "model": config["model"],
        "temperature": config["temperature"],
    }
    if config["backend"] == "ollama":
        return ChatOllama(
            **common,
            base_url=config["base_url"],
            num_predict=config["max_tokens"],
            repeat_penalty=REPEAT_PENALTY,
            client_kwargs={"timeout": config["timeout_s"]},
        )
    if config["backend"] == "llamacpp":
        return ChatOpenAI(
            **common,
            base_url=f"{config['base_url']}/v1",
            api_key="not-required",
            timeout=config["timeout_s"],
            max_completion_tokens=config["max_tokens"],
        )
    raise ValueError(f"Unsupported LLM backend: {config['backend']}")


def invoke_structured(
    prompt: str,
    config: dict[str, Any],
    model: ChatOllama | ChatOpenAI | Any | None = None,
) -> dict[str, Any]:
    chat_model = model or build_chat_model(config)
    method = "json_schema" if config["backend"] == "ollama" else "json_mode"
    structured_model = chat_model.with_structured_output(LlmDraft, method=method)
    result = structured_model.invoke(prompt)
    return LlmDraft.model_validate(result).model_dump()
