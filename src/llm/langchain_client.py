
from __future__ import annotations

from typing import Any, Literal

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict


class LlmDraft(BaseModel):

    model_config = ConfigDict(extra="forbid")

    priority: Literal["HIGH", "MEDIUM", "LOW"]
    explanation_th: str
    remediation: list[str]


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
