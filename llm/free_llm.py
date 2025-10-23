import os
from functools import lru_cache
from typing import Iterable, List, Optional, Union, Dict, Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def _get_device() -> int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return 0
    return -1


def _default_model_name() -> str:
    return os.getenv("FREE_LLM_MODEL", "google/flan-t5-base")


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=_get_device(),
    )


def _build_prompt(messages: Iterable[Dict[str, Any]]) -> str:
    prompt_segments: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if role == "system":
            prefix = "[System]"
        elif role == "assistant":
            prefix = "[Assistant]"
        else:
            prefix = "[User]"
        prompt_segments.append(f"{prefix}\n{content.strip()}")
    prompt_segments.append("[Assistant]\n")
    return "\n\n".join(prompt_segments)


def generate_text(
    prompt_or_messages: Union[str, Iterable[Dict[str, Any]]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
    else:
        prompt = _build_prompt(prompt_or_messages)

    selected_model = model_name or _default_model_name()
    generator = _load_pipeline(selected_model)
    sampling_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0.0,
    }
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        sampling_kwargs["repetition_penalty"] = repetition_penalty

    outputs = generator(prompt, **sampling_kwargs)
    text = outputs[0]["generated_text"]
    return text.strip()
