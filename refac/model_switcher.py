#!/usr/bin/env python3
"""
Funzioni per cambiare modello LLM e configurazione API a runtime
senza riavviare l'applicazione VaultRAG.

Refactored: works with VaultRagContext instead of module-level globals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core import VaultRagContext

log = logging.getLogger("vaultrag.model_switcher")

# Modelli predefiniti disponibili
AVAILABLE_MODELS = {
    "nvidia-free": {
        "model": "nemotron-3-super-120b-a12b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "",
    },
    "gemini-free": {
        "model": "google/gemini-2.0-flash-lite-001",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Google Gemini Flash Lite (gratuito via OpenRouter)",
    },
    "gemma-free": {
        "model": "google/gemma-4-31b-it:free",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Google Gemma 4 31B (gratuito via OpenRouter)",
    },
    "qwen-free": {
        "model": "qwen/qwen2.5-72b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Qwen 2.5 72B (gratuito via OpenRouter)",
    },
    "llama-free": {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Llama 3.3 70B (gratuito via OpenRouter)",
    },
    "qwen-plus": {
        "model": "qwen-plus",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Qwen-Plus (richiede credito DashScope)",
    },
}


def change_model(ctx: "VaultRagContext", model_name: str,
                 api_key: str | None = None, base_url: str | None = None) -> dict:
    """
    Cambia il modello LLM e/o la configurazione API a runtime.

    Args:
        ctx:         VaultRagContext instance
        model_name:  Nome del modello predefinito (es: "gemini-free")
                     oppure nome diretto del modello (es: "google/gemma-4-31b-it:free")
        api_key:     Nuova API key (opzionale, mantiene quella corrente se None)
        base_url:    Nuovo base URL (opzionale, mantiene quello corrente se None)

    Returns:
        Dict con status e configurazione corrente
    """
    # Determine if model_name is a preset or a direct name
    if model_name in AVAILABLE_MODELS:
        preset = AVAILABLE_MODELS[model_name]
        new_model = preset["model"]
        new_base_url = base_url or preset["base_url"]
        log.info(f"Preset: {model_name} -> {new_model}")
    else:
        new_model = model_name
        new_base_url = base_url or ctx.base_url
        log.info(f"Direct model: {new_model}")

    old_model = ctx.model

    # Check if client needs recreation
    client_changed = (
        (new_base_url != ctx.base_url)
        or (api_key is not None and api_key != ctx.api_key)
    )

    # Update context
    ctx.model = new_model
    if new_base_url:
        ctx.base_url = new_base_url
    if api_key:
        ctx.api_key = api_key

    # Recreate client if needed
    if client_changed:
        ctx.reset_llm_client()
        log.info(f"LLM client recreated (base_url={ctx.base_url})")

    log.info(f"Model changed: {old_model} -> {ctx.model}")

    return {
        "status": "ok",
        "previous_model": old_model,
        "current_model": ctx.model,
        "base_url": ctx.base_url,
        "client_recreated": client_changed,
    }


def list_available_models() -> list[dict]:
    """Lista dei modelli predefiniti disponibili."""
    return [
        {
            "key": key,
            "model": info["model"],
            "description": info["description"],
        }
        for key, info in AVAILABLE_MODELS.items()
    ]


def get_current_model(ctx: "VaultRagContext") -> dict:
    """Configurazione corrente del modello."""
    return {
        "model": ctx.model,
        "base_url": ctx.base_url,
        "api_key_configured": bool(ctx.api_key),
    }


def test_model(ctx: "VaultRagContext") -> dict:
    """Testa il modello corrente con una semplice chiamata."""
    import time

    try:
        start = time.time()
        response = ctx.get_llm().chat.completions.create(
            model=ctx.model,
            messages=[{"role": "user", "content": "Rispondi con una sola parola: ok"}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content or ""
        return {
            "status": "ok",
            "response": text.strip(),
            "latency_seconds": round(elapsed, 2),
            "model": ctx.model,
        }
    except Exception as exc:
        log.error(f"Test modello fallito: {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "model": ctx.model,
        }
