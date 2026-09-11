"""
LLM service selection.

`rag_engine` was written against the Ollama service directly, so setting
LLM_PROVIDER to anything else had no effect on chat -- requests still went to
localhost:11434 and failed when Ollama wasn't running.

This module exposes `get_llm_service()`, which returns either the original
Ollama service or an adapter around the provider factory's LangChain model. Both
expose the same `generate()` / `generate_stream()` interface, so rag_engine needs
no changes beyond swapping which service it asks for.
"""
from typing import AsyncGenerator, Optional

from loguru import logger

from app.config import settings


class LangChainLLMService:
    """Adapts a LangChain chat model to the Ollama service's interface."""

    def __init__(self):
        from app.providers.factory import get_llm
        self._llm = get_llm()
        logger.info(
            f"LLM service: {settings.LLM_PROVIDER} ({settings.LLM_MODEL})"
        )

    def _messages(self, prompt: str, system_prompt: Optional[str]):
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _bound(self, temperature: float, max_tokens: int):
        """Apply per-call overrides where the provider supports them."""
        try:
            return self._llm.bind(temperature=temperature, max_tokens=max_tokens)
        except Exception:
            return self._llm

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        try:
            result = await self._bound(temperature, max_tokens).ainvoke(
                self._messages(prompt, system_prompt)
            )
            content = getattr(result, "content", result)
            if isinstance(content, list):
                # Some providers return content blocks rather than a plain string.
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return (content or "").strip()
        except Exception as e:
            logger.error(f"LLM error ({settings.LLM_PROVIDER}): {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self._bound(temperature, max_tokens).astream(
                self._messages(prompt, system_prompt)
            ):
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in text
                    )
                if text:
                    yield text
        except Exception as e:
            logger.error(f"LLM streaming error ({settings.LLM_PROVIDER}): {e}")
            raise

    async def health_check(self) -> bool:
        try:
            await self.generate("ping", max_tokens=5)
            return True
        except Exception:
            return False


_llm_service = None


def get_llm_service():
    """
    Return the chat service matching LLM_PROVIDER.

    ollama -> the original local service; anything else -> the provider factory.
    """
    global _llm_service
    if _llm_service is None:
        if settings.LLM_PROVIDER == "ollama":
            from app.services.ollama import get_ollama_service
            _llm_service = get_ollama_service()
        else:
            _llm_service = LangChainLLMService()
    return _llm_service
