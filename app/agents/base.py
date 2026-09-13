"""
base.py

Tum agent'larin ortak iskeleti. Her agent bir LLMProvider ile kurulur ve
PipelineState'i alip guncellenmis halini dondurur; LangGraph node'u olarak bu run() baglanir.
"""

from abc import ABC, abstractmethod

from app.providers.base import LLMProvider
from app.state import PipelineState


class BaseAgent(ABC):
    """Somut agent'lar (Vision, Validation, RAG, Report...) bu sinifi miras alir."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """State'i okur, kendi alanlarini doldurur ve guncel state'i dondurur."""
