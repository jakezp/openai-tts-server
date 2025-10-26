from typing import Generic, TypeVar

from huggingface_hub import ModelCardData
from pydantic import BaseModel

from speaches.api_types import ModelTask
from speaches.model_registry import ModelRegistry


ManagerT = TypeVar('ManagerT')
RegistryT = TypeVar('RegistryT', bound=ModelRegistry)


class Executor(BaseModel, Generic[ManagerT, RegistryT]):
    name: str
    model_manager: ManagerT
    model_registry: RegistryT
    task: ModelTask

    model_config = {"arbitrary_types_allowed": True}

    def can_handle_model(self, model_id: str, model_card_data: ModelCardData) -> bool:
        return self.model_registry.hf_model_filter.passes_filter(model_id, model_card_data)
