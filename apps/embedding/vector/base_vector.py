from typing import List, Dict, Any
from setting.models_provider import get_model_credential
from setting.models_provider.impl.local_model_provider.model.embedding import LocalEmbedding


class BaseVectorStore:
    def __init__(self):
        self.embedding_model = None

    def get_embedding_model(self, model_type: str, model_name: str, model_credential: Dict[str, Any]) -> Any:
        if self.embedding_model is None:
            self.embedding_model = LocalEmbedding.new_instance(
                model_type=model_type,
                model_name=model_name,
                model_credential=model_credential
            )
        return self.embedding_model

    def batch_save(self, data_list: List[Dict[str, Any]], embedding_model: Any, is_the_task_interrupted: bool) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model cannot be None")
        self._batch_save(data_list, embedding_model, is_the_task_interrupted)

    def _batch_save(self, data_list: List[Dict[str, Any]], embedding_model: Any, is_the_task_interrupted: bool) -> None:
        raise NotImplementedError
