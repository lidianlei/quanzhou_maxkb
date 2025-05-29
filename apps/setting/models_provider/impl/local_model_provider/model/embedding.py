from typing import List, Dict, Any
from setting.models_provider.base_model_provider import MaxKBBaseModel
from sentence_transformers import SentenceTransformer
import torch
import os
import json


class LocalEmbedding(MaxKBBaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def new_instance(model_type: str, model_name: str, model_credential: Dict[str, object], **model_kwargs) -> 'LocalEmbedding':
        instance = LocalEmbedding()
        instance.model_type = model_type
        instance.model_name = model_name
        instance.model_credential = model_credential
        instance.model_path = "/opt/maxkb/model/base"
        return instance

    def _load_model(self):
        if self.model is None:
            try:
                print(f"Loading model from local path: {self.model_path}")
                if not os.path.exists(self.model_path):
                    raise Exception(f"Model path does not exist: {self.model_path}")
                
                # 使用本地模型文件
                self.model = SentenceTransformer(
                    model_name_or_path=self.model_path,
                    device=self.device,
                    cache_folder=self.model_path
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            self._load_model()
            print(f"Embedding {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error in embed_documents: {str(e)}")
            raise Exception(f"Error in embedding: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        try:
            self._load_model()
            print(f"Embedding query: {text[:50]}...")
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            print(f"Error in embed_query: {str(e)}")
            raise Exception(f"Error in embedding: {str(e)}")
