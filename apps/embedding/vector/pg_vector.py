from typing import List, Dict, Any
from django.db import transaction
from embedding.vector.base_vector import BaseVectorStore


class PGVector(BaseVectorStore):
    def _batch_save(self, data_list: List[Dict[str, Any]], embedding_model: Any, is_the_task_interrupted: bool) -> None:
        try:
            with transaction.atomic():
                for data in data_list:
                    if is_the_task_interrupted():
                        break
                    # 检查数据结构并获取内容
                    content = data.get('content', data.get('text', ''))
                    if not content:
                        print(f"Warning: No content found in data: {data}")
                        continue
                        
                    embeddings = embedding_model.embed_documents([content])
                    data['embedding'] = embeddings[0]
        except Exception as e:
            raise Exception(f"Error in batch save: {str(e)}")

    def delete_by_document_id(self, document_id: str) -> None:
        try:
            with transaction.atomic():
                # 从数据库中删除与文档ID相关的所有向量
                from embedding.models import Embedding
                Embedding.objects.filter(document_id=document_id).delete()
        except Exception as e:
            raise Exception(f"Error in delete_by_document_id: {str(e)}")

    def delete_by_paragraph_id(self, paragraph_id: str) -> None:
        try:
            with transaction.atomic():
                # 从数据库中删除与段落ID相关的所有向量
                from embedding.models import Embedding
                Embedding.objects.filter(paragraph_id=paragraph_id).delete()
        except Exception as e:
            raise Exception(f"Error in delete_by_paragraph_id: {str(e)}")
