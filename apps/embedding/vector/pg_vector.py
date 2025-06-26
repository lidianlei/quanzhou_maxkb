from typing import List, Dict, Any
from django.db import transaction
from embedding.vector.base_vector import BaseVectorStore
import numpy as np
import json


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

    def delete_by_dataset_id(self, dataset_id: str) -> None:
        try:
            with transaction.atomic():
                # 从数据库中删除与数据集ID相关的所有向量
                from embedding.models import Embedding
                Embedding.objects.filter(dataset_id=dataset_id).delete()
        except Exception as e:
            raise Exception(f"Error in delete_by_dataset_id: {str(e)}")

    def delete_by_source_id(self, source_id: str, source_type: str) -> None:
        try:
            with transaction.atomic():
                from embedding.models import Embedding
                Embedding.objects.filter(source_id=source_id, source_type=source_type).delete()
        except Exception as e:
            raise Exception(f"Error in delete_by_source_id: {str(e)}")

    def delete_by_source_ids(self, source_ids: list, source_type: str) -> None:
        try:
            with transaction.atomic():
                from embedding.models import Embedding
                Embedding.objects.filter(source_id__in=source_ids, source_type=source_type).delete()
        except Exception as e:
            raise Exception(f"Error in delete_by_source_ids: {str(e)}")

    def save(self, text, is_active, source_type, source_id, document_id, paragraph_id, dataset_id, embedding):
        try:
            from embedding.models import Embedding
            from dataset.models.data_set import Document, Paragraph, DataSet
            import uuid
            # 生成embedding向量
            embeddings = embedding.embed_documents([text])
            embedding_vector = embeddings[0]
            # 检查是否有 NaN 或 None
            if embedding_vector is None or np.any(np.isnan(embedding_vector)):
                raise Exception("Embedding vector contains NaN or None, cannot save to database.")
            # 创建Embedding对象
            emb = Embedding(
                id=str(uuid.uuid1()),
                source_id=source_id,
                source_type=source_type,
                is_active=is_active,
                dataset_id=dataset_id,
                document_id=document_id,
                paragraph_id=paragraph_id,
                embedding=embedding_vector,
                meta={},
            )
            emb.save()
        except Exception as e:
            raise Exception(f"Error in save: {str(e)}")

    def update_by_paragraph_id(self, paragraph_id: str, update_fields: dict) -> None:
        try:
            with transaction.atomic():
                from embedding.models import Embedding
                Embedding.objects.filter(paragraph_id=paragraph_id).update(**update_fields)
        except Exception as e:
            raise Exception(f"Error in update_by_paragraph_id: {str(e)}")

    def hit_test(self, query_text, dataset_id, exclude_document_id_list, top_number, similarity, search_mode, embedding):
        try:
            from embedding.models import Embedding
            # 生成查询向量
            query_vector = embedding.embed_documents([query_text])[0]
            if query_vector is None or np.any(np.isnan(query_vector)):
                raise Exception("Query embedding vector contains NaN or None, cannot search.")
            # 构建基础查询
            qs = Embedding.objects.filter(dataset_id__in=dataset_id)
            if exclude_document_id_list:
                qs = qs.exclude(document_id__in=exclude_document_id_list)
            # 计算相似度（假设embedding字段为向量，使用余弦相似度）
            results = []
            for emb in qs:
                emb_vec = emb.embedding
                # 如果是字符串，先转为list
                if isinstance(emb_vec, str):
                    try:
                        emb_vec = json.loads(emb_vec)
                    except Exception:
                        continue  # 跳过无法解析的向量
                emb_vec = np.array(emb_vec, dtype=np.float32)
                if emb_vec is None or np.any(np.isnan(emb_vec)):
                    continue
                # 计算余弦相似度
                sim = float(np.dot(query_vector, emb_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(emb_vec)))
                if sim >= similarity:
                    results.append((emb, sim))
            # 排序并取top_n
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_number]
            return [
                {
                    "paragraph_id": emb.paragraph_id if hasattr(emb, 'paragraph_id') else None,
                    "similarity": sim,
                    "comprehensive_score": sim,  # 如有更复杂评分可替换
                    "id": emb.id
                }
                for emb, sim in results
            ]
        except Exception as e:
            raise Exception(f"Error in hit_test: {str(e)}")
