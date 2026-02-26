from FlagEmbedding import BGEM3FlagModel
from typing import List, Optional
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction


class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    """BGE-M3 稀疏嵌入，支持传入本地模型路径。"""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            try:
                import sys
                import os
                _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                import config
                model_path = config.get_embed_model_path()
            except Exception:
                model_path = "BAAI/bge-m3"
        self.model = BGEM3FlagModel(model_path, use_fp16=False)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result