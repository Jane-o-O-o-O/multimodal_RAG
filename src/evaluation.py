import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from utils.data_preprocessing import (
    load_dataset,
    parse_pdf,
    convert_to_documents,
    load_img_captions
)
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext
)
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from ragas.integrations.llama_index import evaluate
from ragas.metrics import (
    answer_similarity,
    answer_relevancy,
    context_precision,
    context_recall,
)
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from typing import List


# 稀疏嵌入函数（用于混合检索）
class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

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


# def ragas_evaluation(
#     query_engine,
#     metrics,
#     dataset,
#     llm,
#     embeddings,
#     raise_exceptions
# ):
#     result = evaluate(query_engine,
#                       metrics,
#                       dataset,
#                       llm,
#                       embeddings,
#                       raise_exceptions)
#     return result.to_pandas()

def beir_evaluation():
    pass


if __name__ == '__main__':
    # 使用相对路径，基于项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    res_path = os.path.join(project_root, 'data', 'results')
    pdf_root_path = os.path.join(project_root, 'data', 'evaluation')
    data_root_path = os.path.join(pdf_root_path, 'data')
    pdf_file_start_id = 3900889

    # 确保结果目录存在
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(pdf_root_path, exist_ok=True)

    llm = Ollama(model="qwen2", request_timeout=60.0)
    # 模型路径改为相对路径，或者使用HuggingFace模型名称
    embed_path = os.path.join(project_root, 'models', 'bge-m3')  # 本地模型
    # 或者直接使用 HuggingFace 模型: embed_path = "BAAI/bge-m3"
    embed_model = HuggingFaceEmbedding(embed_path)
    rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)

    Settings.llm = llm
    Settings.embed_model = embed_model

    metrics = [
        answer_similarity,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Milvus Lite: local file mode (no Docker, no server port)
    # This creates/uses a local SQLite-backed Milvus Lite instance at the path
    milvus_db_path = os.path.join(data_root_path, "milvus.db")
    
    # 初始化稀疏嵌入函数
    sparse_fn = ExampleEmbeddingFunction()

    for i in tqdm(range(23), desc='Evaluating'):
        pdf_id = (i + pdf_file_start_id)
        document_path = os.path.join(pdf_root_path, str(pdf_id) + '.pdf')
        data_path = os.path.join(data_root_path, str(pdf_id))

        vector_store = MilvusVectorStore(
            uri=milvus_db_path,
            collection_name='doc' + str(pdf_id) + 'ImgConv',
            dim=1024,
            overwrite=True,
            enable_sparse=True,
            sparse_embedding_function=sparse_fn,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Naive RAG (baseline)
        # raw_docs = parse_pdf(document_path)
        # documents = convert_to_documents(raw_docs)

        # Naive RAG + Image Convertor
        raw_docs = parse_pdf(document_path, extract_images_in_pdf=True)
        docs = load_img_captions(raw_docs=raw_docs,
                                 csv_path=os.path.join(data_path, 'img_captions.csv'))
        documents = convert_to_documents(docs)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        query_engine = index.as_query_engine(
            similarity_top_k=10, node_postprocessors=[rerank]
        )

        dataset = load_dataset(os.path.join(data_path, 'testset.csv'))

        result = evaluate(
            query_engine = query_engine,
            metrics = metrics,
            dataset = dataset,
            llm = llm,
            embeddings = embed_model,
            raise_exceptions = False
        )

        result.to_pandas().to_csv(os.path.join(res_path, str(pdf_id) + '.csv'))