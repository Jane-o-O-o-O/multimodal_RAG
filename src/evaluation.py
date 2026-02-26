import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

import config
from utils.data_preprocessing import (
    load_dataset,
    parse_pdf,
    convert_to_documents,
    load_img_captions,
)
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
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


# 稀疏嵌入函数（用于混合检索），使用 config 中的本地模型路径
class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self, model_path: str = None):
        model_path = model_path or config.get_embed_model_path()
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


if __name__ == "__main__":
    # 统一使用 config：本地 Milvus Lite + 本地模型路径
    res_path = config.RESULTS_DIR
    pdf_root_path = config.EVALUATION_DIR
    data_root_path = config.EVALUATION_DATA_DIR
    pdf_file_start_id = config.EVAL_PDF_START_ID
    config.ensure_dirs()

    llm = Ollama(
        model=config.OLLAMA_MODEL,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )
    embed_path = config.get_embed_model_path()
    reranker_path = config.get_reranker_model_path()
    embed_model = HuggingFaceEmbedding(embed_path)
    rerank = FlagEmbeddingReranker(model=reranker_path, top_n=5)

    Settings.llm = llm
    Settings.embed_model = embed_model

    metrics = [
        answer_similarity,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Milvus Lite：本地文件，无需服务
    milvus_db_path = config.MILVUS_DB_PATH
    sparse_fn = ExampleEmbeddingFunction()

    for i in tqdm(range(config.EVAL_NUM_DOCS), desc="Evaluating"):
        pdf_id = (i + pdf_file_start_id)
        document_path = os.path.join(pdf_root_path, str(pdf_id) + ".pdf")
        data_path = os.path.join(data_root_path, str(pdf_id))
        testset_path = os.path.join(data_path, "testset.csv")

        if not os.path.exists(document_path):
            tqdm.write(f"跳过 {pdf_id}: PDF 不存在 {document_path}")
            continue
        if not os.path.exists(testset_path):
            tqdm.write(f"跳过 {pdf_id}: 测试集不存在 {testset_path}")
            continue
        if not os.path.exists(os.path.join(data_path, "img_captions.csv")):
            tqdm.write(f"跳过 {pdf_id}: 缺少 data/{pdf_id}/img_captions.csv")
            continue

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