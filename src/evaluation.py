import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

from mmRAG.utils.data_preprocessing import (
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
from ragas.integrations.llama_index import evaluate
from ragas.metrics import (
    answer_similarity,
    answer_relevancy,
    context_precision,
    context_recall,
)
from tqdm import tqdm


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
    res_path = '/home/project/data/jc/mmRAG/mmRAG/data/pls_convertor'
    pdf_root_path = '/home/project/data/jc/mmRAG/evaluation'
    data_root_path = os.path.join(pdf_root_path, 'data')
    pdf_file_start_id = 3900889

    llm = Ollama(model="qwen2", request_timeout=60.0)
    embed_path = "/home/project/data/jc/mmRAG/model/bge-m3"
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