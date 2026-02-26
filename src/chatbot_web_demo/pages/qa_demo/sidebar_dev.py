import streamlit as st
from .ui_dev import clear_query_history
from .hybrid_dev import ExampleEmbeddingFunction
from .system_prompt import EXPERT_Q_AND_A_SYSTEM
from .data_preprocessing import parse_pdf, convert_to_documents, convert_img_to_tables
from .summary_utils import summarize_docs, save_summary

import os
import sys
from io import StringIO
from pymilvus import connections
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama

import logging

logging.basicConfig(level=logging.INFO)

# 使用项目统一 config（本地 Milvus Lite + 本地模型路径）
# sidebar_dev 在 src/chatbot_web_demo/pages/qa_demo/，需 4 层 .. 到项目根
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import config

DATA_DIR = config.DATA_DIR
INPUT_DIR = config.INPUT_DIR_PDF
MILVUS_DB_PATH = config.MILVUS_DB_PATH
config.ensure_dirs()

# Initialize Milvus Lite connection (once per session)
@st.cache_resource(show_spinner=False)
def init_milvus_connection():
    """Connect to Milvus Lite (local file mode)."""
    connections.connect(
        alias="default",
        uri=MILVUS_DB_PATH,  # Local file: milvus.db
    )
    return True

init_milvus_connection()


def clear_dirs():
    # make sure the directories exist and no files are in them
    # So this is a bit of a hack, but it works for now
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    else:
        for file in os.listdir(INPUT_DIR):
            os.remove(os.path.join(INPUT_DIR, file))


@st.cache_resource(show_spinner=False)
def get_embed_model(embed_path):
    logging.info(f"Loading {embed_path}")
    return HuggingFaceEmbedding(model_name=embed_path, device="cpu")


@st.cache_resource(show_spinner=False)
def get_reranker():
    return FlagEmbeddingReranker(model=config.get_reranker_model_path(), top_n=5)


@st.cache_resource(show_spinner=False)
def get_sparse_fn():
    return ExampleEmbeddingFunction(model_path=config.get_embed_model_path())


@st.cache_resource(show_spinner=False)
def get_models():
    llm = Ollama(
        model=config.OLLAMA_MODEL,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )
    embed_model = get_embed_model(embed_path=config.get_embed_model_path())
    reranker = get_reranker()
    return llm, embed_model, reranker


def load_model():
    llm, embed_model, reranker = get_models()
    Settings.llm = llm
    Settings.embed_model = embed_model
    st.session_state["llm"] = llm
    st.session_state["embed_model"] = embed_model
    st.session_state["reranker"] = reranker


def get_milvus_collections_list():
    """List all collections in Milvus Lite."""
    from pymilvus import list_collections
    try:
        colls = list_collections()
        st.session_state["milvus_collections"] = sorted(colls)
    except Exception as e:
        st.session_state["milvus_collections"] = []
        logging.warning(f"Failed to list collections: {e}")


def reset_engine():
    # st.session_state.clear()
    st.session_state["is_ready"] = False
    get_milvus_collections_list()


if "uploaded_files" not in st.session_state.keys():
    st.session_state["uploaded_files"] = []

if "is_ready" not in st.session_state.keys():
    st.session_state["is_ready"] = False

if "milvus_collections" not in st.session_state:
    st.session_state["milvus_collections"] = []

if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1


# @st.cache_data
def choose_docs():
    get_milvus_collections_list()
    pdf_doc_list = []
    for collection in st.session_state["milvus_collections"]:
        if collection.startswith("doc_"):
            pdf_doc_list.append(collection.split("_")[1] + ".pdf")
    selected_doc = st.selectbox(
        "选择文档",
        pdf_doc_list,
        placeholder="请选择文档" if pdf_doc_list else "暂无文档，请先上传",
    )
    if selected_doc:
        st.session_state["selected_doc"] = "doc_" + selected_doc.split(".")[0]
    elif not pdf_doc_list:
        st.session_state["selected_doc"] = None
        st.session_state["is_ready"] = False


# @st.cache_data cannot be used opon st.file_uploader
def upload_data():
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1
    uploaded_files = st.file_uploader(
         "请上传您的文档",
        on_change=reset_engine,
        accept_multiple_files=True,
        type=["pdf", "md", "txt"],
        key=str(st.session_state["uploader_key"]) + "/file_uploader"
    )
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        process_data()
        st.session_state["uploader_key"] += 1
        st.rerun()

        # st.session_state["seleted_doc"] = st.session_state["uploaded_file_name"]


def parse_data():
    documents = []
    for uploaded_file in st.session_state["uploaded_files"]:
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                clear_dirs()
                filepath = f"{INPUT_DIR}/{uploaded_file.name}"
                file_id = uploaded_file.name.split(".")[0]
                os.makedirs(os.path.join(DATA_DIR, file_id), exist_ok=True)
                file_data_path = os.path.join(DATA_DIR, file_id)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                print(f"Processing {filepath}")
                raw_docs = parse_pdf(
                    filepath,
                    extract_image_block_output_dir=os.path.join(
                        file_data_path, "images"
                    ),
                    extract_images_in_pdf=True,
                )
                docs = convert_img_to_tables(raw_docs, file_data_path)
                documents, text_seq = convert_to_documents(docs)
                summary = summarize_docs(text_seq)
                save_summary(summary, file_data_path)
            else:
                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # To read file as string:
                string_data = stringio.read()
                documents.append(
                    Document(
                        text=string_data, metadata={"file_name": uploaded_file.name}
                    )
                )
        st.session_state["uploaded_file_name"] = uploaded_file.name

    return documents


def create_vector_index(documents):
    collection_name = st.session_state["uploaded_file_name"].split(".")[0]
    # 使用 Settings 而不是 ServiceContext (已弃用)
    # Settings 已经在 load_model() 中设置

    # Milvus Lite: local file mode
    vector_store = MilvusVectorStore(
        uri=MILVUS_DB_PATH,
        collection_name=f"doc_{collection_name}",
        dim=1024,
        overwrite=True,
        enable_sparse=True,
        sparse_embedding_function=get_sparse_fn(),
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def build_query_engine_from_db(collection_name):
    # Milvus Lite: load from local file
    vector_store = MilvusVectorStore(
        uri=MILVUS_DB_PATH,
        collection_name=collection_name,
        dim=1024,
        overwrite=False,
        enable_sparse=True,
        sparse_embedding_function=get_sparse_fn(),
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": 60},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[st.session_state["reranker"]]
    )
    return query_engine


def build_query_engine_from_index(index):
    rerank = FlagEmbeddingReranker(model=config.get_reranker_model_path(), top_n=5)
    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[st.session_state["reranker"]]
    )

    return query_engine


def success_message():
    st.success("文档加载成功！", icon="👉")


def success_doc_processing_message():
    st.success("文档处理成功！现在，您可在上方选择文档后输入问题进行查询！", icon="✅")


def init_engine():
    st.session_state["is_ready"] = True


def build_query_engine():
    if st.session_state["selected_doc"]:
        collection_name = st.session_state["selected_doc"]
        query_engine = build_query_engine_from_db(collection_name)
        st.session_state["query_engine"] = query_engine
        init_engine()
        success_message()


def process_data():
    with st.sidebar:
        with st.spinner("处理中"):
            documents = parse_data()
            index = create_vector_index(documents)

            # query_engine = build_query_engine_from_index(index)
            # st.session_state["query_engine"] = query_engine
            st.session_state["uploaded_files"] = []
            get_milvus_collections_list()
            success_doc_processing_message()


def sidebar():
    with st.sidebar:
        logo_path = os.path.join(PROJECT_ROOT, "assets", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        st.markdown(
            "## 指引\n"
            "1. 上传您的文档或选择知识库中已有的文档📄\n"
            "2. 输入您想问的问题💬\n"
        )
        load_model()
        choose_docs()
        upload_data()

        st.button("清除聊天记录", on_click=clear_query_history)
