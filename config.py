# -*- coding: utf-8 -*-
"""
项目统一配置：路径与本地模型。
使用本地 Milvus Lite 数据库，模型优先使用本地目录，便于后期接入本地模型。
"""
import os

# 项目根目录（multimodal_RAG/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# ---------- 数据与存储 ----------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR_PDF = os.path.join(DATA_DIR, "pdf-inputs")       # Web 上传 PDF 临时目录
INPUT_DIR_DETECT = os.path.join(DATA_DIR, "detect-inputs")  # 识别工具上传目录
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")     # 评估用 PDF 目录
EVALUATION_DATA_DIR = os.path.join(EVALUATION_DIR, "data")  # 评估数据（testset、img_captions 等）
RESULTS_DIR = os.path.join(DATA_DIR, "results")           # 评估结果 CSV 输出

# Milvus Lite：本地文件，无需启动服务
MILVUS_DB_PATH = os.path.join(DATA_DIR, "milvus.db")

# ---------- 模型目录（本地优先，不存在时回退到 HuggingFace）----------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 嵌入模型：本地目录名或 HuggingFace 模型 ID
EMBED_MODEL_NAME = "bge-m3"
def get_embed_model_path():
    p = os.path.join(MODELS_DIR, EMBED_MODEL_NAME)
    return p if os.path.isdir(p) else f"BAAI/{EMBED_MODEL_NAME}"

# 重排序模型
RERANKER_MODEL_NAME = "bge-reranker-large"
def get_reranker_model_path():
    p = os.path.join(MODELS_DIR, RERANKER_MODEL_NAME)
    return p if os.path.isdir(p) else f"BAAI/{RERANKER_MODEL_NAME}"

# Ollama 本地 LLM 模型名
OLLAMA_MODEL = "qwen2"
OLLAMA_REQUEST_TIMEOUT = 60.0

# 可选：多模态图像转表格（MiniCPM-V / InternVL）本地路径，为空则使用 HF 或跳过
CPM_MODEL_PATH = os.path.join(MODELS_DIR, "MiniCPM-V")  # 可选，按需下载
INTERNVL_MODEL_PATH = os.path.join(MODELS_DIR, "InternVL")  # 可选

# ---------- 评估默认参数 ----------
EVAL_PDF_START_ID = 3900889
EVAL_NUM_DOCS = 23

# ---------- 确保常用目录存在 ----------
def ensure_dirs():
    for d in (DATA_DIR, INPUT_DIR_PDF, INPUT_DIR_DETECT, RESULTS_DIR, EVALUATION_DIR, EVALUATION_DATA_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

ensure_dirs()
