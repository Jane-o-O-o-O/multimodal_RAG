# multimodal_RAG

多模态 RAG：PDF 解析、图文混合检索（BGE-M3 + Milvus Lite）、本地 LLM 问答与 RAGAs 评估。

---

## 快速开始

```bash
# 安装
pip install -r requirements.txt
pip install -r src/chatbot_web_demo/requirements.txt

# Web 演示（项目根目录执行）
streamlit run src/chatbot_web_demo/streamlit_app.py

# 评估
python src/evaluation.py
```

---

## 配置

所有路径与模型在项目根目录 **`config.py`** 中配置：

| 配置项 | 说明 | 默认 |
|--------|------|------|
| `MILVUS_DB_PATH` | Milvus Lite 库文件 | `data/milvus.db` |
| `MODELS_DIR` | 本地模型根目录 | `models/` |
| `get_embed_model_path()` | 嵌入模型（如 bge-m3） | 本地 `models/bge-m3` 或 HF |
| `get_reranker_model_path()` | 重排模型 | 本地或 HF |
| `OLLAMA_MODEL` | Ollama 模型名 | `qwen2` |
| `CPM_MODEL_PATH` | 图表转表格（MiniCPM，可选） | `models/MiniCPM-V` |

修改模型或路径时只改 `config.py` 即可。可选依赖 Nebula Graph 见 `dependencies/nebulaGraph`。

---

## 本地模型部署与下载

项目默认从 **`models/`**（即 `config.MODELS_DIR`）读取本地模型；目录不存在时自动回退到 HuggingFace。建议在项目根下建好 `models/` 再按需下载。

### 1. 嵌入模型（必选，检索用）

- **接口**：`config.get_embed_model_path()`，默认对应目录 `models/bge-m3`。
- **HuggingFace**：`BAAI/bge-m3`。

```bash
# 项目根目录执行，将模型下载到 models/bge-m3
pip install huggingface_hub
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3 --local-dir-use-symlinks False
```

或在 Python 中：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="BAAI/bge-m3", local_dir="models/bge-m3", local_dir_use_symlinks=False)
```

### 2. 重排序模型（必选）

- **接口**：`config.get_reranker_model_path()`，默认目录 `models/bge-reranker-large`。
- **HuggingFace**：`BAAI/bge-reranker-large`。

```bash
huggingface-cli download BAAI/bge-reranker-large --local-dir models/bge-reranker-large --local-dir-use-symlinks False
```

### 3. 大模型 / 问答（Ollama，必选）

- **接口**：`config.OLLAMA_MODEL`（默认 `qwen2`），由 Ollama 本地服务加载，不放在 `models/` 下。

1. 安装 [Ollama](https://ollama.com/) 并启动服务。
2. 拉取与配置一致的模型名，例如：

```bash
ollama pull qwen2
```

若改用其他模型，在 `config.py` 中修改 `OLLAMA_MODEL` 即可（如 `qwen:14b` 则执行 `ollama pull qwen:14b`）。

### 4. 图表转表格（可选，上传带图/表 PDF 时需要）

- **接口**：`config.CPM_MODEL_PATH`，默认目录 `models/MiniCPM-V`。
- 用于将 PDF 内表格/图片转为文本，未部署时上传含图表的 PDF 会报错。

从 HuggingFace 下载 MiniCPM-V 到 `models/MiniCPM-V`（具体仓库名以官方为准，如 `openbmb/MiniCPM-V-2` 等），或自行将已有权重放到该目录，并保证目录结构与代码中 `from_pretrained(local_dir)` 的预期一致。

### 目录与 config 对应关系

| 本机目录（项目根下） | config 用途 |
|----------------------|-------------|
| `models/bge-m3` | 嵌入，`get_embed_model_path()` |
| `models/bge-reranker-large` | 重排，`get_reranker_model_path()` |
| `models/MiniCPM-V` | 图表转表格，`CPM_MODEL_PATH`（可选） |

嵌入/重排未放到上述目录时，程序会自动使用 HuggingFace 上的同名模型（需网络）。Ollama 模型名与 `config.OLLAMA_MODEL` 保持一致即可。

---

## 运行流程与已知问题

**Web 流程**：项目根目录执行 `streamlit run src/chatbot_web_demo/streamlit_app.py` → 侧栏选/上传文档 → 选文档后问答。依赖：Ollama 已启动且拉取 `config.OLLAMA_MODEL`；嵌入/重排见上表（无本地则用 HF）。

**评估流程**：项目根执行 `python src/evaluation.py`。会遍历 `config.EVAL_NUM_DOCS` 个 PDF（`data/evaluation/{id}.pdf`），每个需存在 `data/evaluation/data/{id}/testset.csv` 与 `img_captions.csv`；缺失的项会跳过并打日志。依赖：ragas 0.1.x（`pip install ragas==0.1.9`）。

**已知限制**：① Web 上传多文件时仅最后一个 PDF 会被建索引（单次上传建议单文件）。② 未配置 MiniCPM 时，上传带图/表的 PDF 会报错，需在 `config.CPM_MODEL_PATH` 放模型或后续改逻辑跳过图像转换。
