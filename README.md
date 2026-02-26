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

## 运行流程与已知问题

**Web 流程**：项目根目录执行 `streamlit run src/chatbot_web_demo/streamlit_app.py` → 侧栏选/上传文档 → 选文档后问答。依赖：Ollama 已启动且拉取 `config.OLLAMA_MODEL`；嵌入/重排见上表（无本地则用 HF）。

**评估流程**：项目根执行 `python src/evaluation.py`。会遍历 `config.EVAL_NUM_DOCS` 个 PDF（`data/evaluation/{id}.pdf`），每个需存在 `data/evaluation/data/{id}/testset.csv` 与 `img_captions.csv`；缺失的项会跳过并打日志。依赖：ragas 0.1.x（`pip install ragas==0.1.9`）。

**已知限制**：① Web 上传多文件时仅最后一个 PDF 会被建索引（单次上传建议单文件）。② 未配置 MiniCPM 时，上传带图/表的 PDF 会报错，需在 `config.CPM_MODEL_PATH` 放模型或后续改逻辑跳过图像转换。
