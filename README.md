# multimodal_RAG

## Introduction
> TODO


## Get Started

Follow these steps to get started with this project:

### 1. Install Dependencies

First, install all the necessary Python dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 2. Milvus Lite（单容器/本地文件模式）

Milvus Lite 以本地文件形式运行，无需启动服务或开放 19530 端口。示例 URI：`file:./data/milvus_lite.db`。

> 如果需要传统 Milvus 服务，可以参考 `dependencies/milvus`，但单容器模式建议使用 Lite。

### 3. Start Nebula Graph

Finally, start Nebula Graph by navigating to the corresponding directory and running the installation script:

```bash
cd dependencies/nebulaGraph
bash install.sh
```

### 4. Run the Code
- 顶层依赖安装：`pip install -r requirements.txt`
- Streamlit Demo：`pip install -r src/chatbot_web_demo/requirements.txt`，然后 `streamlit run src/chatbot_web_demo/streamlit_app.py`
- 评估脚本：`python src/evaluation.py`（默认使用本地 Milvus Lite 文件）


## Acknowledgements

This work is built with reference to the code of the following projects:

- [Milvus](https://github.com/milvus-io/milvus)
- [Nebula Graph](https://github.com/vesoft-inc/nebula)
- [RAGAs](https://github.com/explodinggradients/ragas)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM-V)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Ollama](https://github.com/ollama/ollama)
- [BEIR](https://github.com/beir-cellar/beir)

Thanks for their awesome work!

