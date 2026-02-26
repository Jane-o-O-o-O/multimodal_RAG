import os
from typing import Sequence
from llama_index.core.response_synthesizers import Refine

def summarize_docs(text: Sequence[str]) -> str:
    """
    Summarize a list of documents using the Llama Index Refine Response Synthesizers.
    """
    summarizer = Refine(verbose=True)
    response = summarizer.get_response(
        "请你着重关注文档的数据部分，分点给出该文档的摘要(保留重要数据)，最后给出总结",
        text
    )
    return response

def save_summary(summary: str, document_path: str) -> None:
    """
    Save the summary to a file.
    """
    summary_filepath = os.path.join(document_path, "summary.txt")
    with open(summary_filepath, 'w', encoding='utf-8') as file:
        file.write(summary)
    print(f"Summary saved to {summary_filepath}")

def read_summary(document_path: str) -> str:
    """
    Read the summary from a file. If file does not exist, return a short placeholder.
    """
    summary_filepath = os.path.join(document_path, "summary.txt")
    if not os.path.exists(summary_filepath):
        return "（暂无摘要）"
    with open(summary_filepath, "r", encoding="utf-8") as file:
        return file.read()

