# ragas 0.1.x 评估用：从 CSV 加载 testset
try:
    from ragas.testset.generator import TestDataset
    from ragas.testset.evolutions import DataRow
except ImportError:
    TestDataset = None
    DataRow = None

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from llama_index.core import Document
from typing import Optional
import pandas as pd
import os
from tqdm import tqdm

def parse_pdf(
    pdf_path: str,
    extract_image_block_output_dir: Optional[str] = None,
    extract_images_in_pdf: bool = False
):
    """
    Parse a PDF file using the `unstructured` library.
    """

    assert os.path.exists(pdf_path), f"PDF file {pdf_path} does not exist"
    if extract_images_in_pdf:
        return partition_pdf(pdf_path,
                             strategy='hi_res',
                             extract_images_in_pdf=True,
                             extract_image_block_types=["Image", "Table"],
                             extract_image_block_to_payload=False,
                             extract_image_block_output_dir=extract_image_block_output_dir
        )
    else:
        return partition_pdf(pdf_path,
                             strategy='hi_res',
                             extract_images_in_pdf=False
        )

def convert_to_documents(documents,
              max_characters=512,
              overlap=50):
    """
    convert the partitioned documents to llamaindex Document objects.
    """

    chunks = chunk_elements(documents,
                            max_characters=max_characters,
                            overlap=overlap)
    documents = []
    for chunk in tqdm(chunks, desc="Converting to documents"):
        document = Document(
            doc_id=chunk.to_dict()["element_id"],
            text=chunk.to_dict()["text"],
            metadata={"page_number": chunk.to_dict()["metadata"]["page_number"],
                  "filename": chunk.to_dict()["metadata"]["filename"]}
        )
        documents.append(document)

    return documents

def load_dataset(dataset_path: str):
    """
    从 CSV 加载 RAGAs 评估用数据集。
    CSV 需包含列: question, contexts, ground_truth, evolution_type, metadata。
    返回 ragas evaluate() 所需的 dict 格式；若 ragas 未安装或 API 不可用则返回 None。
    """
    if TestDataset is None or DataRow is None:
        raise ImportError(
            "load_dataset 需要 ragas 0.1.x 的 TestDataset/DataRow。"
            "请安装: pip install ragas==0.1.9"
        )
    assert dataset_path.endswith(".csv"), "Dataset file must be a CSV file."
    assert os.path.exists(dataset_path), f"Dataset file not found: {dataset_path}"

    df = pd.read_csv(dataset_path, quotechar='"', skipinitialspace=True)
    required = ["question", "contexts", "ground_truth"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {col}. 当前列: {list(df.columns)}")

    data_rows = []
    for _, row in df.iterrows():
        contexts = row["contexts"]
        if isinstance(contexts, str):
            contexts = eval(contexts)
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = eval(metadata) if metadata.strip() else {}
        evolution_type = row.get("evolution_type", "single_turn")
        data_row = DataRow(
            question=row["question"],
            contexts=contexts,
            ground_truth=row["ground_truth"],
            evolution_type=evolution_type,
            metadata=metadata,
        )
        data_rows.append(data_row)

    test_dataset = TestDataset(test_data=data_rows).to_dataset().to_dict()
    return test_dataset

def load_img_captions(raw_docs,
                     csv_path):
    """
    Load the image captions from a CSV file.
    """

    img_caption = pd.read_csv(csv_path)
    pdf_id = os.path.normpath(csv_path).split(os.sep)[-2]
    img_ids = []
    documents = []

    for full_img_path in img_caption["image_path"]:
        img_id = os.path.basename(os.path.normpath(str(full_img_path)))
        img_ids.append(img_id)

    for doc in raw_docs:
        if doc.to_dict()["type"] in ["Table", "Image"]:
            doc_img_id = os.path.basename(os.path.normpath(doc.to_dict()["metadata"]["image_path"]))
            if doc_img_id in img_ids:
                caption = img_caption.loc[img_caption["image_path"] == f"./images/{pdf_id}/{doc_img_id}", "caption"].values[0]
                converted_doc = doc
                converted_doc.text = caption
                documents.append(converted_doc)
        else:
            documents.append(doc)

    return documents