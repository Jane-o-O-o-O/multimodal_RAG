# ragas 0.4.x版本API已变化，testset相关功能已移除
# from ragas.testset.generator import TestDataset
# from ragas.testset.evolutions import DataRow
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

def load_dataset(dataset_path: str) -> TestDataset:
    """
    Load the dataset from a CSV file.
    """

    assert dataset_path.endswith('.csv'), 'Dataset file must be a CSV file.'

    df = pd.read_csv(dataset_path,
                     quotechar='"',
                     skipinitialspace=True,)

    data_rows = []
    for _, row in df.iterrows():
        data_row = DataRow(
            question=row['question'],
            contexts=eval(row['contexts']),
            ground_truth=row['ground_truth'],
            evolution_type=row['evolution_type'],
            metadata=eval(row['metadata'])
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
    pdf_id = csv_path.split("/")[-2]
    img_ids = []
    documents = []

    for full_img_path in img_caption["image_path"]:
        img_id = full_img_path.split("/")[-1]
        img_ids.append(img_id)

    for doc in raw_docs:
        if doc.to_dict()["type"] in ["Table", "Image"]:
            doc_img_id = doc.to_dict()["metadata"]["image_path"].split("/")[-1]
            if doc_img_id in img_ids:
                caption = img_caption.loc[img_caption["image_path"] == f"./images/{pdf_id}/{doc_img_id}", "caption"].values[0]
                converted_doc = doc
                converted_doc.text = caption
                documents.append(converted_doc)
        else:
            documents.append(doc)

    return documents