import cv2
import numpy as np
from .ditod import add_vit_config
import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import os
from PIL import Image
from pdf2image import convert_from_path
import PyPDF2

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def is_toc_page(page, toc_keywords):
    text = page.extract_text()
    for keyword in toc_keywords:
        if keyword in text:
            return True
    return False


def has_jump_elements(page):
    try:
        if "/Annots" in page and len(page["/Annots"]) > 0:
            return True
    except KeyError:
        pass
    return False


def filter_pdf(input_pdf_path, output_pdf_path, toc_keywords):
    pdf_reader = PyPDF2.PdfReader(input_pdf_path)
    pdf_writer = PyPDF2.PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        if page_num >= 8 or not (
            is_toc_page(page, toc_keywords) or has_jump_elements(page)
        ):
            pdf_writer.add_page(page)

    with open(output_pdf_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)


def detect_pdf(pdf_path, output_folder):

    assert pdf_path.endswith(".pdf"), "Input file must be a PDF file."

    pdf_path = pdf_path
    output_folder = output_folder
    # 使用相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "icdar19_configs", "cascade", "cascade_dit_base.yaml")
    model_path = os.path.join(current_dir, "icdar19_modern", "model.pth")
    opts = ["MODEL.WEIGHTS", model_path]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: convert PDF to images
    print(f"*******正在处理目录下{pdf_path}的文件******")

    # images = convert_from_path(pdf_path)
    toc_keywords = ["内容目录", "图表目录"]
    filtered_pdf_path = "filtered_" + os.path.basename(pdf_path)
    filter_pdf(pdf_path, filtered_pdf_path, toc_keywords)

    # print(f"*******正在处理目录下{filtered_pdf_path}的文件******")
    images = convert_from_path(filtered_pdf_path)

    # Step 6: run inference for each image and save the results
    for i, image in enumerate(images):
        print(f"===现在对page_{i+1}进行表格检测处理===")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 将原始的图像保存起来
        output_path_o = os.path.join(output_folder, f"detected_{i + 1}_o.png")
        cv2.imwrite(output_path_o, img)

        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if cfg.DATASETS.TEST[0] == "icdar2019_test":
            md.set(thing_classes=["table"])
        else:
            md.set(thing_classes=["text", "title", "list", "table", "figure"])

        output = predictor(img)["instances"]

        pred_classes = output.pred_classes
        pred_boxes = output.pred_boxes
        pred_score = output.scores
        # 获取边界框的坐标
        bboxes = pred_boxes.tensor.cpu().numpy().astype(int)
        # 创建一个图像列表来保存裁剪的结果
        cropped_images = []
        # 遍历每个边界框
        for j, bbox in enumerate(bboxes):
            # 裁剪图像
            x0, y0, x1, y1 = bbox
            cropped_img = img[y0:y1, x0:x1]

            # 可选：保存裁剪的图像
            output_file_name_cropped = os.path.join(
                output_folder, f"detected_{i + 1}_cropped_{j}.png"
            )
            cv2.imwrite(output_file_name_cropped, cropped_img)

            # 将裁剪的图像添加到列表中
            cropped_images.append(cropped_img)

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # 接下来就是对我们裁减后的表格做其他操作
        for cropped_img in cropped_images:
            pass
        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # 这里就是将原图作效果展示的
        v = Visualizer(
            img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
        )
        result = v.draw_instance_predictions(output.to("cpu"))
        result_image = result.get_image()[:, :, ::-1]

        # Step 7: save the result image
        output_path = os.path.join(output_folder, f"detected_{i+1}.png")
        cv2.imwrite(output_path, result_image)

    print(f"*******表格检测的图片结果，已经保存到{output_folder}文件中，请查看******")

    del predictor
    del cfg

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
