import cv2
import torch
from .ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def detect_image(image_path, output_file_name):

    assert image_path.endswith(".jpg") or image_path.endswith(
        ".png"
    ), "Input image should be a jpg or png file."

    image_path = image_path
    output_file_name = output_file_name
    # 使用相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "icdar19_configs", "cascade", "cascade_dit_base.yaml")
    model_path = os.path.join(current_dir, "icdar19_modern", "model.pth")
    opts = ["MODEL.WEIGHTS", model_path]

    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.MODEL.WEIGHTS = model_path
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference
    img = cv2.imread(image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0] == "icdar2019_test":
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

    output = predictor(img)["instances"]

    pred_classes = output.pred_classes
    pred_boxes = output.pred_boxes
    pred_score = output.scores
    print(f"******* pred_classes: {pred_classes} *******")
    print(f"******* pred_boxes:{pred_boxes} *******")
    print(f"******* pred_score:{pred_score} *******")

    # 获取边界框的坐标
    # bboxes = pred_boxes.tensor.cpu().numpy()
    bboxes = pred_boxes.tensor.cpu().numpy().astype(int)

    # 创建一个图像列表来保存裁剪的结果
    cropped_images = []
    # 遍历每个边界框
    for i, bbox in enumerate(bboxes):
        # 裁剪图像
        x0, y0, x1, y1 = bbox
        cropped_img = img[y0:y1, x0:x1]

        # 可选：保存裁剪的图像
        output_file_name_cropped = f"{output_file_name.split('.')[0]}_cropped_{i}.png"
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

    v = Visualizer(img[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # step 6: save
    cv2.imwrite(output_file_name, result_image)

    del predictor
    del cfg

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


