from .system_prompt import TABLES_AND_CHARTS_CONVERT_SYSTEM
import os
import sys
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Optional

# 本地模型路径：优先使用 config，便于后期接入模型
def _get_cpm_model_path():
    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_dir, "..", "..", ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        import config
        p = config.CPM_MODEL_PATH
        return p if os.path.isdir(p) else None
    except Exception:
        return None

# 工厂使用时可传入路径，否则用 config；未配置则 get_instance 时需传入
CPM_MODEL_PATH = _get_cpm_model_path()


class CPMConvertorFactory:
    _instance = None

    @staticmethod
    def get_instance(device="cuda:1" if torch.cuda.is_available() else "cpu"):
        """Static method to get the singleton instance of CPMConvertor."""
        if CPMConvertorFactory._instance is None:
            CPMConvertorFactory._instance = CPMConvertor(device=device)
        return CPMConvertorFactory._instance


class CPMConvertor:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path or CPM_MODEL_PATH
        if not self.model_name_or_path:
            raise ValueError(
                "未配置 MiniCPM 模型路径。请在项目 config.py 中设置 CPM_MODEL_PATH，"
                "或将模型放到 models/MiniCPM-V 目录。"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": self.device},
            torch_dtype=torch.float16,
        ).eval()

    def convert(
        self,
        img_path: str,
        query: str,
        sampling=True,
        temperature=0.1,
        system_prompt: str = TABLES_AND_CHARTS_CONVERT_SYSTEM,
    ):
        assert (
            img_path.endswith(".png")
            or img_path.endswith(".jpg")
            or img_path.endswith(".jpeg")
        ), "Only support png, jpg, jpeg format"
        assert os.path.exists(img_path), f"{img_path} does not exist"

        image = Image.open(img_path).convert("RGB")
        msgs = [{"role": "user", "content": query}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        return res

    def clear_GPU_mem(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
