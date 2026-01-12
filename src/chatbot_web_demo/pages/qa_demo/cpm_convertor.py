from .system_prompt import TABLES_AND_CHARTS_CONVERT_SYSTEM
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Optional

# Use relative path from project root

)
CPM_MODEL_PATH = os.path.j5")lama3-V-2_PM-L", "MiniCodelT_ROOT, "moin(PROJEC', '..')'..', '....',  'SCRIPT_DIR,path.join(path(os.abs = os.path.T_ROOTPROJECh(__file__)).path.abspatrname(osos.path.di = PT_DIRSCRI


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
        model_name_or_path: Optional[str] = CPM_MODEL_PATH,
        device: Optional[str] = "cuda:1" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device},
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
