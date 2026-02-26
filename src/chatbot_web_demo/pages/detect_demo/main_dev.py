import streamlit as st
import os
import sys

# 保证可导入项目 config（运行 streamlit 时 cwd 通常为项目根）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config

from .table_detection.inference import detect_image
from .table_detection.inference_pdf import detect_pdf

if "uploaded_detect_file" not in st.session_state.keys():
    st.session_state["uploaded_detect_file"] = []

if "output_file_or_folder" not in st.session_state.keys():
    st.session_state["output_file_or_folder"] = []

DATA_DIR = config.DATA_DIR
INPUT_DIR = config.INPUT_DIR_DETECT
PROJECT_ROOT = config.PROJECT_ROOT

config.ensure_dirs()


def clear_dirs():
    # make sure the directories exist and no files are in them
    # So this is a bit of a hack, but it works for now
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    else:
        for file in os.listdir(INPUT_DIR):
            os.remove(os.path.join(INPUT_DIR, file))


def process_data():
    file = st.session_state["uploaded_detect_file"]
    if file:
        clear_dirs()
        file_name = file.name
        filepath = os.path.join(INPUT_DIR, file_name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
        if file.type == "application/pdf":
            os.makedirs(os.path.join(DATA_DIR, file_name.split(".")[0]), exist_ok=True)
            output_dir = os.path.join(DATA_DIR, file_name.split(".")[0])
            detect_pdf(filepath, output_dir)
            st.session_state["output_file_or_folder"] = output_dir
        else:
            output_file_name = file_name.split(".")[0] + "_detected.png"
            output_file_path = os.path.join(DATA_DIR, output_file_name)
            detect_image(filepath, output_file_path)
            st.session_state["output_file_or_folder"] = output_file_path
        st.success("识别成功！")


def visulize_img():
    output_file_or_folder = st.session_state.get("output_file_or_folder")
    if not output_file_or_folder:
        return
    if os.path.isdir(output_file_or_folder):
        all_files = os.listdir(output_file_or_folder)
        all_files_path = [os.path.join(output_file_or_folder, f) for f in all_files]
        all_files_captions = [f"{f} 识别结果" for f in all_files]
        st.image(all_files_path, width=200, caption=all_files_captions)
    else:
        st.image(output_file_or_folder, width=200, caption="识别结果")


def upload_data():
    uploaded_detect_file = st.file_uploader(
        "请上传您的文档或图像",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    st.session_state["uploaded_detect_file"] = uploaded_detect_file
    if uploaded_detect_file:
        with st.spinner("正在识别..."):
            process_data()


def detect_demo():
    st.header("识别工具")
    with st.sidebar:
        logo_path = os.path.join(PROJECT_ROOT, "assets", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)

    upload_data()
    visulize_img()
