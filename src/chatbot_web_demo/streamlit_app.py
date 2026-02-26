import streamlit as st

from pages.qa_demo.main_dev import qa_demo

# 可选页面：按需懒加载，缺失时不报错
def get_detect_demo():
    try:
        from pages.detect_demo.main_dev import detect_demo
        return detect_demo
    except Exception:
        return None

def get_doc_parse_demo():
    try:
        from pages.doc_parse_demo.main_dev import doc_parse_demo
        return doc_parse_demo
    except Exception:
        return None

page_names_to_funcs = {
    "智能问答": qa_demo,
}

# Try to add optional pages
detect_demo = get_detect_demo()
if detect_demo:
    page_names_to_funcs["识别工具"] = detect_demo

doc_parse_demo = get_doc_parse_demo()
if doc_parse_demo:
    page_names_to_funcs["研报解析"] = doc_parse_demo

selected_page = st.sidebar.selectbox(
        "选择页面", 
        page_names_to_funcs.keys(),
)
page_names_to_funcs[selected_page]()
