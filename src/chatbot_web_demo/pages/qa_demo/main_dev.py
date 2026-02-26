import os
import streamlit as st

from .sidebar_dev import (
    sidebar,
    build_query_engine,
    get_milvus_collections_list,
    DATA_DIR,
    PROJECT_ROOT
)
from .ui_dev import clear_query_history
from .summary_utils import read_summary

def qa_demo():
    # Support both Linux and Windows; fallback to placeholder if asset not found
    logo_path = os.path.join(PROJECT_ROOT, 'assets', 'logo.jpg')
    if not os.path.exists(logo_path):
        logo_path = None  # Will skip image rendering if missing

    st.header("国泰智能问答")
    st.header("GuoTai AI Q&A:book:")


    if "is_ready" not in st.session_state.keys():
        st.session_state['is_ready'] = False

    get_milvus_collections_list()
    sidebar()

    build_query_engine()


    if st.session_state.get("is_ready") and st.session_state.get("selected_doc"):
        # selected_doc 形如 "doc_文件名"，取前缀后的部分作为 doc_id（支持中文等任意文件名）
        raw = st.session_state["selected_doc"]
        current_doc_id = raw.replace("doc_", "", 1) if raw.startswith("doc_") else raw
        current_doc = f"{current_doc_id}.pdf"
        current_doc_path = os.path.join(DATA_DIR, current_doc_id)
        summary = read_summary(current_doc_path)
        st.write("当前文档：", current_doc)
        st.write(summary)
        st.markdown("---")
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "有什么能够帮到您？"}]

        for message in st.session_state.messages:
            avatar = logo_path if (message["role"] == "assistant" and logo_path) else ('🧑‍💻' if message["role"] == "user" else None)
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar='🧑‍💻'):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant", avatar=('🤖' if logo_path else None)):
                with st.spinner("Thinking ... "):
                    resp = st.session_state['query_engine'].query(prompt)
                    response, sources = resp.response, resp.source_nodes

                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

                st.markdown("-------------------")
                for idx in range(len(sources)):
                    # st.write(f"源文档 {idx+1}:\n{sources[idx].text}")
                    st.write(f"源文档 **{idx+1}**:")
                    st.write(f"{sources[idx].text}")
                    # st.write(f"相关得分: {sources[idx].score}")
                    page_number = sources[idx].metadata.get('page_number', "1")
                    st.write(f"源页码: **{page_number}**")
                    st.markdown("-------------------")

                # sources = [sources[idx].text for idx in range(len(sources))]
                # st.write(sources)

    else:
        clear_query_history()
