"""
Windows环境基础功能测试
跳过Milvus Lite测试（仅Linux/macOS支持）
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_basic_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试基本包导入")
    print("=" * 60)
    
    tests = []
    
    # 测试核心包
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
        tests.append(("pandas", True))
    except ImportError as e:
        print(f"✗ pandas: {e}")
        tests.append(("pandas", False))
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
        tests.append(("numpy", True))
    except ImportError as e:
        print(f"✗ numpy: {e}")
        tests.append(("numpy", False))
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        tests.append(("torch", True))
    except ImportError as e:
        print(f"✗ torch: {e}")
        tests.append(("torch", False))
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
        tests.append(("transformers", True))
    except ImportError as e:
        print(f"✗ transformers: {e}")
        tests.append(("transformers", False))
    
    try:
        import streamlit
        print(f"✓ streamlit {streamlit.__version__}")
        tests.append(("streamlit", True))
    except ImportError as e:
        print(f"✗ streamlit: {e}")
        tests.append(("streamlit", False))
    
    try:
        from unstructured.partition.pdf import partition_pdf
        print(f"✓ unstructured")
        tests.append(("unstructured", True))
    except ImportError as e:
        print(f"✗ unstructured: {e}")
        tests.append(("unstructured", False))
    
    try:
        import ragas
        print(f"✓ ragas {ragas.__version__}")
        tests.append(("ragas", True))
    except ImportError as e:
        print(f"✗ ragas: {e}")
        tests.append(("ragas", False))
    
    return tests


def test_llama_index():
    """测试LlamaIndex组件"""
    print("\n" + "=" * 60)
    print("测试LlamaIndex组件")
    print("=" * 60)
    
    tests = []
    
    try:
        from llama_index.core import Settings, Document
        print(f"✓ llama_index.core")
        tests.append(("llama_index.core", True))
    except ImportError as e:
        print(f"✗ llama_index.core: {e}")
        tests.append(("llama_index.core", False))
        return tests
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"✓ llama_index.embeddings.huggingface")
        tests.append(("llama_index.embeddings", True))
    except ImportError as e:
        print(f"✗ llama_index.embeddings: {e}")
        tests.append(("llama_index.embeddings", False))
    
    try:
        from llama_index.llms.ollama import Ollama
        print(f"✓ llama_index.llms.ollama")
        tests.append(("llama_index.llms", True))
    except ImportError as e:
        print(f"✗ llama_index.llms: {e}")
        tests.append(("llama_index.llms", False))
    
    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        print(f"✓ llama_index.postprocessor")
        tests.append(("llama_index.postprocessor", True))
    except ImportError as e:
        print(f"✗ llama_index.postprocessor: {e}")
        tests.append(("llama_index.postprocessor", False))
    
    return tests


def test_project_imports():
    """测试项目模块导入"""
    print("\n" + "=" * 60)
    print("测试项目模块导入")
    print("=" * 60)
    
    tests = []
    
    try:
        from utils.data_preprocessing import parse_pdf, convert_to_documents
        print(f"✓ utils.data_preprocessing")
        tests.append(("utils.data_preprocessing", True))
    except ImportError as e:
        print(f"✗ utils.data_preprocessing: {e}")
        tests.append(("utils.data_preprocessing", False))
    
    try:
        from utils.system_prompt import EXPERT_Q_AND_A_SYSTEM
        print(f"✓ utils.system_prompt")
        tests.append(("utils.system_prompt", True))
    except ImportError as e:
        print(f"✗ utils.system_prompt: {e}")
        tests.append(("utils.system_prompt", False))
    
    try:
        from src.chatbot_web_demo.pages.qa_demo.hybrid_dev import ExampleEmbeddingFunction
        print(f"✓ src.chatbot_web_demo.pages.qa_demo.hybrid_dev")
        tests.append(("hybrid_dev", True))
    except ImportError as e:
        print(f"✗ hybrid_dev: {e}")
        tests.append(("hybrid_dev", False))
    
    return tests


def test_evaluation_module():
    """测试评估模块"""
    print("\n" + "=" * 60)
    print("测试评估模块")
    print("=" * 60)
    
    try:
        sys.path.insert(0, os.path.join(project_root, 'src'))
        import evaluation
        print(f"✓ evaluation模块导入成功")
        
        # 检查ExampleEmbeddingFunction类
        if hasattr(evaluation, 'ExampleEmbeddingFunction'):
            print(f"✓ ExampleEmbeddingFunction类定义正确")
            return [("evaluation", True)]
        else:
            print(f"✗ 缺少ExampleEmbeddingFunction类")
            return [("evaluation", False)]
    except Exception as e:
        print(f"✗ evaluation模块: {e}")
        return [("evaluation", False)]


def test_milvus_skip():
    """Milvus测试（Windows跳过）"""
    print("\n" + "=" * 60)
    print("Milvus Lite测试")
    print("=" * 60)
    print("⏭️  跳过 - Milvus Lite仅在Linux/macOS上支持")
    print("   在Linux服务器上将自动可用")
    return [("milvus_lite", "skipped")]


def main():
    print("\n🔍 Windows环境基础功能测试\n")
    print("注意: Milvus Lite测试将跳过（仅Linux支持）\n")
    
    all_results = []
    
    # 运行所有测试
    all_results.extend(test_basic_imports())
    all_results.extend(test_llama_index())
    all_results.extend(test_project_imports())
    all_results.extend(test_evaluation_module())
    milvus_result = test_milvus_skip()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result is True)
    failed = sum(1 for _, result in all_results if result is False)
    total = len(all_results)
    
    for name, result in all_results:
        if result is True:
            print(f"✓ {name}")
        elif result is False:
            print(f"✗ {name}")
    
    print(f"\nMilvus Lite: ⏭️  跳过（Linux专用）")
    
    print(f"\n通过: {passed}/{total}")
    print(f"失败: {failed}/{total}")
    
    if failed == 0:
        print("\n🎉 所有Windows可测试组件正常！")
        print("💡 在Linux服务器上部署后，Milvus Lite将自动可用。")
        return True
    else:
        print(f"\n⚠️  {failed}个组件测试失败，请检查安装。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
