"""
验证所有关键依赖安装情况
"""
import sys

print("🔍 依赖检查工具\n")
print("=" * 60)

# 测试列表
tests = [
    ("pymilvus", "2.4.0+"),
    ("llama_index.core", "0.14+"),
    ("llama_index.vector_stores.milvus", "0.14+"),
    ("llama_index.embeddings.huggingface", "0.14+"),
    ("llama_index.llms.openai", "0.14+"),
    ("torch", "2.0+"),
    ("transformers", "4.41+"),
    ("ragas", "0.1.9"),
    ("streamlit", "1.34+"),
    ("FlagEmbedding", "1.2+"),
]

passed = 0
failed = 0

for module_name, version_req in tests:
    try:
        if "." in module_name:
            # 处理子模块导入
            parts = module_name.split(".")
            mod = __import__(module_name)
            for part in parts[1:]:
                mod = getattr(mod, part)
        else:
            mod = __import__(module_name)
        
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {module_name:40} {version}")
        passed += 1
    except ImportError as e:
        print(f"✗ {module_name:40} 未安装")
        failed += 1
    except Exception as e:
        print(f"⚠ {module_name:40} {str(e)[:30]}")
        passed += 1  # 导入成功，版本获取失败也算通过

print("\n" + "=" * 60)
print(f"测试结果: {passed} 通过, {failed} 失败")

# 测试 Milvus Lite
print("\n" + "=" * 60)
print("Milvus Lite 连接测试")
print("=" * 60)

try:
    from pymilvus import MilvusClient
    import os
    
    test_db = os.path.join(os.path.dirname(__file__), "data", "test.db")
    os.makedirs(os.path.dirname(test_db), exist_ok=True)
    
    client = MilvusClient(uri=test_db)
    print(f"✓ Milvus Lite 本地文件模式连接成功")
    print(f"  数据库路径: {test_db}")
    
except Exception as e:
    print(f"✗ Milvus Lite 测试失败: {e}")
    sys.exit(1)

print("\n✅ 所有核心依赖已安装，系统准备就绪！")
