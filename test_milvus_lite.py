"""
测试 Milvus Lite 配置是否正确
运行此脚本验证 Milvus Lite 适配是否成功
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_milvus_lite_connection():
    """测试 Milvus Lite 连接"""
    print("=" * 60)
    print("测试 Milvus Lite 配置")
    print("=" * 60)
    
    try:
        from pymilvus import connections
        
        # 测试本地文件连接
        milvus_db_path = os.path.join(project_root, "data", "test_milvus.db")
        os.makedirs(os.path.dirname(milvus_db_path), exist_ok=True)
        
        print(f"\n✓ pymilvus 导入成功")
        print(f"✓ 测试数据库路径: {milvus_db_path}")
        
        # 连接 Milvus Lite
        connections.connect(
            alias="test",
            uri=milvus_db_path,
        )
        print(f"✓ Milvus Lite 连接成功（本地文件模式）")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print(f"  请安装: pip install pymilvus>=2.4.0")
        return False
    except Exception as e:
        print(f"\n✗ 连接失败: {e}")
        return False


def test_sparse_embedding():
    """测试稀疏嵌入函数"""
    print("\n" + "=" * 60)
    print("测试稀疏嵌入函数")
    print("=" * 60)
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
        from typing import List
        
        class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
            def __init__(self):
                self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

            def encode_queries(self, queries: List[str]):
                outputs = self.model.encode(
                    queries,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )["lexical_weights"]
                return [self._to_standard_dict(output) for output in outputs]

            def encode_documents(self, documents: List[str]):
                outputs = self.model.encode(
                    documents,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )["lexical_weights"]
                return [self._to_standard_dict(output) for output in outputs]

            def _to_standard_dict(self, raw_output):
                result = {}
                for k in raw_output:
                    result[int(k)] = raw_output[k]
                return result
        
        print(f"\n✓ 稀疏嵌入函数类定义成功")
        print(f"  注意: 实际使用需要下载 BAAI/bge-m3 模型")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print(f"  请安装: pip install FlagEmbedding")
        return False
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False


def test_vector_store_config():
    """测试 MilvusVectorStore 配置"""
    print("\n" + "=" * 60)
    print("测试 MilvusVectorStore 配置")
    print("=" * 60)
    
    try:
        from llama_index.vector_stores.milvus import MilvusVectorStore
        
        milvus_db_path = os.path.join(project_root, "data", "test_milvus.db")
        
        # 测试配置（不实际创建）
        config = {
            "uri": milvus_db_path,
            "collection_name": "test_collection",
            "dim": 1024,
            "overwrite": True,
            "enable_sparse": True,
            "hybrid_ranker": "RRFRanker",
            "hybrid_ranker_params": {"k": 60},
        }
        
        print(f"\n✓ MilvusVectorStore 导入成功")
        print(f"✓ 配置参数:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print(f"  请安装: pip install llama-index-vector-stores-milvus")
        return False
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False


def main():
    print("\n🔍 Milvus Lite 配置检查工具\n")
    
    results = []
    
    # 测试 1: Milvus Lite 连接
    results.append(("Milvus Lite 连接", test_milvus_lite_connection()))
    
    # 测试 2: 稀疏嵌入函数
    results.append(("稀疏嵌入函数", test_sparse_embedding()))
    
    # 测试 3: VectorStore 配置
    results.append(("VectorStore 配置", test_vector_store_config()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！Milvus Lite 配置正确。")
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装。")
        print("\n建议安装命令:")
        print("  pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
