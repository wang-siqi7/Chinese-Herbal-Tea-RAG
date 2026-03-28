# 在你的项目文件夹下新建一个 test_retriever.py 文件，内容如下：

from rag import RagService
from knowledge_base import KnowledgeBaseService

# 先确保向量库有数据
kb = KnowledgeBaseService()
count = kb.chroma._collection.count()
print(f"向量库文档总数：{count}")

# 创建 RagService 实例
rag = RagService()

# 获取检索器（从 __get_chain 里复制出来的创建逻辑）
retriever = rag.vector_service.vector_store.as_retriever(
    search_kwargs={"k": 100}
)

# 测试不同查询
queries = ["用户", "有多少个用户", "线下渠道", "包装偏好", "随便问问"]
for q in queries:
    docs = retriever.invoke(q)
    print(f"查询「{q}」返回 {len(docs)} 个文档")