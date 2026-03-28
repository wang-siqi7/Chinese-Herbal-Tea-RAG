from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


class RagService(object):
    def __init__(self):
        from knowledge_base import KnowledgeBaseService
        self.kb_service = KnowledgeBaseService()
        self.total_users = self.kb_service.chroma._collection.count()

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个中药茶饮消费者洞察助手。以下是从问卷数据中提取的用户信息：{context}。如果用户数据中缺少某个维度的信息，请说明'该维度数据不够'，不要强行编造"),
                ("system", "对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user", "用户问题：{input}\n请基于以上用户数据，简洁地回答用户的问题，如果有多个用户，可以总结趋势、列出比例")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        # 判断是否是统计类问题
        def is_statistical_query(input_dict):
            query = input_dict.get("input", "").lower()
            keywords = ["多少", "几个", "总数", "总共", "有多少", "一共", "统计"]
            return any(kw in query for kw in keywords)

        # 直接从 self.kb_service 获取用户数量
        def answer_statistical_query(input_dict):
            return f"当前共有 {self.total_users} 个用户参与本次中药茶饮调研。"

        # 检索器：使用配置中的 retrieval_k
        retriever = self.vector_service.vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_k}
        )

        # 原有的 format 函数
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        # 原有的 RAG 链
        rag_chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        # 路由：根据问题类型选择
        def route(input_dict):
            if is_statistical_query(input_dict):
                return answer_statistical_query(input_dict)
            else:
                return rag_chain.invoke(input_dict)

        # 包装成 Runnable
        main_chain = RunnableLambda(route)

        # 加上历史记录
        conversation_chain = RunnableWithMessageHistory(
            main_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain