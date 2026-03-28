
md5_path = "./md5.text"


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# spliter
chunk_size = 2000
chunk_overlap = 200
separators = ["\n\n", "【","\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 2000        # 文本分割的阈值

#
similarity_threshold = 30            # 检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
