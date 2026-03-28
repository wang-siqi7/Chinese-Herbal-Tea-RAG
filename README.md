# 中药茶饮消费者洞察 RAG 系统

基于 LangChain + ChromaDB 构建的 RAG 问答系统，用于分析中药茶饮问卷数据。

## 功能
- 上传问卷数据（TXT），自动向量化存储
- 自然语言问答：支持统计类问题（“有多少用户”）和条件筛选类问题（“从线下渠道了解的用户喜欢什么包装”）
- 路由逻辑：统计问题直接返回总数，检索问题走 RAG 链

## 技术栈
- Python 3.9+
- LangChain
- ChromaDB
- Streamlit
- 阿里云百炼（qwen3-max、text-embedding-v4）

## 数据
168 份真实问卷，4 个维度（了解渠道、包装偏好、改进方向、开放题建议）

## 运行
```bash
# 上传数据
streamlit run app_file_uploader.py

# 问答界面
streamlit run app_qa.py
