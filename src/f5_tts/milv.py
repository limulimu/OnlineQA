import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Filter
from sentence_transformers import SentenceTransformer
import pandas as pd

# 初始化 Qdrant 客户端
client = QdrantClient(path="vectors")
# client = QdrantClient(url="http://localhost:6333")
# 创建 Qdrant 集合
collection_name = "qa_collection"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")
)

# 加载中文嵌入模型
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# # 示例问题和答案
# # questions = ["你好吗？", "天气怎么样？", "你的名字是什么？"]
# # answers = ["我很好，谢谢！", "今天天气很好。", "我的名字是小明。"]
# QA = pd.read_excel("all.xlsx")
# questions = QA['Q'].to_list()
# answers = QA['A'].to_list()
# firm_ids = QA['firm_id'].to_list()  # Assuming 'firm_id' column exists
# product_ids = QA['product_id'].to_list()  # Assuming 'product_id' column exists
# q_ids = QA['ID'].to_list()
# # 将问题向量化
# question_embeddings = model.encode(questions)

# # 插入向量到 Qdrant
# points = [
#     PointStruct(id=i, vector=question_embeddings[i], payload={"question": questions[i], "answer": answers[i],"q_id":q_ids[i],"firm_id":firm_ids[i],"product_id":product_ids[i]})
#     for i in range(len(questions))
# ]
# client.upsert(collection_name=collection_name, points=points)

# # 查询示例
# query_question = "如果坏了怎么退货"
# query_embedding = model.encode([query_question])[0]

# # 搜索最相似的问题
# search_results = client.search(
#     collection_name=collection_name,
#     query_vector=query_embedding,
#     limit=1,
#     with_payload=True,
# )

# # 输出最相似的问题及其答案
# for result in search_results:
#     print(f"最相似的问题: {result.payload['question']}, 答案: {result.payload['answer']},距离:{result.score}")
