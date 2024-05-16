from tqdm import tqdm

import psycopg2
from sqlalchemy import make_url

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

if __name__ == "__main__":
    db_name = "vector_db"
    host = "localhost"
    password = "12345678"
    port = "5432"
    user = "postgres"

    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    print("Connect to DB Sucessful")
    conn.autocommit = True

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="test",
        embed_dim=768,  # openai embedding dimension
    )

    embed_model = HuggingFaceEmbedding(model_name="bkai-foundation-models/vietnamese-bi-encoder")

    query_str = "Cần chạy xe tới khi đi đăng ký xe máy hay không?"
    
    query_embedding = embed_model.get_query_embedding(query_str)

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )

    query_result = vector_store.query(vector_store_query)
    print(query_result.nodes[0].get_content())