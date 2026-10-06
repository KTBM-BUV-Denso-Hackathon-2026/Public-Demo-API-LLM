from database.config import get_db
from embedder.engine import get_model
import psycopg2

def save_to_raw_data(columns, values):
    """
    Save data to the rag_rawdata table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create a string of placeholders for the values
            placeholders = ', '.join(['%s'] * len(values))
            # Create a string of column names
            columns_str = ', '.join(columns)
            # Create the SQL query
            query = f"INSERT INTO rag_rawdata ({columns_str}) VALUES ({placeholders})"
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()

def save_to_rag_data(columns, values):
    """
    Save data to the rag_ragdata table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """# Assuming the first value is the content to be embedded
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create a string of placeholders for the values
            placeholders = ', '.join(['%s'] * len(values))
            # Create a string of column names
            columns_str = ', '.join(columns)
            # Create the SQL query
            query = f"INSERT INTO rag_ragdata ({columns_str}) VALUES ({placeholders})"
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()

def save_to_embedding_data(columns, values):
    """
    Save data to the rag_embedding table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create a string of placeholders for the values
            placeholders = ', '.join(['%s'] * len(values))
            # Create a string of column names
            columns_str = ', '.join(columns)
            # Create the SQL query
            query = f"INSERT INTO rag_chunkdata (id,chunk_content,chunk_index,metadata,vector,fts) VALUES (%s, %s, %s, %s, %s, %s)"
            
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()

def batch_search_rag_data(latest_queries, top_k=5):
    """
    Search for relevant data in the rag_ragdata table based on the latest query.

    Args:
        latest_queries (str): The latest list of queries to search for.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of dictionaries containing the search results.
    """
    embedder_model = get_model()  # Load the embedding model
    embeddings = embedder_model.encode(latest_queries)  # Generate embeddings for the latest queries
    with get_db() as conn:
        with conn.cursor() as cursor:
            #embs = em
            # Create the SQL query to search for relevant data
            query = f"""
                SELECT * FROM rag_ragdata
                WHERE content ILIKE %s
                ORDER BY ts_rank_cd(to_tsvector(content), plainto_tsquery(%s)) DESC
                LIMIT %s
            """
            retrieved_contexts = []
            for emb in embeddings:
                # Tìm top 2 đoạn văn bản liên quan nhất dựa trên Cosine Distance (<->)
                conn.execute(
                    query,
                    (emb.tolist(), top_k)
                )
                docs = conn.fetchall()
                context_str = "\n".join([f"- {d[0]}" for d in docs])
                retrieved_contexts.append(context_str)
            return retrieved_contexts

def search_rag_data(latest_query, top_k=5):
    """
    Search for relevant data in the rag_ragdata table based on the latest query.

    Args:
        latest_query (str): The single latest query to search for.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of dictionaries containing the search results.
    """
    embedder_model = get_model()  # Load the embedding model
    embedding = embedder_model.encode([latest_query])[0]  # Generate embedding for the latest query
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create the SQL query to search for relevant data
            query = f"""
                SELECT * FROM rag_ragdata
                WHERE content ILIKE %s
                ORDER BY ts_rank_cd(to_tsvector(content), plainto_tsquery(%s)) DESC
                LIMIT %s
            """
            # Tìm top 2 đoạn văn bản liên quan nhất dựa trên Cosine Distance (<->)
            cursor.execute(
                query,
                (embedding.tolist(), top_k)
            )
            docs = cursor.fetchall()
            context_str = "\n".join([f"- {d[0]}" for d in docs])
            return context_str