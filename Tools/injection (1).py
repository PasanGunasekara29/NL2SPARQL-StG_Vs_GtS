from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")



def insert_batch(df, text_col, id_col, batch_size=100):
    """
    Insert data into ChromaDB in batches with metadata.
    - text_col: column to embed
    - id_col: unique identifier column
    """
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]

        ids = batch[id_col].astype(str).tolist()
        texts = batch[text_col].tolist()

        # Generate embeddings from text
        embeddings = embedder.encode(texts).tolist()

        # Metadata = all other columns (including original text if needed)
        metadata = batch.drop(columns=[id_col]).to_dict(orient="records")

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=texts  # keep raw text for direct retrieval
        )
    print(f"✅ Inserted {len(df)} records in batches of {batch_size}")

def query_collection(query_text, collection, top_k=5, where=None):
    """
    Query by text similarity with metadata return
    """
    query_embedding = embedder.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where
    )
    return results

def context_pattern(response):
  text=""
  for items in range(len(response["metadatas"][0])):
    question = response["metadatas"][0][items].get("corrected_question")
    patten_question = response["metadatas"][0][items].get("intermediary_question")
    text += f"{question} to relevant pattern is {patten_question}\n"
  return text




if __name__ =="__main__":
  # Init Chroma
  chroma_client = chromadb.PersistentClient(path="./LC-QuAD")
  collection = chroma_client.create_collection(name="qa_collection")
