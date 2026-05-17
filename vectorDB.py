from sentence_transformers import SentenceTransformer
# ==============================
# Data Processing
# ==============================
import pandas as pd
import json

# ==============================
# Vector Database
# ==============================
import chromadb

def vector_db(path: str, question: str, sparql: str):
    # 1. Load dataset and drop missing rows
    if path.endswith(".json"):
        df = pd.read_json(path).dropna().reset_index(drop=True)
    elif path.endswith(".csv"):
        df = pd.read_csv(path).dropna().reset_index(drop=True)

    # 2. Load multilingual embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Connect to Chroma (persist to ./chroma_db)
    client = chromadb.PersistentClient(path="./LC_QuAD2.0")

    # 4. Create or get collection
    collection = client.get_or_create_collection(
        name="qa_collection",
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    # 5. Prepare inputs
    questions = ["query: " + q for q in df[question].astype(str).tolist()]
    ids = [str(i) for i in range(len(df))]
    documents = df[question].astype(str).tolist()
    metas = df[[sparql]].to_dict(orient="records")

    # 6. Generate embeddings
    embeddings = model.encode(
        questions,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    ).tolist()

    # 7. Insert in batches (<= 5461)
    batch_size = 5000
    for i in range(0, len(df), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metas[i:i + batch_size]
        )
        print(f"Inserted {min(i + batch_size, len(df))} / {len(df)}")

    print("Data stored in ChromaDB")
    print("Total records in collection:", collection.count())

    return collection   # return the collection

if __name__ == "__main__":
    pass
