from sentence_transformers import SentenceTransformer
import pandas as pd
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
    client = chromadb.PersistentClient(path="./LC-QuAD")


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


def context(retrieve :list):
  text= ""
  for items in range(len(retrieve)):
    text += f"{retrieve[items].get('english_text')} to SPARQL is {retrieve[items].get('sparql_query')} \n"
  return text

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./LC_QuAD2.0")
collection = client.get_collection(name="qa_collection")

def search_chroma(client, query_text, top_k=3):
  
    query_embedding = model.encode([query_text], normalize_embeddings=True).tolist()

    # Query collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Parse results
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        sparql_query = None
        if isinstance(meta, dict):
            sparql_query = (
                meta.get("sparql") or
                meta.get("sparql_wikidata") or
                meta.get("sparql_query") or
                meta.get("query") or
                meta.get("sparql_dbpedia18")
            )
        elif isinstance(meta, str):
            sparql_query = meta

        output.append({
            "english_text": doc,
            "sparql_query": sparql_query,
            "similarity_score": round(1 - dist, 4)  # cosine similarity
        })

    return output


if __name__ == "__main__":
    # Connect to existing database
    client = chromadb.PersistentClient(path="./LC-QuAD")

    
    # Example: search
    results = search_chroma(client, "Who is the president of France?", top_k=3)
    context_text = context(results)
    print(context_text)
    print(results[0])
