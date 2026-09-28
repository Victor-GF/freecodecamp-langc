from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings():
    text=  "What is machine learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

def similarity_search():
    docs = [
        # 1. Documento Técnico (A resposta CERTA, mas escrita em jargão de baixo nível)
        "The platform input layer captures WM_KEYDOWN and WM_KEYUP messages within the Win32 PeekMessage loop to update the system state.",
        
        # 2. Documento Distrator (Resposta ERRADA, mas usa linguagem coloquial genérica)
        "When a gamer wants to play, they must press a button on their mechanical keyboard.",
        
        # 3. Outro contexto (Nível mais alto)
        "In GDScript, you can easily connect a UI button signal to detect when it is pressed by the user.",
        
        # 4. Totalmente irrelevante
        "Machine Learning enables AI applications."
    ]

    query = "how do I make the engine recognize when I press a button on my keyboard?"
    
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity: ")
    for doc,score in ranked_docs:
        print(f"\t{score:.4f}: {doc}")

def embedding_caching():
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="exercise"
        )

        text = "What is Reinforcement Learning?"

        print("First call (API):")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"\tEmbedded {len(vectors1)} documents")

        print("\nSecond call (Cache):")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"\tEmbedded {len(vectors2)} documents")

        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")

if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()