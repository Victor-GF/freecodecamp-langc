from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")

def connect_to_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=POSTGRES_DB_URL,
        use_jsonb=True
    )

    return vectorstore

def verify_connection(vectorstore):
    from langchain_core.documents import Document

    test_doc = Document(
        page_content="This is a test document to verify Supabase blablabla",
        metadata={"test": True}
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Added test document: {ids [0]}")

        # Search for it
        results = vectorstore.similarity_search("test document")
        if results:
            print(f"✅ Search works: {results[0].page_content}")

        # Clean up
        # vectorstore.delete(ids)
        # print("✅ Cleanup complete")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    print(f"\n\tConnecting to Supabase...")
    print(f"\tHost: {POSTGRES_DB_URL.split('@')[1].split(':')[0]}")

    vectorstore = connect_to_db()
    verify_connection(vectorstore)

    

if __name__ == "__main__":
    main()