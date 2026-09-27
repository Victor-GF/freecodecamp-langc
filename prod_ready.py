from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need 0Auth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def smart_chunker(
    text: str, 
    use_semantic: bool = True,
    fallback_chunk_size: int = 500
) -> list[str]:
    """Production chunking with semantic as primary, recursive as fallback"""
    
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            chunks = chunker.split_text(text)

            max_chunk_size = 2000
            # fallback to recursive for oversized chunks
            if any(len(c) > max_chunk_size for c in chunks):
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}, using fallback")
            return _recursive_fallback(text, fallback_chunk_size)

    return _recursive_fallback(text, fallback_chunk_size)

def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )
    return splitter.split_text(text)

chunks = smart_chunker(document, use_semantic=True)
print(f"Created {len(chunks)} semantic chunks")

test_queries = [
    'How do I authenticate with 0Auth2?',
    'What happens when I hit the rate limit?',
    'How are webhooks secured?',
    'What format are errors returned in?'
]

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="smart_chunking"
)

print(f"\n{'='*60}")
print(" TESTANDO BUSCA (SMART CHUNKER)")
print(f"{'='*60}")

for query in test_queries:
    # k=1 retorna apenas o melhor resultado
    results = vectorstore.similarity_search(query, k=1)
        
    print(f"Query: \"{query}\"")
    # Imprime os primeiros 150 caracteres para visualização rápida
    print(f"Retrieved: {results[0].page_content[:150]}...")
    print("-" * 60)