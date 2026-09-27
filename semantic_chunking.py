from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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

PARAGRAPH = '\n\n'
LINE = '\n'
PHRASE = '. '
WORD = ' '
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[PARAGRAPH, LINE, PHRASE, WORD]
)

recursive_chunks = recursive_splitter.split_text(document)

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(document)


recursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embeddings,
    collection_name="recursive_chunks",
)

semantic_vectorstore= Chroma.from_texts(
    semantic_chunks,
    embeddings,
    collection_name="semantic_chunks"
)

test_queries = [
    'How do I authenticate with 0Auth2?',
    'What happens when I hit the rate limit?',
    'How are webhooks secured?',
    'What format are errors returned in?'
]

def test_retrieval(query, vectorstore, name):
    results = vectorstore.similarity_search(query, k=1)
    print(f"\n{name} - Query: \"{query}\"")
    print(f"Retrieved: {results[0].page_content[:150]}...")
    return results[0].page_content

print(f"\n{'='*60}")
print(" RETRIEVAL TESTS")
print(f"{'='*60}")

for query in test_queries:
    print('=' * 60)
    recursive_result = test_retrieval(query, recursive_vectorstore, "RECURSIVE")
    semantic_result = test_retrieval(query, semantic_vectorstore, "SEMANTIC")

