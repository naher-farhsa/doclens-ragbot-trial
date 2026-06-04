from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_token_chunks(
    docs,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
):
    """
    Token-based chunking using RecursiveCharacterTextSplitter with tiktoken encoder.
    This is independent of hierarchical chunking - works directly on original documents.
    """
    print("=" * 60)
    print("[TOKEN CHUNKING] Starting token-based chunking")
    print(f"[TOKEN CHUNKING] Input: {len(docs)} document(s)")
    print(f"[TOKEN CHUNKING] Config: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"[TOKEN CHUNKING] Encoding: cl100k_base (tiktoken)")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("[TOKEN CHUNKING] Splitting documents into token-based chunks...")
    chunks = splitter.split_documents(docs)

    print(f"[TOKEN CHUNKING] Generated {len(chunks)} token chunks")

    for index, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_strategy": "token_recursive",
            "chunk_id": f"token_{index}",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })

    # Log sample chunks
    print(f"[TOKEN CHUNKING] Metadata added to all {len(chunks)} chunks")
    if chunks:
        print(f"[TOKEN CHUNKING] Sample chunk 0: {chunks[0].page_content[:100]}...")
        print(f"[TOKEN CHUNKING] Sample chunk 0 metadata: {chunks[0].metadata}")

    print(f"[TOKEN CHUNKING] Token-based chunking completed: {len(chunks)} chunks")
    print("=" * 60)

    return chunks
