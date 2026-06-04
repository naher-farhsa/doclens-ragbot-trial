from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4


def create_hierarchical_chunks(
    docs,
    parent_chunk_size: int = 1800,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 60,
):
    """
    Hierarchical two-level parent-child chunking.
    This is independent of token chunking - works directly on original documents.
    
    Level 1: Parent chunks (large context blocks)
    Level 2: Child chunks (smaller precise chunks derived from each parent)
    """
    print("=" * 60)
    print("[HIERARCHICAL CHUNKING] Starting hierarchical parent-child chunking")
    print(f"[HIERARCHICAL CHUNKING] Input: {len(docs)} document(s)")
    print(f"[HIERARCHICAL CHUNKING] Parent config: chunk_size={parent_chunk_size}, chunk_overlap={parent_chunk_overlap}")
    print(f"[HIERARCHICAL CHUNKING] Child config: chunk_size={child_chunk_size}, chunk_overlap={child_chunk_overlap}")
    print(f"[HIERARCHICAL CHUNKING] Encoding: cl100k_base (tiktoken)")
    print("=" * 60)

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )

    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    print("[HIERARCHICAL CHUNKING] Step 1: Creating parent chunks from original documents...")
    parent_chunks = parent_splitter.split_documents(docs)
    print(f"[HIERARCHICAL CHUNKING] Generated {len(parent_chunks)} parent chunks")

    final_parent_chunks = []
    final_child_chunks = []

    print("[HIERARCHICAL CHUNKING] Step 2: Creating child chunks from each parent...")
    for parent_index, parent_doc in enumerate(parent_chunks):
        parent_id = f"parent_{parent_index}_{uuid4().hex[:8]}"

        parent_doc.metadata.update({
            "chunk_strategy": "hierarchical_parent",
            "parent_id": parent_id,
            "parent_chunk_size": parent_chunk_size,
            "parent_chunk_overlap": parent_chunk_overlap,
        })

        final_parent_chunks.append(parent_doc)

        child_docs = child_splitter.split_documents([parent_doc])

        print(f"[HIERARCHICAL CHUNKING]   Parent {parent_index} (id={parent_id}) → {len(child_docs)} child chunks")

        for child_index, child_doc in enumerate(child_docs):
            child_doc.metadata.update({
                "chunk_strategy": "hierarchical_child",
                "parent_id": parent_id,
                "child_id": f"{parent_id}_child_{child_index}",
                "child_index": child_index,
                "child_chunk_size": child_chunk_size,
                "child_chunk_overlap": child_chunk_overlap,
            })

            final_child_chunks.append(child_doc)

    # Log summary
    print(f"[HIERARCHICAL CHUNKING] Total parent chunks: {len(final_parent_chunks)}")
    print(f"[HIERARCHICAL CHUNKING] Total child chunks: {len(final_child_chunks)}")

    if final_parent_chunks:
        print(f"[HIERARCHICAL CHUNKING] Sample parent 0: {final_parent_chunks[0].page_content[:100]}...")
        print(f"[HIERARCHICAL CHUNKING] Sample parent 0 metadata: {final_parent_chunks[0].metadata}")

    if final_child_chunks:
        print(f"[HIERARCHICAL CHUNKING] Sample child 0: {final_child_chunks[0].page_content[:100]}...")
        print(f"[HIERARCHICAL CHUNKING] Sample child 0 metadata: {final_child_chunks[0].metadata}")

    print(f"[HIERARCHICAL CHUNKING] Hierarchical chunking completed")
    print("=" * 60)

    return final_parent_chunks, final_child_chunks
