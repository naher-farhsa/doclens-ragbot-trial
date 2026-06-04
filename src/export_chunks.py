import json
import os


def export_chunks_to_jsonl(chunks, output_path: str):
    """
    Export chunks to JSONL file for later retrieval/testing.
    Each line is a JSON object with page_content and metadata.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[EXPORT] Exporting {len(chunks)} chunks to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[EXPORT] Successfully exported {len(chunks)} chunks to {output_path}")
