from src.ingestion import run_ingestion_pipeline
from src.retrieval import run_retrieval_pipeline    
from src.generation import run_generation_pipeline
from langchain_core.messages import HumanMessage, AIMessage

file_path="./doc/IAI.pdf"

# ==========================================
# INGESTION: Independent chunking from original document
# Pipeline 1: Original Doc → Token Chunks (independent)
# Pipeline 2: Original Doc → Hierarchical Chunks (independent)
# ==========================================
run_ingestion_pipeline(file_path)

print("\n" + "=" * 60)
print("  DocLens RAG Chatbot is Ready!")
print("  Ask any questions about your document.")
print("  Type 'exit' or 'quit' to end the session.")
print("=" * 60 + "\n")

# Maintain separate independent chat histories
chat_history_token = []
chat_history_hier = []

while True:
    try:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        print(f"\n[SYSTEM] Processing query: '{query}'...")
        
        # ==========================================
        # RETRIEVAL: Independent retrieval from both chunk types
        # Step 1: Token Chunks Retrieval
        # Step 2: Hierarchical Chunks Retrieval
        # ==========================================
        retrieved_results = run_retrieval_pipeline(query)
        
        token_docs = retrieved_results.get("token_docs", [])
        hierarchical_docs = retrieved_results.get("hierarchical_docs", [])
        
        # ==========================================
        # ANSWER GENERATION: Independent from both chunk types
        # ==========================================
        print("\n" + "#" * 60)
        print("# GENERATING RESPONSES")
        print("#" * 60)
        
        # 1. Token Chunks Answer
        print("\n[TOKEN PIPELINE] Querying...")
        token_result = run_generation_pipeline(query, token_docs, chat_history_token)
        chat_history_token.append(HumanMessage(content=query))
        chat_history_token.append(AIMessage(content=token_result.answer))
        
        # 2. Hierarchical Chunks Answer
        print("\n[HIERARCHICAL PIPELINE] Querying...")
        hier_result = run_generation_pipeline(query, hierarchical_docs, chat_history_hier)
        chat_history_hier.append(HumanMessage(content=query))
        chat_history_hier.append(AIMessage(content=hier_result.answer))
        
        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        print("\n" + "=" * 80)
        print("                        RAG RESPONSE COMPARISON")
        print("=" * 80)
        
        print("\n=== [TOKEN CHUNKS PIPELINE RESPONSE] ===")
        print(f"Answer: {token_result.answer}")
        if token_result.sources:
            print("\nSources/Citations:")
            for idx, src in enumerate(token_result.sources):
                print(f"  [{idx+1}] {src.strip()}")
        
        print("\n=== [HIERARCHICAL CHUNKS PIPELINE RESPONSE] ===")
        print(f"Answer: {hier_result.answer}")
        if hier_result.sources:
            print("\nSources/Citations:")
            for idx, src in enumerate(hier_result.sources):
                print(f"  [{idx+1}] {src.strip()}")
        print("\n" + "=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred in chat session: {e}")