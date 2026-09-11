import os
import sys
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Fix encoding for Windows terminals that don't support UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("=" * 60)
    print("   RAG-Based Document Q&A System (CLI)")
    print("=" * 60)
    
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n[ERROR] GOOGLE_API_KEY is missing.")
        print("Please open the '.env' file in this folder and paste your Google Gemini API key next to 'GOOGLE_API_KEY='.")
        print("Example: GOOGLE_API_KEY=AIzaSyAXXXX...")
        return

    model_name = os.environ.get("MODEL_NAME", "gemini-3.5-flash")
    print(f"\n   Model:  {model_name}")
    print(f"   API Key: {'*' * 10}...{os.environ['GOOGLE_API_KEY'][-4:]}")

    print("\n[Step 1] Initializing RAG Pipeline (Loading Embeddings Model)...")
    pipeline = RAGPipeline()
    
    doc_path = "sample_document.md"
    if not os.path.exists(doc_path):
        print(f"\n[ERROR] Could not find {doc_path} in the current directory.")
        return
        
    print(f"\n[Step 2] Ingesting Document: {doc_path}")
    print("   -> Extracting text...")
    print("   -> Chunking text...")
    print("   -> Generating Embeddings and Storing in ChromaDB...")
    
    try:
        result_msg = pipeline.process_and_index_document(doc_path)
        print(f"   [SUCCESS] {result_msg}")
    except Exception as e:
        print(f"   [ERROR] Failed to process document: {e}")
        return
        
    print("\n[Step 3] Setting up QA Chain (Connecting to LLM)...")
    pipeline.setup_qa_chain()
    print("   [SUCCESS] Pipeline is ready!")
    
    print("\n" + "=" * 60)
    print("   Ask questions about the document!")
    print("   (Type 'quit' or 'exit' to stop)")
    print("=" * 60)
    
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        if not query.strip():
            continue
            
        print("\n   Searching Vector Database for Context...")
        print("   Generating Answer with LLM...\n")
        try:
            answer, sources = pipeline.answer_question(query)
            
            print("=" * 60)
            print("   GENERATED ANSWER")
            print("=" * 60)
            print(f"\n{answer.strip()}\n")
            
            print("=" * 60)
            print("   RETRIEVED CONTEXT (Source chunks from Vector DB)")
            print("=" * 60)
            for i, doc in enumerate(sources):
                print(f"\n   [Chunk {i+1}]")
                print(f"   {doc.page_content.strip()}")
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"\n[ERROR] Error generating answer: {e}")

if __name__ == "__main__":
    main()
