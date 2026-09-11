import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from google import genai

class RAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # 3. Embeddings: Using a small, fast local embedding model from HuggingFace
        # This way the user doesn't need an API key for embeddings, and it's free/fast.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.client = None
        
        # Try to load existing vectorstore
        if os.path.exists(persist_directory):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception:
                pass
                
    def load_document(self, file_path: str) -> List[Document]:
        """
        1. Document Processing: Extract text from PDF, TXT, DOCX, or MD files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.md':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        return loader.load()

    def process_and_index_document(self, file_path: str) -> str:
        """
        Processes a file, extracts text, chunks it, and indexes into Vector DB.
        """
        # Step 1: Load Document
        documents = self.load_document(file_path)
        
        # Step 2: Text Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Step 4: Vector Database (Index embeddings into ChromaDB)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        return f"Successfully processed {len(chunks)} chunks from {os.path.basename(file_path)}."

    def setup_qa_chain(self):
        """
        Initializes the Google GenAI client.
        """
        if not self.vectorstore:
            raise ValueError("Vector database is not initialized. Please upload a document first.")
            
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing.")

        # Initialize the new Google GenAI client (uses v1 API, not v1beta)
        self.client = genai.Client(api_key=api_key)

    def answer_question(self, query: str) -> Tuple[str, List[Document]]:
        """
        Takes a user query, performs vector search, and generates an answer using LLM.
        """
        if self.client is None:
            self.setup_qa_chain()
            
        # 5. Similarity Search & Retrieval
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        source_docs = retriever.invoke(query)
        
        # 6. Generate the Final Answer
        context = "\n\n".join(doc.page_content for doc in source_docs)
        
        prompt = f"""Use the following pieces of retrieved context to answer the question. 
If the answer cannot be found in the context, clearly state that the information is not available in the document.
Do not try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""
        
        model_name = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        return response.text, source_docs
