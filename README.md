# RAG-Based Document Q&A System

## 1. Project Overview
This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, TXT, MD) and ask natural language questions about their content. Instead of relying solely on an LLM's pre-trained knowledge, the system searches the uploaded document for relevant information and uses it to construct a highly accurate and context-specific answer. 

If the answer isn't in the document, the system will explicitly state that the information is not available, preventing hallucinations.

## 2. Architecture / Workflow
The application follows a standard RAG pipeline:

1. **Ingest (Document Processing):** The system loads the specified document (PDF, TXT, DOCX, MD) using LangChain loaders (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`).
2. **Chunk (Text Chunking):** The extracted text is split into smaller, manageable chunks using a `RecursiveCharacterTextSplitter`.
3. **Embed (Embeddings):** Each text chunk is converted into a high-dimensional vector representation (embedding) using a local HuggingFace embedding model (`all-MiniLM-L6-v2`).
4. **Index (Vector Database):** The embeddings and their corresponding text chunks are stored locally in **ChromaDB**.
5. **Retrieve (Similarity Search):** When a user asks a question, the query is embedded using the same model. A vector similarity search is performed against ChromaDB to find the top K most relevant text chunks.
6. **Generate (LLM):** The retrieved context chunks and the user's original question are passed to **Google Gemini (gemini-1.5-pro)** via a strict prompt template to generate the final answer.

## 3. Technologies Used
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (`sentence-transformers`, `all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **LLM:** Google Gemini API (`gemini-1.5-pro`)
* **Document Loaders:** `PyPDF2`, `docx2txt`, LangChain Community Loaders

## 4. Setup Instructions
1. Ensure you have **Python 3.8+** installed.
2. Clone this repository or open the project folder.
3. Open a terminal and create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
6. Rename `.env.example` to `.env` and paste your API key inside:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 5. How to run the application
Run the following command in your terminal from the project root:
```bash
python main.py
```
This will start the command-line interface where the document is ingested and you can chat with it.

## 6. Sample Document Used for Testing
A sample document named `sample_document.md` is included in the repository. It contains a fictional "Acme Corp - Remote Work and Equipment Policy". 
You can upload this file via the sidebar to test questions like:
* *"What is the work from home policy?"*
* *"How much is the home office stipend?"*

## 7. RAG Concepts Explanation

### What are embeddings?
Embeddings are numerical representations (vectors) of text. They capture the semantic meaning and context of words, sentences, or paragraphs in a high-dimensional mathematical space. This allows computers to understand relationships between text concepts (e.g., understanding that "dog" and "puppy" are closer in meaning than "dog" and "car").

### Why is a vector database required?
Standard relational databases search for exact keyword matches. A vector database is designed specifically to store and query high-dimensional embeddings efficiently. It allows us to perform "similarity searches," finding text chunks that are *semantically related* to a user's question, even if they don't share the exact same keywords.

### How similarity search works
When a user asks a question, the query is converted into an embedding (a vector). The vector database then calculates the mathematical distance (e.g., Cosine Similarity or Euclidean distance) between the query vector and all the chunk vectors stored in the database. The chunks with vectors that are closest to the query vector are returned as the most "similar" or relevant context.

### What chunk size and overlap were selected?
* **Chunk Size:** 1000 characters
* **Chunk Overlap:** 200 characters

**Explanation:** A chunk size of 1000 provides enough context for the LLM to understand the surrounding information without exceeding token limits or diluting the specific meaning. The 200-character overlap ensures that sentences or concepts aren't abruptly cut in half across two different chunks, preserving the continuity of the information.

### How RAG differs from simply asking an LLM a question
When you simply ask an LLM a question, it generates an answer based purely on the static data it was trained on months or years ago. It might hallucinate (make things up) or lack knowledge of private/recent documents. 
**RAG (Retrieval-Augmented Generation)** intercepts the process by first *retrieving* relevant factual information from your specific documents, and then explicitly instructing the LLM to use *only* that retrieved context to formulate its answer. This grounds the LLM in truth and allows it to answer questions about proprietary or unseen data.
