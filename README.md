# RAG-Based Document Q&A System

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) system that I built to answer questions from a given document without relying only on the model's general knowledge. The idea is simple — you give it a document, ask a question, and it finds the relevant parts of the document and uses those to generate a precise answer.

If the answer isn't actually in the document, the system will honestly say so instead of making something up (which is a common problem with regular LLMs).

The whole application runs from the command line — no web interface needed. Just run the script, and you can start asking questions right away.

---

## Architecture/Workflow

The application follows a standard RAG pipeline. Here's the flow:

```
Document → Text Extraction → Chunking → Embeddings → Vector Database → Similarity Search → Retrieved Context → LLM → Answer
```

Breaking it down step by step:

1. **Document Processing** — The system takes in a file (supports PDF, TXT, DOCX, and Markdown) and extracts all the text content from it using LangChain's document loaders.

2. **Text Chunking** — The extracted text is split into smaller overlapping pieces using a RecursiveCharacterTextSplitter. I chose a chunk size of 1000 characters with an overlap of 200 characters. More on why below.

3. **Embedding Generation** — Each chunk gets converted into a numerical vector (embedding) that captures its semantic meaning. I used HuggingFace's `all-MiniLM-L6-v2` model for this, which runs locally so there's no extra API cost.

4. **Vector Database Storage** — All the embeddings are stored in ChromaDB, a lightweight vector database that runs locally in the project folder under `./chroma_db/`.

5. **Similarity Search** — When you ask a question, your question also gets converted into an embedding. The system then searches ChromaDB to find the top 4 chunks that are most semantically similar to your question.

6. **LLM Answer Generation** — The retrieved chunks (context) along with your original question are sent to Google Gemini, which generates a final answer based strictly on the provided context.

---

## Technologies Used

- **Python** as the primary programming language
- **LangChain** for orchestrating the document loading, text splitting, and embeddings pipeline
- **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) for generating embeddings locally without needing an API
- **ChromaDB** as the local vector database for storing and searching embeddings
- **Google Gemini API** (`gemini-3.5-flash`) as the large language model for generating answers
- **PyPDF2** for PDF document parsing
- **python-dotenv** for managing API keys through environment variables

---

## Setup Instructions

**Step 1:** Make sure you have Python 3.8 or higher installed on your machine.

**Step 2:** Clone this repository and navigate into it:
```bash
git clone https://github.com/CloudHarshitha/RAG_ASSGINMENT_2.git
cd RAG_ASSGINMENT_2
```

**Step 3:** Install the required Python packages:
```bash
python -m pip install -r requirements.txt
```

**Step 4:** Get a free API key from [Google AI Studio](https://aistudio.google.com/).

**Step 5:** Create a `.env` file in the project root (you can copy from the example):
```
GOOGLE_API_KEY=paste_your_actual_key_here
MODEL_NAME=gemini-3.5-flash
```

## How to run the application

**Step 6:** Run the application:
```bash
python main.py
```

The system will load the sample document, process it, and then you can start asking questions interactively.

---

## Sample document used for testing

I've included a sample document called `sample_document.md` in the repository. It contains a fictional "Acme Corp — Remote Work and Equipment Policy" that covers topics like remote work eligibility, home office stipends, equipment provided, VPN requirements, and communication expectations.

Some questions you can try:
- "What is the company's work from home policy?"
- "Which VPN should employees use when working from home?"
- "How much is the home office stipend?"
- "What equipment does the company provide to employees?"

If you ask something that's not in the document (like "What is the company's stock price?"), the system will clearly say that the information is not available in the document.

---

## Explanation of Key RAG Concepts

### 1. What are embeddings?

* Think of embeddings as translating human words into computer math. 
* We turn a sentence into a list of numbers (a vector) that captures its actual *meaning*.
* **Example:** If one sentence says "stipend of $500" and another says "office equipment allowance", the computer knows they mean the same thing because their number lists (vectors) are very similar!

### 2. Why do we need a vector database?

* Normal databases are dumb — they only find exact word matches.
* A vector database (like ChromaDB) stores our meaning-based number lists.
* **Example:** If you ask "Do I get money for my desk?", a normal database won't find the answer because the document says "stipend of $500 for home office setup". The vector database understands the meaning and finds it perfectly!

### 3. How does similarity search work?

* When you ask a question, we turn it into a number list too.
* We then ask the database: "Which document chunks have numbers mathematically closest to my question's numbers?"
* **Example:** Your question about "company VPN" is mathematically compared to the entire document. The database calculates the distance and pulls out the specific section about the "Acme Corp VPN" and "Security Protocols" because they are the closest match.

### 4. Why did you choose your chunk size and overlap?

* **Chunk Size (1000):** Not too big, not too small. It's just enough text for the LLM to get full context (like the entire "Home Office Equipment Setup" section) without being overwhelmed by unrelated stuff.
* **Overlap (200):** Think of this as a safety net. If a sentence about the "$50 per month internet bill" falls right on the edge of a chunk, the overlap ensures the whole sentence makes it into the next chunk so nothing gets chopped in half.

### 5. How is RAG different from simply asking an LLM?

* **Regular LLMs** just guess answers from memory. If you ask ChatGPT about "Acme Corp's VPN", it will hallucinate because it hasn't seen our private document.
* **RAG** gives the LLM an open-book test. We search the document *first*, hand the LLM the right paragraph, and say "Read this and answer the question."

**Normal LLM:**
`Question → LLM → Answer`

**RAG:**
```text
Question
   ↓
Search Document
   ↓
Retrieve Relevant Context
   ↓
LLM
   ↓
Answer
```

This keeps the answers 100% grounded in actual facts from our document!

---

## Project Structure

```
RAG_ASSGINMENT_2/
├── main.py              — Entry point, runs the interactive CLI
├── rag_pipeline.py      — Core RAG logic (loading, chunking, embedding, retrieval, generation)
├── sample_document.md   — Sample policy document for testing
├── requirements.txt     — Python dependencies
├── .env.example         — Template showing required environment variables
├── .env                 — Your actual API keys (gitignored, not pushed to GitHub)
├── .gitignore           — Keeps sensitive files out of version control
└── chroma_db/           — Auto-created vector database folder (gitignored)
```
