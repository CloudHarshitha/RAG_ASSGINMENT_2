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

### What are Embeddings?

Embeddings are basically a way of converting text into numbers so that a computer can understand meaning. When we turn a sentence into an embedding, we get a list of numbers (a vector) that represents what that sentence is about. Sentences with similar meanings end up with similar vectors. For example, "remote work policy" and "work from home guidelines" would have very close embeddings even though the actual words are different.

I used the `all-MiniLM-L6-v2` model from HuggingFace for generating these embeddings. It runs entirely on your local machine, so there's no API cost involved and it's quite fast.

### Why is a Vector Database Needed?

A regular database can only search for exact keyword matches. But when someone asks "Can I work from home?", we want the system to also find text that talks about "remote work eligibility" — even though the words are completely different. A vector database like ChromaDB stores the embeddings and lets us search by meaning rather than by exact words. This is what makes the whole retrieval step work.

### How Similarity Search Works

When you type a question, the system converts your question into an embedding vector. Then ChromaDB calculates the mathematical distance (using cosine similarity) between your question vector and all the stored chunk vectors. The chunks with the smallest distance — meaning the closest semantic match — are returned as the most relevant context. In this project, I retrieve the top 4 most relevant chunks.

### Chunk Size and Overlap — Why 1000 and 200?

I chose a chunk size of 1000 characters and an overlap of 200 characters. A chunk size of 1000 is large enough to preserve meaningful context within each chunk, but small enough that the retrieval stays focused and doesn't pull in too much irrelevant information. The 200-character overlap is there to make sure that if an important sentence happens to fall right at the boundary between two chunks, it doesn't get cut in half — both neighboring chunks will contain it.

### How RAG is Different from Just Asking an LLM

When you ask a regular LLM (like ChatGPT or Gemini) a question directly, it answers purely from whatever it learned during training. It has no idea what's in your specific document, and it might confidently give you wrong information (this is called hallucination).

RAG solves this by adding a retrieval step before generation. Instead of letting the LLM answer from memory, we first search our document for the most relevant information, and then we pass that information to the LLM with clear instructions: "Answer this question using ONLY the context I've given you." This keeps the answers grounded in actual facts from the document.

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
