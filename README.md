# RAG-Based Document Q&A System

## What does this project do?

Ever wished you could just *ask* a document a question and get a straight answer? That's exactly what this project does.

Instead of reading through an entire PDF or policy document to find one specific detail, you simply feed the document into this system and ask your question in plain English. The system searches through the document, finds the most relevant sections, and uses Google's Gemini AI to give you a clear, accurate answer — grounded entirely in the document's content.

If the answer isn't in the document, it'll tell you that honestly instead of making something up.

---

## How it works (the RAG pipeline)

Here's the step-by-step flow of what happens behind the scenes:

```
Document → Text Extraction → Chunking → Embeddings → Vector Database → Similarity Search → Retrieved Context → LLM → Answer
```

1. **Load the document** — You provide a file (PDF, TXT, DOCX, or Markdown), and the system reads all the text from it.
2. **Split into chunks** — The full text gets broken down into smaller, overlapping pieces so the system can search through them efficiently.
3. **Create embeddings** — Each chunk is converted into a numerical representation (a vector) that captures its meaning. This is done locally using a HuggingFace model, so no API calls are needed for this step.
4. **Store in a vector database** — All the chunk embeddings are saved into ChromaDB, a local vector database that lives right in your project folder.
5. **Search for relevant context** — When you ask a question, your question is also turned into an embedding. The system then finds the chunks whose meaning is closest to your question.
6. **Generate the answer** — The most relevant chunks are sent to Google Gemini along with your question, and the model crafts a response using *only* the information from those chunks.

---

## Tech stack

| Component | What I used |
|---|---|
| Language | Python |
| Orchestration | LangChain |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) — runs locally, no API needed |
| Vector Database | ChromaDB (stored locally in `./chroma_db/`) |
| LLM | Google Gemini API (`gemini-3.5-flash`) |
| Document Loaders | PyPDF2, LangChain Community Loaders |

---

## How to set it up

### Prerequisites
- Python 3.8 or higher
- A Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/))

### Steps

1. **Clone this repo**
   ```bash
   git clone https://github.com/CloudHarshitha/RAG_ASSGINMENT_2.git
   cd RAG_ASSGINMENT_2
   ```

2. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Set up your API key**
   - Copy `.env.example` to `.env`
   - Open `.env` and paste your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_actual_key_here
     MODEL_NAME=gemini-3.5-flash
     ```

4. **Run it!**
   ```bash
   python main.py
   ```

That's it. The system will load the sample document, index it, and drop you into an interactive Q&A session.

---

## Sample questions you can try

A sample document (`sample_document.md`) is included — it's a fictional "Acme Corp Remote Work & Equipment Policy." Try asking:

- *"What is the work from home policy?"*
- *"Which VPN should employees connect when working from home?"*
- *"How much is the home office stipend?"*
- *"What equipment does the company provide?"*

---

## Understanding the RAG concepts

### What are embeddings?
Think of embeddings as a way to translate words into numbers that capture their *meaning*. The word "puppy" and "dog" would get similar numbers because they mean similar things, while "puppy" and "car" would be far apart. This is how the system understands which chunks of text are relevant to your question — it's not just matching keywords, it's matching meaning.

### Why do we need a vector database?
A normal database searches for exact keyword matches. But when you ask "What's the WFH policy?", you want it to also find text that says "remote work eligibility" — even though the words are completely different. A vector database like ChromaDB stores these meaning-based embeddings and can quickly find the most semantically similar chunks to your question.

### How does similarity search work?
When you type a question, it gets converted into an embedding (a list of numbers). The database then calculates the mathematical distance between your question's embedding and every stored chunk's embedding. The chunks with the smallest distance (i.e., most similar meaning) are returned as the relevant context.

### Why chunk size = 1000 and overlap = 200?
- **Chunk size of 1000 characters** gives each chunk enough surrounding context for the LLM to understand the information properly, without being so large that it dilutes the specific detail.
- **Overlap of 200 characters** ensures that if an important sentence falls right at the boundary between two chunks, it won't get cut in half — both chunks will contain it.

### How is RAG different from just asking an LLM directly?
When you ask ChatGPT or Gemini a question directly, it answers from its training data — which might be outdated, incomplete, or simply wrong for your specific use case. It has no idea what's in *your* company's policy document.

RAG fixes this by first *retrieving* the actual relevant text from your document, and then telling the LLM: "Here's the context — answer based on THIS, not your general knowledge." This keeps the answers factual and grounded in reality.

---

## Project structure

```
RAG_ASSGINMENT_2/
├── main.py              # CLI application entry point
├── rag_pipeline.py      # Core RAG logic (loading, chunking, embedding, retrieval, generation)
├── sample_document.md   # Sample document for testing
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .env                 # Your actual API keys (not committed to git)
├── .gitignore           # Keeps sensitive files out of the repo
└── chroma_db/           # Auto-generated vector database (not committed)
```
