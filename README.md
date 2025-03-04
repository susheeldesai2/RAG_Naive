Retrieval-Augmented Generation (RAG) System

Overview

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline that enhances LLM responses by retrieving relevant information from external documents (PDFs). The system extracts text from PDFs, chunks the text, embeds it into a vector database (Pinecone), and performs similarity searches to improve LLM-generated answers.

Features

✅ Extracts text from PDF files using PyPDF2✅ Splits text into manageable chunks using langchain_text_splitters✅ Embeds text chunks using SentenceTransformer✅ Stores embeddings in Pinecone vector database✅ Accepts user queries and retrieves relevant document chunks✅ Uses LangChain and ChatGroq to generate contextual responses

Installation

To get started, clone this repository and install the required dependencies:

# Clone the repository
git clone <your-repo-link>
cd <your-repo-folder>

# Install dependencies
pip install -r requirements.txt

Dependencies

Ensure the following dependencies are installed:

python-dotenv
PyPDF2
langchain
pinecone
langchain-groq
sentence-transformers

How It Works

1️⃣ Extract Text from PDF

The system reads the text from a PDF file using PyPDF2.

pdf_reader = PdfReader("incorrect_facts.pdf")
text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])

2️⃣ Chunk the Extracted Text

Using CharacterTextSplitter, the extracted text is divided into smaller, meaningful chunks for retrieval efficiency.

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50, length_function=len)
chunks = text_splitter.split_text(text)

3️⃣ Store Chunks in a Vector Database

Each chunk is embedded using SentenceTransformer, and the embeddings are stored in Pinecone.

embedding_model = SentenceTransformer("BAAI/bge-small-en")
chunk_embedding = embedding_model.encode(chunk, normalize_embeddings=True)
index.upsert([(str(i+1), chunk_embedding.tolist(), {"chunk": chunk})])

4️⃣ User Queries the System

A user asks a question, which is embedded and compared against the stored embeddings.

query = "How do Birds Migrate"
question_embedding = embedding_model.encode(query, normalize_embeddings=True)
result = index.query(vector=question_embedding.tolist(), top_k=3, include_metadata=True)

5️⃣ Retrieve Relevant Chunks and Generate Answer

The system retrieves the most relevant chunks and combines them with the user query before passing them to an LLM for final answer generation.

augmented_text = "\n\n".join([match.metadata["chunk"] for match in result.matches])
prompt = PromptTemplate(input_variables=["context", "question"], template="""
You are a helpful assistant. Use the context provided to answer the question accurately.
Only use this context, not your knowledge.

Context:{context}
Question:{question}
""")
chain = prompt | llm
response = chain.invoke({"context": augmented_text, "question": query})
print(response.content)

Future Enhancements 🚀

Optimizing with RAGAS to improve retrieval and filtering

Adding more sophisticated ranking and retrieval mechanisms

Expanding to support additional document types (e.g., Word, CSV)
