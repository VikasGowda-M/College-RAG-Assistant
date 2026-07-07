# ============================================
# IMPORT LIBRARIES
# ============================================

import os

from dotenv import load_dotenv

import chromadb

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai


# ============================================
# LOAD API KEY
# ============================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)


# ============================================
# LOAD EMBEDDING MODEL (LOCAL)
# ============================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================
# PDF FOLDER
# ============================================

folder_path = "data"


# ============================================
# CHUNK STORAGE
# ============================================

all_chunks = []


# ============================================
# CREATE CHUNK SPLITTER
# ============================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)


# ============================================
# READ EVERY PDF
# ============================================

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

        chunks = splitter.split_text(text)

        print(f"\n{file}")

        print(f"Total Chunks : {len(chunks)}")

        all_chunks.extend(chunks)


# ============================================
# CREATE EMBEDDINGS
# ============================================

embeddings = []

print("\nCreating Embeddings...\n")

for chunk in all_chunks:

    vector = embedding_model.encode(chunk).tolist()

    embeddings.append(vector)


# ============================================
# SUCCESS MESSAGE
# ============================================

print("=" * 50)

print("Embeddings Created Successfully")

print(f"Total Chunks : {len(all_chunks)}")

print(f"Total Embeddings : {len(embeddings)}")

print("=" * 50)

# ============================================
# CREATE CHROMADB CLIENT
# ============================================

client_db = chromadb.Client()

# ============================================
# CREATE COLLECTION
# ============================================

collection = client_db.create_collection(
    name="college_rag"
)

# ============================================
# STORE CHUNKS IN CHROMADB
# ============================================

for i in range(len(all_chunks)):

    collection.add(

        ids=[str(i)],

        documents=[all_chunks[i]],

        embeddings=[embeddings[i]]

    )

print("=" * 50)
print("ChromaDB Storage Completed")
print(f"Stored Records : {collection.count()}")
print("=" * 50)

# ============================================
# USER QUESTION
# ============================================

question = input("\nAsk your question : ")

# ============================================
# CONVERT QUESTION TO EMBEDDING
# ============================================

question_embedding = embedding_model.encode(question).tolist()

# ============================================
# SEARCH CHROMADB
# ============================================

results = collection.query(

    query_embeddings=[question_embedding],

    n_results=3

)

print("\n")
print("=" * 50)
print("Top 3 Retrieved Chunks")
print("=" * 50)

for i, doc in enumerate(results["documents"][0]):

    print(f"\nResult {i+1}\n")

    print(doc)

    print("-" * 80)

context = "\n\n".join(results["documents"][0])

prompt = f"""
You are a college assistant.

Answer ONLY using the information provided below.

If the answer is not present in the context,
say "I couldn't find the answer in the college documents."

Context:
{context}

Question:
{question}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(response.text)