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
metadata_list = []

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

        for page_number, page in enumerate(reader.pages):

            page_text = page.extract_text()

            if page_text:

                chunks = splitter.split_text(page_text)

                for chunk in chunks:

                    all_chunks.append(chunk)

                    metadata_list.append({

                        "source": file,

                        "page": page_number + 1

                    })

        # print(f"{file} Loaded")


# ============================================
# CREATE EMBEDDINGS
# ============================================

embeddings = []

# print("\nCreating Embeddings...\n")

for chunk in all_chunks:

    vector = embedding_model.encode(chunk).tolist()

    embeddings.append(vector)


# ============================================
# SUCCESS MESSAGE
# ============================================

# print("=" * 50)

# print("Embeddings Created Successfully")

# print(f"Total Chunks : {len(all_chunks)}")

# print(f"Total Embeddings : {len(embeddings)}")

# print("=" * 50)

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

        embeddings=[embeddings[i]],

        metadatas=[metadata_list[i]]

    )


        

    

# print("=" * 50)
# print("ChromaDB Storage Completed")
# print(f"Stored Records : {collection.count()}")
# print("=" * 50)

# ============================================
# USER QUESTION
# ============================================

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":

        print("Goodbye 👋")

        break

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(

        query_embeddings=[question_embedding],

        n_results=3,

        include=["documents", "metadatas"]

    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
You are a college assistant.

Answer ONLY using the context.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt

    )

    print("\n🤖", response.text)

    print("\nSources:")

    for meta in results["metadatas"][0]:

        print(f"📄 {meta['source']} | Page {meta['page']}")

