from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

RAG_PROMPT_TEMPLATE = """Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni.
Si la réponse ne se trouve pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""

app = FastAPI()

# Autorise le frontend React (qui tournera sur un autre port) à appeler cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargé une seule fois au démarrage du serveur, pas à chaque question
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("../faiss_index", embedding_model, allow_dangerous_deserialization=True)


class Question(BaseModel):
    question: str


def build_prompt(question: str, retrieved_docs) -> str:
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_docs)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


ddef call_llm(prompt: str) -> str:
    client = InferenceClient(token=HF_TOKEN)
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct:fastest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content


@app.post("/ask")
def ask(payload: Question):
    retrieved = vectorstore.similarity_search_with_score(payload.question, k=3)
    prompt = build_prompt(payload.question, retrieved)
    answer = call_llm(prompt)
    sources = [doc.page_content[:150] for doc, _ in retrieved]
    return {"answer": answer, "sources": sources}