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


def build_prompt(question: str, retrieved_docs) -> str:
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_docs)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def call_llm(prompt: str) -> str:
    client = InferenceClient(
        provider="featherless-ai",
        token=HF_TOKEN,
    )
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


def generate_answer(question: str, retrieved_docs) -> str:
    prompt = build_prompt(question, retrieved_docs)
    return call_llm(prompt)


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    question = "Qu'est-ce que le chunking et pourquoi est-il important ?"
    retrieved = vectorstore.similarity_search_with_score(question, k=3)

    print(f"Question : {question}\n")
    print("Réponse générée :\n")
    print(generate_answer(question, retrieved))