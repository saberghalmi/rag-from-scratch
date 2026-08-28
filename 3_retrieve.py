from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_index(save_path: str, embedding_model):
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)


def retrieve(vectorstore, question: str, k: int = 3):
    return vectorstore.similarity_search_with_score(question, k=k)


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = load_index("faiss_index", embedding_model)

    question = "Qu'est-ce que le chunking et pourquoi est-il important ?"
    results = retrieve(vectorstore, question, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"--- Résultat {i+1} (score={score:.4f}) ---\n{doc.page_content[:200]}...\n")