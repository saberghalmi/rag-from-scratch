from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from importlib import util


def build_and_save_index(chunks, save_path: str = "faiss_index"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(save_path)
    print(f"Index sauvegardé dans ./{save_path}")
    return vectorstore


if __name__ == "__main__":
    spec = util.spec_from_file_location("chunking_mod", "1_chunking.py")
    chunking_mod = util.module_from_spec(spec)
    spec.loader.exec_module(chunking_mod)

    chunks = chunking_mod.load_and_chunk("data/sample_doc.txt")
    build_and_save_index(chunks)