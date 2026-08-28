from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(file_path: str, chunk_size: int = 400, chunk_overlap: int = 60):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    chunks = load_and_chunk("data/sample_doc.txt")
    print(f"Nombre de chunks : {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} ---\n{c.page_content[:150]}...\n")