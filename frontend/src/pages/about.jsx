function About() {
  return (
    <div className="container">
      <h1>À propos</h1>
      <p className="subtitle">
        Ce projet est un système RAG (Retrieval-Augmented Generation) construit
        avec LangChain, FAISS, HuggingFace et FastAPI.
      </p>
      <div className="answer-box">
        <h2>Stack utilisée</h2>
        <p>Chunking, Embeddings, Index vectoriel FAISS, LLM via HuggingFace Inference API.</p>
      </div>
    </div>
  );
}

export default About;