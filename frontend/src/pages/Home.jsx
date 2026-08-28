import { useState } from "react";

function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setSources([]);
    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setAnswer("Erreur : impossible de contacter le serveur backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>RAG Q&amp;A</h1>
      <p className="subtitle">Pose une question sur tes documents</p>
      <div className="input-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Qu'est-ce que le chunking ?"
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "..." : "Envoyer"}
        </button>
      </div>
      {answer && (
        <div className="answer-box">
          <h2>Réponse</h2>
          <p>{answer}</p>
        </div>
      )}
      {sources.length > 0 && (
        <div className="sources-box">
          <h2>Sources utilisées</h2>
          {sources.map((s, i) => (
            <div key={i} className="source-item">{s}...</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Home;