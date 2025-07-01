import React, { useState } from "react";
import axios from "axios";

const Chatbot = () => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAskQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/chat", { question });
      setResponse(res.data.response);
    } catch (error) {
      setResponse("Error fetching response. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      <h3>🤖 AI Chatbot</h3>
      <input
        type="text"
        placeholder="Ask a question about ML..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button onClick={handleAskQuestion} disabled={loading}>
        {loading ? "Thinking..." : "Ask"}
      </button>
      {response && <div className="chat-response">{response}</div>}
    </div>
  );
};

export default Chatbot;
