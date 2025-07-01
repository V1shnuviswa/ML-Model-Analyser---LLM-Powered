import React, { useState } from "react";
import { motion } from "framer-motion";
import { FaCloudUploadAlt } from "react-icons/fa";
import { MdCheckCircle, MdError } from "react-icons/md";
import axios from "axios";
import "../App.css";

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data);
    } catch (error) {
      setError("Upload failed. Please try again.");
      console.error("Upload failed:", error);
    }
    setLoading(false);
  };

  const handleSearch = async () => {
    if (!question.trim()) return;

    try {
      const response = await axios.post("http://localhost:5000/search", { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("Error fetching response:", error);
      setAnswer("Error fetching answer. Try again.");
    }
  };

  const handleDownload = () => {
    fetch("http://localhost:5000/generate_pdf")
      .then((response) => response.blob())
      .then((blob) => {
        const link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "ML_Project_Analysis.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      })
      .catch((error) => console.error("Error downloading PDF:", error));
  };

  const renderSuggestions = (suggestions) => {
    return (
      <ul className="list-disc ml-6">
        {Object.entries(suggestions).map(([key, value]) => (
          <li key={key}>
            <strong>{key.replace(/_/g, " ")}:</strong>
            <ul className="ml-4 list-square">
              {Array.isArray(value)
                ? value.map((item, index) => <li key={index}>{item}</li>)
                : <li>{value.toString()}</li>}
            </ul>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="upload-container">
      <motion.div className="upload-box" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
        <h2 className="upload-title">Upload Your Model File</h2>
        <p className="upload-subtitle">Select a Python file for AI analysis</p>

        <label className="file-upload">
          <FaCloudUploadAlt className="upload-icon" />
          <input type="file" className="file-input" onChange={handleFileChange} />
          <span>{file ? file.name : "Click to select a file"}</span>
        </label>

        {error && <p className="error-message"><MdError className="error-icon" /> {error}</p>}

        <button className={`upload-button ${loading ? "disabled" : ""}`} onClick={handleUpload} disabled={loading}>
          {loading ? "Uploading..." : "Upload File"}
        </button>
      </motion.div>

      {results && (
        <motion.div className="results-box" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
          <h3 className="results-title"><MdCheckCircle className="success-icon" /> Analysis Results</h3>
          <div className="results-content">
            <h4>📊 <strong>Model Analysis:</strong></h4>
            {results.model_analysis ? renderSuggestions(results.model_analysis) : <p>No analysis available.</p>}

            <h4>🔍 <strong>Code Analysis:</strong></h4>
            {results.code_analysis ? renderSuggestions(results.code_analysis) : <p>No code insights available.</p>}

            <h4>🤖 <strong>AI-Based Suggestions:</strong></h4>
            {results.ai_suggestions ? renderSuggestions(results.ai_suggestions) : <p>No AI suggestions available.</p>}
          </div>

          {/* PDF Download Button */}
          <button
            onClick={handleDownload}
            style={{
              marginTop: "20px",
              padding: "10px 20px",
              background: "#007BFF",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Download Report PDF
          </button>

          {/* Visualize Button */}
          <button
            onClick={() => window.location.href = "/visualize"}
            style={{
              marginTop: "10px",
              marginLeft: "10px",
              padding: "10px 20px",
              background: "#28a745",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Visualize Model Evaluation
          </button>
        </motion.div>
      )}

      {results && (
        <motion.div className="search-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
          <h3 className="search-title">🔎 Ask a Question About the Model</h3>
          <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask something about the uploaded model..." className="search-input" />
          <button onClick={handleSearch} className="search-button">Search</button>
          {answer && <p className="search-result">Answer: {answer}</p>}
        </motion.div>
      )}
    </div>
  );
};

export default UploadPage;
