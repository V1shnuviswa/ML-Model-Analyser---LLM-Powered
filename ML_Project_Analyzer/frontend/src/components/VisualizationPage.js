import React, { useState } from "react";
import axios from "axios";
import { FaCloudUploadAlt } from "react-icons/fa";
import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom"; // <- For navigation
import "../App.css";

function VisualizationPage() {
  const [file, setFile] = useState(null);
  const [visuals, setVisuals] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const navigate = useNavigate(); // <-- Hook to navigate

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setVisuals(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) return setError("⚠️ Please upload a Python file!");

    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5000/model-evaluation", formData);
      setVisuals(res.data.visualization);
    } catch (err) {
      console.error(err);
      setError("❌ Error fetching visualizations. Check server or file format.");
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate(-1); // Navigate to previous page, or use: navigate("/upload") for a specific route
  };

  return (
    <div className="upload-container">
      <div className="upload-box">
        <h1 className="upload-title">Model Visualization Dashboard 📊</h1>
        <p className="upload-subtitle">Upload a Python ML model file to generate evaluation visualizations.</p>

        <label className="file-upload">
          <FaCloudUploadAlt className="upload-icon" />
          <span>{file ? file.name : "Click to upload .py file"}</span>
          <input type="file" accept=".py" onChange={handleFileChange} className="file-input" />
        </label>

        <button
          onClick={handleUpload}
          className={`upload-button ${loading || !file ? "disabled" : ""}`}
          disabled={loading || !file}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="animate-spin" size={18} />
              Evaluating...
            </span>
          ) : (
            "Upload and Visualize"
          )}
        </button>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}
      </div>

      {visuals && (
        <div className="results-box">
          <h2 className="upload-title">📈 Visual Analysis</h2>
          <div className="dashboard-container">
            {visuals.accuracy && (
              <div className="chart-box">
                <h3 className="upload-subtitle">Accuracy</h3>
                <img
                  src={`data:image/png;base64,${visuals.accuracy}`}
                  alt="Accuracy"
                  className="rounded-xl w-full"
                />
              </div>
            )}

            {visuals.f1_score && (
              <div className="chart-box">
                <h3 className="upload-subtitle">F1 Score</h3>
                <img
                  src={`data:image/png;base64,${visuals.f1_score}`}
                  alt="F1 Score"
                  className="rounded-xl w-full"
                />
              </div>
            )}

            {visuals.energy && (
              <div className="chart-box">
                <h3 className="upload-subtitle">Energy Consumption</h3>
                <img
                  src={`data:image/png;base64,${visuals.energy}`}
                  alt="Energy"
                  className="rounded-xl w-full"
                />
              </div>
            )}

            {visuals.confusion_matrix && (
              <div className="chart-box">
                <h3 className="upload-subtitle">Confusion Matrix</h3>
                <img
                  src={`data:image/png;base64,${visuals.confusion_matrix}`}
                  alt="Confusion Matrix"
                  className="rounded-xl w-full"
                />
              </div>
            )}
          </div>

          {/* Back Button */}
          <div className="mt-6 text-center">
            <button
              onClick={handleBack}
              className="upload-button bg-gray-200 text-black hover:bg-gray-300"
            >
              🔙 Back
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default VisualizationPage;
