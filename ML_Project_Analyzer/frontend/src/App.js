import React from "react";
import { Routes, Route } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import VisualizationPage from "./components/VisualizationPage"; // 🔥 Import visualization component

function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/visualize" element={<VisualizationPage />} /> {/* ✅ New route */}
    </Routes>
  );
}

export default App;
