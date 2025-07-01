import { useState } from "react";

function SearchBar() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleSearch = async () => {
    if (!query.trim()) return;  // Prevent empty queries

    try {
      const res = await fetch("http://localhost:5000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      setResponse(data.response || "No answer found.");
    } catch (error) {
      setResponse("Error fetching data. Try again.");
      console.error("Search error:", error);
    }
  };

  return (
    <div className="p-4">
      <input
        type="text"
        placeholder="Ask me anything about ML..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="border p-2 w-full"
      />
      <button onClick={handleSearch} className="bg-blue-500 text-white p-2 mt-2">
        Search
      </button>
      {response && <p className="mt-4">{response}</p>}
    </div>
  );
}

export default SearchBar;
