import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  function onSelect(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API_URL}/api/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <h1>AI Based Crop Detection</h1>
      <p className="subtitle">Upload a leaf photo to identify the crop and detect disease.</p>

      <form onSubmit={onSubmit} className="card">
        <input type="file" accept="image/*" onChange={onSelect} />
        {preview && <img src={preview} alt="Selected leaf" className="preview" />}
        <button type="submit" disabled={!file || loading}>
          {loading ? "Analysing..." : "Detect"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="card result">
          <h2>{result.crop}</h2>
          <p className={result.healthy ? "healthy" : "diseased"}>
            {result.condition} - {(result.confidence * 100).toFixed(1)}% confidence
          </p>
          <p>{result.remedy}</p>
          <ul>
            {result.alternatives.map((alt) => (
              <li key={alt.label}>
                {alt.label} ({(alt.confidence * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
