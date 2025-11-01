'use client'
import { useState, ChangeEvent, FormEvent } from "react";

type Response= {
        top_category: string,
        confidence: number,
        matches: any[]
    } | null;

export default function Home() {
  const [file, setFile] = useState<File|null>(null);
  const [preview, setPreview] = useState<string|null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Response>(null);
  const [error, setError] = useState(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files![0];
    setFile(f);
    if (f) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview(null);
    }
    setResult(null);
  }

  async function handleSubmit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    if (!file) return alert("Pick an image first");
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`${apiUrl}/classify`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Server error");
      }
      const data = await resp.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Visual Product Search (demo)</h1>
      <p>Upload a product image — the system finds the top predicted category and shows DB items in that category.</p>

      <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <div style={{ marginTop: 10 }}>
          <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {preview && (
        <div style={{ marginBottom: 12 }}>
          <strong>Preview:</strong>
          <div>
            <img src={preview} alt="preview" style={{ maxWidth: 320, borderRadius: 8, marginTop: 8 }} />
          </div>
        </div>
      )}

      {error && <div style={{ color: "red" }}>Error: {error}</div>}

      {result && (
        <div>
          <h2>Top Prediction</h2>
          <div>
            <strong>Category:</strong> {result.top_category} <br />
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
          </div>

          <h3 style={{ marginTop: 16 }}>Matches from DB</h3>
          {result.matches.length === 0 && <div>No items found for this category.</div>}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
            {result.matches.map((m) => (
              <div key={m._id} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12, width: 220 }}>
                {m.image_url ? <img src={m.image_url} alt={m.name} style={{ width: "100%", height: 120, objectFit: "cover", borderRadius: 6 }} /> : null}
                <div style={{ marginTop: 8 }}>
                  <strong>{m.name}</strong>
                </div>
                <div>Price: ${m.price}</div>
                <div style={{ fontSize: 12, color: "#555" }}>Categories: {m.categories.join(", ")}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
