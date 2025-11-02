'use client'
import { useState, ChangeEvent, FormEvent } from "react";

/**
 * Frontend image-processing + single-shot classification flow:
 * 1) User selects a file and clicks Run
 * 2) The app processes the image with a pipeline (client-side)
 * 3) Displays all processed images
 * 4) Sends ONE processed image (default: histogram equalized) to backend /classify
 * 5) Displays the classification result (category/confidence) and DB matches returned by backend
 *
 * Notes:
 * - Processing is done at a reduced working size (maxDimension) for speed.
 * - If payload becomes too big, consider shrinking thumbnails before sending.
 */

type Match = {
  name: string;
  price: number;
  categories: string[];
  image_url?: string;
  _id?: string;
};

type ProcessedResult = {
  method: string;
  image_data?: string; // data URL
  // classification fields are shown separately (single-shot)
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [processed, setProcessed] = useState<ProcessedResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Single-shot classification result from backend
  const [topCategory, setTopCategory] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [matches, setMatches] = useState<Match[] | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Which processed image to send to backend after processing finishes
  const DEFAULT_SEND_METHOD = "histogram_equalization";

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f) setPreview(URL.createObjectURL(f));
    else setPreview(null);
    setProcessed(null);
    setTopCategory(null);
    setConfidence(null);
    setMatches(null);
    setError(null);
  }

  // ---------- Utility: canvas helpers ----------
  async function fileToImage(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = (e) => {
        URL.revokeObjectURL(url);
        reject(new Error("Failed to load image"));
      };
      img.src = url;
    });
  }

  // Draw image to canvas with max dimension to reduce compute
  function drawToCanvas(img: HTMLImageElement, maxDimension = 512) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const ratio = img.width / img.height;
    let w = img.width;
    let h = img.height;
    if (Math.max(w, h) > maxDimension) {
      if (w >= h) {
        w = maxDimension;
        h = Math.round(maxDimension / ratio);
      } else {
        h = maxDimension;
        w = Math.round(maxDimension * ratio);
      }
    }
    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(img, 0, 0, w, h);
    return canvas;
  }

  function canvasToImageData(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d")!;
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  function putImageDataToCanvas(imgData: ImageData) {
    const canvas = document.createElement("canvas");
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    const ctx = canvas.getContext("2d")!;
    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  function canvasToDataURL(canvas: HTMLCanvasElement, type = "image/jpeg", quality = 0.85) {
    return canvas.toDataURL(type, quality);
  }

  // ---------- Basic pixel helpers ----------
  function cloneImageData(src: ImageData) {
    return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
  }

  function getGrayArray(imgData: ImageData) {
    const { data, width, height } = imgData;
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      gray[j] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
  }

  // ---------- Algorithm: Histogram equalization on V channel (HSV) ----------
  function histogramEqualization(imgData: ImageData) {
    // convert to HSV, equalize V, convert back
    const { data, width, height } = imgData;
    const out = cloneImageData(imgData);
    // compute histogram on V
    const hist = new Uint32Array(256);
    const hsvV = new Uint8ClampedArray(width * height);
    let idx = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      const v = Math.round(max * 255);
      hsvV[idx++] = v;
      hist[v]++;
    }
    // cumulative
    const cdf = new Uint32Array(256);
    cdf[0] = hist[0];
    for (let i = 1; i < 256; i++) cdf[i] = cdf[i-1] + hist[i];
    const cdfMin = cdf.find(v => v > 0) || 0;
    const total = width * height;
    // map
    const map = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) {
      map[i] = Math.round((cdf[i] - cdfMin) / (total - cdfMin) * 255);
    }
    // apply mapping to V channel and convert back RGB (approx)
    idx = 0;
    for (let i = 0; i < data.length; i += 4, idx++) {
      // convert pixel to HSV
      let r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      let h = 0, s = 0, v = max;
      const d = max - min;
      if (max !== 0) s = d / max;
      if (d !== 0) {
        switch (max) {
          case r: h = (g - b) / d + (g < b ? 6 : 0); break;
          case g: h = (b - r) / d + 2; break;
          case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
      }
      // replace v with equalized map
      v = map[hsvV[idx]] / 255;
      // HSV->RGB
      let rr=0, gg=0, bb=0;
      if (s === 0) {
        rr = gg = bb = v;
      } else {
        const i_h = Math.floor(h * 6);
        const f = h * 6 - i_h;
        const p = v * (1 - s);
        const q = v * (1 - s * f);
        const t = v * (1 - s * (1 - f));
        switch (i_h % 6) {
          case 0: rr = v; gg = t; bb = p; break;
          case 1: rr = q; gg = v; bb = p; break;
          case 2: rr = p; gg = v; bb = t; break;
          case 3: rr = p; gg = q; bb = v; break;
          case 4: rr = t; gg = p; bb = v; break;
          case 5: rr = v; gg = p; bb = q; break;
        }
      }
      out.data[i] = Math.round(rr * 255);
      out.data[i+1] = Math.round(gg * 255);
      out.data[i+2] = Math.round(bb * 255);
      out.data[i+3] = data[i+3];
    }
    return out;
  }

  // ---------- Brightness & contrast ----------
  function brightnessContrast(imgData: ImageData, brightness = 1.1, contrast = 1.1) {
    // brightness: factor (1.0 no change), contrast: factor (1.0 no change)
    const out = cloneImageData(imgData);
    const b = brightness;
    const c = contrast;
    // standard formula: new = (pixel - 128) * c + 128 * b
    for (let i = 0; i < out.data.length; i += 4) {
      for (let ch = 0; ch < 3; ch++) {
        let px = out.data[i+ch];
        px = (px - 128) * c + 128;
        px = px * b;
        out.data[i+ch] = Math.max(0, Math.min(255, Math.round(px)));
      }
    }
    return out;
  }

  // ---------- Convolution kernel ----------
  function convolve(imgData: ImageData, kernel: number[], kw: number, kh: number, normalize = true) {
    const { width, height } = imgData;
    const src = imgData.data;
    const out = new ImageData(width, height);
    const ksum = kernel.reduce((a,b)=>a+b,0);
    const halfW = Math.floor(kw/2);
    const halfH = Math.floor(kh/2);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0;
        for (let ky = 0; ky < kh; ky++) {
          for (let kx = 0; kx < kw; kx++) {
            const px = x + kx - halfW;
            const py = y + ky - halfH;
            if (px < 0 || px >= width || py < 0 || py >= height) continue;
            const idx = (py * width + px) * 4;
            const kval = kernel[ky*kw + kx];
            r += src[idx] * kval;
            g += src[idx+1] * kval;
            b += src[idx+2] * kval;
          }
        }
        if (normalize && ksum !== 0) { r /= ksum; g /= ksum; b /= ksum; }
        const outIdx = (y * width + x) * 4;
        out.data[outIdx] = Math.max(0, Math.min(255, Math.round(r)));
        out.data[outIdx+1] = Math.max(0, Math.min(255, Math.round(g)));
        out.data[outIdx+2] = Math.max(0, Math.min(255, Math.round(b)));
        out.data[outIdx+3] = 255;
      }
    }
    return out;
  }

  // ---------- Mean (box) blur ----------
  function meanFilter(imgData: ImageData, k = 5) {
    // simple separable approach would be faster, but keep readable: use square kernel
    const kernel = new Array(k*k).fill(1);
    return convolve(imgData, kernel, k, k, true);
  }

  // ---------- Median filter (slow on large images) ----------
  function medianFilter(imgData: ImageData, k = 3) {
    const { width, height } = imgData;
    const out = new ImageData(width, height);
    const half = Math.floor(k/2);
    const src = imgData.data;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const rvals: number[] = [], gvals: number[] = [], bvals: number[] = [];
        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            const px = Math.min(width-1, Math.max(0, x + kx));
            const py = Math.min(height-1, Math.max(0, y + ky));
            const idx = (py * width + px) * 4;
            rvals.push(src[idx]); gvals.push(src[idx+1]); bvals.push(src[idx+2]);
          }
        }
        rvals.sort((a,b)=>a-b); gvals.sort((a,b)=>a-b); bvals.sort((a,b)=>a-b);
        const m = Math.floor(rvals.length/2);
        const outIdx = (y * width + x) * 4;
        out.data[outIdx] = rvals[m];
        out.data[outIdx+1] = gvals[m];
        out.data[outIdx+2] = bvals[m];
        out.data[outIdx+3] = 255;
      }
    }
    return out;
  }

  // ---------- Laplacian sharpen ----------
  function laplacianSharpen(imgData: ImageData) {
    // kernel: center 5, neighbors -1
    const kernel = [
      0, -1, 0,
      -1, 5, -1,
      0, -1, 0
    ];
    return convolve(imgData, kernel, 3, 3, false);
  }

  // ---------- Low-pass (Gaussian-like via box blur multiple times) ----------
  function lowPass(imgData: ImageData) {
    // apply small box blur multiple times approximates gaussian
    let out = imgData;
    out = meanFilter(out, 5);
    out = meanFilter(out, 3);
    return out;
  }

  // ---------- High-pass (original - lowpass) ----------
  function highPass(imgData: ImageData) {
    const { width, height } = imgData;
    const low = lowPass(imgData);
    const out = new ImageData(width, height);
    for (let i = 0; i < out.data.length; i += 4) {
      out.data[i] = Math.max(0, Math.min(255, imgData.data[i] - low.data[i] + 128));
      out.data[i+1] = Math.max(0, Math.min(255, imgData.data[i+1] - low.data[i+1] + 128));
      out.data[i+2] = Math.max(0, Math.min(255, imgData.data[i+2] - low.data[i+2] + 128));
      out.data[i+3] = 255;
    }
    return out;
  }

  // ---------- Sobel edge detector (magnitude) ----------
  function sobelEdges(imgData: ImageData) {
    const { width, height } = imgData;
    const gray = getGrayArray(imgData);
    const out = new ImageData(width, height);
    const gxKernel = [-1,0,1,-2,0,2,-1,0,1];
    const gyKernel = [-1,-2,-1,0,0,0,1,2,1];
    const kw = 3, kh = 3, half = 1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let gx = 0, gy = 0;
        for (let ky = 0; ky < kh; ky++) {
          for (let kx = 0; kx < kw; kx++) {
            const px = Math.min(width-1, Math.max(0, x + kx - half));
            const py = Math.min(height-1, Math.max(0, y + ky - half));
            const val = gray[py*width + px];
            gx += val * gxKernel[ky*kw + kx];
            gy += val * gyKernel[ky*kw + kx];
          }
        }
        const mag = Math.min(255, Math.round(Math.hypot(gx, gy)));
        const idx = (y*width + x)*4;
        out.data[idx] = out.data[idx+1] = out.data[idx+2] = mag;
        out.data[idx+3] = 255;
      }
    }
    return out;
  }

  // ---------- Otsu thresholding ----------
  function otsuThreshold(imgData: ImageData) {
    const { width, height } = imgData;
    const gray = getGrayArray(imgData);
    const hist = new Uint32Array(256);
    for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
    let total = gray.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0, wB = 0, wF = 0;
    let varMax = 0, threshold = 0;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) * (mB - mF);
      if (varBetween > varMax) {
        varMax = varBetween;
        threshold = t;
      }
    }
    const out = new ImageData(width, height);
    for (let i = 0; i < gray.length; i++) {
      const v = gray[i] >= threshold ? 255 : 0;
      out.data[i*4] = out.data[i*4+1] = out.data[i*4+2] = v;
      out.data[i*4+3] = 255;
    }
    return out;
  }

  // ---------- Color quantization (k-means) - lightweight on downsampled pixels ----------
  function colorKMeans(imgData: ImageData, K = 3, iters = 6) {
    const { width, height, data } = imgData;
    // sample pixels (downsample for speed)
    const sample = [];
    const step = Math.max(1, Math.floor(Math.sqrt((width * height) / 2000))); // aim ~2000 samples
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = (y * width + x) * 4;
        sample.push([data[idx], data[idx+1], data[idx+2]]);
      }
    }
    // init centers randomly
    const centers = [];
    for (let k = 0; k < K; k++) centers.push(sample[Math.floor(Math.random() * sample.length)].slice());
    const labels = new Array(sample.length).fill(0);
    for (let iter = 0; iter < iters; iter++) {
      // assign
      for (let i = 0; i < sample.length; i++) {
        let best = 0, bestd = Infinity;
        for (let c = 0; c < K; c++) {
          const d = (sample[i][0]-centers[c][0])**2 + (sample[i][1]-centers[c][1])**2 + (sample[i][2]-centers[c][2])**2;
          if (d < bestd) { bestd = d; best = c; }
        }
        labels[i] = best;
      }
      // update
      const sums = Array.from({length:K}, ()=>[0,0,0,0]);
      for (let i = 0; i < sample.length; i++) {
        const l = labels[i];
        sums[l][0]+= sample[i][0]; sums[l][1]+= sample[i][1]; sums[l][2]+= sample[i][2]; sums[l][3]+=1;
      }
      for (let c = 0; c < K; c++) {
        if (sums[c][3] > 0) {
          centers[c][0] = sums[c][0]/sums[c][3];
          centers[c][1] = sums[c][1]/sums[c][3];
          centers[c][2] = sums[c][2]/sums[c][3];
        }
      }
    }
    // now apply quantization to full image (nearest center)
    const out = new ImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y*width + x)*4;
        const pix = [data[idx], data[idx+1], data[idx+2]];
        let best = 0, bestd = Infinity;
        for (let c = 0; c < K; c++) {
          const d = (pix[0]-centers[c][0])**2 + (pix[1]-centers[c][1])**2 + (pix[2]-centers[c][2])**2;
          if (d < bestd) { bestd = d; best = c; }
        }
        out.data[idx] = Math.round(centers[best][0]);
        out.data[idx+1] = Math.round(centers[best][1]);
        out.data[idx+2] = Math.round(centers[best][2]);
        out.data[idx+3] = 255;
      }
    }
    return out;
  }

  // ---------- Pipeline: name -> function(imgData) => ImageData ----------
  const PIPELINE_FUNCS: [string, (img: ImageData) => ImageData][] = [
    ["original", (img) => cloneImageData(img)],
    ["histogram_equalization", histogramEqualization],
    ["brightness_contrast", (img) => brightnessContrast(img, 1.15, 1.15)],
    ["mean_filter", (img) => meanFilter(img, 5)],
    ["median_filter", (img) => medianFilter(img, 3)],
    ["laplacian_sharpen", laplacianSharpen],
    ["low_pass_filter", lowPass],
    ["high_pass_filter", highPass],
    ["sobel_edges", sobelEdges],
    ["otsu_threshold", otsuThreshold],
    ["color_kmeans", (img) => colorKMeans(img, 4, 6)]
  ];

  // ---------- Convert ImageData to dataURL (canvas) ----------
  function imageDataToDataURL(imgData: ImageData, type: string = "image/jpeg", quality = 0.8) {
    const canvas = putImageDataToCanvas(imgData);
    return canvasToDataURL(canvas, type, quality);
  }

  // Convert dataURL to Blob to send
  async function dataURLToBlob(dataurl: string) {
    const res = await fetch(dataurl);
    return await res.blob();
  }

  // ---------- processImage: runs pipeline and returns array of results {method, dataURL} ----------
  async function processImageFile(file: File) {
    const img = await fileToImage(file);
    const workCanvas = drawToCanvas(img, 512); // keep reasonable size
    const baseImgData = canvasToImageData(workCanvas);

    const outputs: ProcessedResult[] = [];
    for (const [name, fn] of PIPELINE_FUNCS) {
      try {
        const outImgData = fn(baseImgData);
        const dataUrl = imageDataToDataURL(outImgData, "image/jpeg", 0.8);
        outputs.push({ method: name, image_data: dataUrl });
      } catch (e: any) {
        outputs.push({ method: name, image_data: undefined });
      }
    }
    return outputs;
  }

  // ---------- handleSubmit: process first, then send ONE processed image to backend ----------
  async function handleSubmit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    if (!file) return alert("Pick an image first");
    setLoading(true);
    setError(null);
    setProcessed(null);
    setTopCategory(null);
    setConfidence(null);
    setMatches(null);

    try {
      // 1) Run client-side pipeline
      const outputs = await processImageFile(file);
      setProcessed(outputs);

      // 2) Choose one processed image to send (default to histogram_equalization)
      const chosen = outputs.find(o => o.method === DEFAULT_SEND_METHOD) ?? outputs[0];
      if (!chosen || !chosen.image_data) {
        throw new Error("No processed image available to send");
      }

      // Convert chosen dataURL to blob and send to backend
      const blob = await dataURLToBlob(chosen.image_data);
      const formData = new FormData();
      formData.append("file", blob, `${chosen.method}.jpg`);

      const resp = await fetch(`${apiUrl}/classify`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || "Server error");
      }

      const data = await resp.json();

      // Backend may return two formats:
      // 1) { top_category, confidence, matches }  (single-shot)
      // 2) { results: [...] } (server-side multi-processing)
      if (data.results && Array.isArray(data.results)) {
        // server returned many results (older backend) — show first result as classification
        // still we display processed images from client side
        if (data.results.length > 0 && data.results[0].top_category) {
          setTopCategory(data.results[0].top_category);
          setConfidence(data.results[0].confidence ?? null);
          setMatches(data.results[0].matches ?? null);
        }
      } else {
        setTopCategory(data.top_category ?? null);
        setConfidence(typeof data.confidence === "number" ? data.confidence : null);
        setMatches(Array.isArray(data.matches) ? data.matches : null);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: "2rem auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Visual Product Search — client-side preprocessing</h1>
      <p>Upload an image — the app runs several processing techniques locally, shows each processed image, then sends a single selected processed image to the backend for classification and product lookup.</p>

      <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <div style={{ marginTop: 10 }}>
          <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
            {loading ? "Processing..." : "Start"}
          </button>
        </div>
      </form>

      {preview && (
        <div style={{ marginBottom: 12 }}>
          <strong>Original Preview:</strong>
          <div>
            <img src={preview} alt="preview" style={{ maxWidth: 320, borderRadius: 8, marginTop: 8 }} />
          </div>
        </div>
      )}

      {error && <div style={{ color: "red" }}>Error: {error}</div>}

      {processed && (
        <>
          <h2>Processed Images</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
            {processed.map((p, i) => (
              <div key={i} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{p.method}</div>
                {p.image_data ? (
                  <img src={p.image_data} alt={p.method} style={{ width: "100%", height: 140, objectFit: "cover", borderRadius: 6, marginTop: 8 }} />
                ) : (
                  <div style={{ color: "#b00", marginTop: 8 }}>Processing failed</div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {topCategory && (
        <div style={{ marginTop: 20 }}>
          <h2>Classification result</h2>
          <div>
            <strong>Top category:</strong> {topCategory} <br />
            <strong>Confidence:</strong> {confidence !== null ? (confidence * 100).toFixed(2) + "%" : "—"}
          </div>

          <h3 style={{ marginTop: 12 }}>Matches from DB</h3>
          {(!matches || matches.length === 0) ? (
            <div style={{ color: "#666" }}>No matches found for this category.</div>
          ) : (
            <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
              {matches.map(m => (
                <div key={m._id} style={{ display: "flex", gap: 10, alignItems: "center", border: "1px solid #eee", padding: 8, borderRadius: 6 }}>
                  {m.image_url ? <img src={m.image_url} alt={m.name} style={{ width: 56, height: 56, objectFit: "cover", borderRadius: 6 }} /> : null}
                  <div>
                    <div style={{ fontWeight: 700 }}>{m.name}</div>
                    <div style={{ color: "#666" }}>${m.price}</div>
                    <div style={{ fontSize: 12, color: "#999" }}>{(m.categories || []).join(", ")}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
