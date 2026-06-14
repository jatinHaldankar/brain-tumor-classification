const CLASS_COLORS = {
  "Glioma":     "#ef4444",
  "Meningioma": "#f97316",
  "No Tumor":   "#22c55e",
  "Pituitary":  "#8b5cf6",
};

const uploadZone   = document.getElementById("upload-zone");
const fileInput    = document.getElementById("file-input");
const previewWrap  = document.getElementById("preview-wrapper");
const previewImg   = document.getElementById("preview-img");
const clearBtn     = document.getElementById("clear-btn");
const analyseBtn   = document.getElementById("analyse-btn");
const loading      = document.getElementById("loading");
const results      = document.getElementById("results");
const errorBox     = document.getElementById("error-box");

let base64Image = null;

// ── Drag & drop ──
uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener("click", resetUI);

function handleFile(file) {
  if (!file.type.startsWith("image/")) { showError("Please upload a valid image file."); return; }
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    // Strip data URL prefix to get raw base64
    base64Image = dataUrl.split(",")[1];
    previewImg.src = dataUrl;
    previewWrap.style.display = "block";
    analyseBtn.style.display = "block";
    results.style.display = "none";
    errorBox.style.display = "none";
  };
  reader.readAsDataURL(file);
}

analyseBtn.addEventListener("click", async () => {
  if (!base64Image) return;

  // Show loading state
  analyseBtn.disabled = true;
  loading.style.display = "block";
  results.style.display = "none";
  errorBox.style.display = "none";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64Image }),
    });

    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || "Server error. Please try again."); return; }

    renderResults(data);
  } catch (err) {
    showError("Could not reach the server. Is the FastAPI backend running?");
  } finally {
    loading.style.display = "none";
    analyseBtn.disabled = false;
  }
});

function renderResults(data) {
  const { prediction, confidence, all_probs, info } = data;
  const color = CLASS_COLORS[prediction] || "#6366f1";

  // Header
  document.getElementById("result-header").style.background =
    `linear-gradient(135deg, ${color}18, ${color}08)`;
  document.getElementById("result-header").style.borderColor = `${color}30`;
  document.getElementById("class-name").textContent = prediction;
  document.getElementById("class-name").style.color = color;
  document.getElementById("confidence-pct").textContent = `${confidence.toFixed(1)}%`;
  document.getElementById("confidence-pct").style.color = color;

  // Severity badge
  const badge = document.getElementById("severity-badge");
  const severityColors = { "High": "#ef4444", "Medium": "#f97316", "None": "#22c55e" };
  const sc = severityColors[info.severity] || "#6366f1";
  badge.textContent = `● Severity: ${info.severity}`;
  badge.style.background = `${sc}20`;
  badge.style.color = sc;
  badge.style.border = `1px solid ${sc}40`;

  // Description
  document.getElementById("desc-box").textContent = info.description;

  // Probability bars
  const container = document.getElementById("prob-bars");
  container.innerHTML = "";
  // Sort by probability descending
  const sorted = Object.entries(all_probs).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([cls, pct]) => {
    const c = CLASS_COLORS[cls] || "#6366f1";
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${cls}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" data-pct="${pct}" style="background:${c};"></div>
      </div>
      <span class="prob-pct" style="color:${c}">${pct.toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });

  results.style.display = "block";

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach(bar => {
      bar.style.width = bar.dataset.pct + "%";
    });
  });
}

function showError(msg) {
  loading.style.display = "none";
  errorBox.textContent = "⚠ " + msg;
  errorBox.style.display = "block";
  analyseBtn.disabled = false;
}

function resetUI() {
  base64Image = null;
  fileInput.value = "";
  previewWrap.style.display = "none";
  analyseBtn.style.display = "none";
  results.style.display = "none";
  errorBox.style.display = "none";
}
