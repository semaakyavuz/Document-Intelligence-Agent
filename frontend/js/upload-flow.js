/* upload-flow.js
   Gercek yukleme/SSE mantigi (POST /invoices/stream, POST /invoices/bulk).
   Mantigin cogu onceki tek-dosyalik upload.html'den tasindi/korundu (dropzone,
   sekmeler, toplu-secim birikimi - native <input multiple> her yeniden dialog
   acilisinda onceki secimi degistirdigi icin kendi JS state'imizde biriktiriyoruz,
   bkz. addBulkFiles), sadece sonuc paneli final_report'un tum alanlarini (risk_score,
   risk_level, explanation, validation_checklist, price_history, execution_trace)
   gosterecek sekilde zenginlestirildi. */

const STEP_LABELS = {
  vision: "Görsel okundu",
  rag: "Kurallar getirildi",
  validation: "Doğrulandı",
  report: "Rapor hazır",
};
const RISK_LABELS = { passed: "Geçti", review_required: "İncelenmeli", rejected: "Reddedildi" };
const MAX_BULK_FILES = 10;

/* ---------------- Sekmeler ---------------- */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = true; });
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).hidden = false;
  });
});

/* ---------------- Tek Fatura ---------------- */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const stepsEl = document.getElementById("steps");
const resultCard = document.getElementById("resultCard");
const errorCard = document.getElementById("errorCard");
const errorMessage = document.getElementById("errorMessage");
const newUploadBtn = document.getElementById("newUploadBtn");

// Tiklama zaten <label for="fileInput"> ile native olarak calisiyor (bkz. upload.html);
// burada sadece surukle-birak dinleyicileri gerekiyor.
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
});
newUploadBtn.addEventListener("click", resetUI);

document.querySelectorAll(".sample-card").forEach((card) => {
  card.addEventListener("click", async () => {
    const src = card.dataset.src;
    const name = card.dataset.name;
    const blob = await (await fetch(src)).blob();
    uploadFile(new File([blob], name, { type: blob.type || "image/png" }));
  });
});

function resetUI() {
  document.querySelectorAll(".step-box").forEach((box) => {
    box.classList.remove("done");
    box.querySelector(".status").textContent = "Bekliyor";
  });
  resultCard.hidden = true;
  errorCard.hidden = true;
  newUploadBtn.hidden = true;
  dropzone.hidden = false;
  fileInput.value = "";
  document.getElementById("rChecklistWrap").hidden = true;
  document.getElementById("rAnomaliesWrap").hidden = true;
  document.getElementById("rPriceHistoryWrap").hidden = true;
  document.getElementById("rTraceWrap").hidden = true;
}

function markStepDone(step) {
  const box = stepsEl.querySelector(`[data-step="${step}"]`);
  if (!box) return;
  box.classList.add("done");
  box.querySelector(".status").textContent = STEP_LABELS[step] || "Tamamlandı";
}

function renderChecklist(list) {
  const wrap = document.getElementById("rChecklistWrap");
  const ul = document.getElementById("rChecklist");
  ul.innerHTML = "";
  if (!list || list.length === 0) { wrap.hidden = true; return; }
  wrap.hidden = false;
  list.forEach((item) => {
    const li = document.createElement("li");
    const statusClass = item.passed ? "pass" : "fail";
    const statusText = item.passed ? "Geçti" : "Başarısız";
    const rule = document.createElement("span");
    rule.className = "check-status " + statusClass;
    rule.textContent = statusText;
    const ruleName = document.createElement("span");
    ruleName.className = "check-rule";
    ruleName.textContent = item.rule;
    const message = document.createElement("span");
    message.textContent = item.message;
    li.append(rule, ruleName, message);
    ul.appendChild(li);
  });
}

function renderAnomalies(list) {
  const wrap = document.getElementById("rAnomaliesWrap");
  const ul = document.getElementById("rAnomalies");
  ul.innerHTML = "";
  if (!list || list.length === 0) { wrap.hidden = true; return; }
  wrap.hidden = false;
  list.forEach((a) => {
    const li = document.createElement("li");
    li.textContent = a.message || JSON.stringify(a);
    ul.appendChild(li);
  });
}

function renderPriceHistory(list) {
  const wrap = document.getElementById("rPriceHistoryWrap");
  const ul = document.getElementById("rPriceHistory");
  ul.innerHTML = "";
  if (!list || list.length === 0) { wrap.hidden = true; return; }
  wrap.hidden = false;
  list.forEach((entry) => {
    const li = document.createElement("li");
    const prices = (entry.previous_prices || []).map((p) => `${p} TL`).join(", ");
    li.textContent = `${entry.description}: ${prices}`;
    ul.appendChild(li);
  });
}

function renderTrace(trace) {
  const wrap = document.getElementById("rTraceWrap");
  const container = document.getElementById("rTrace");
  container.innerHTML = "";
  if (!trace || trace.length === 0) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const max = Math.max(...trace.map((t) => t.duration_ms || 0), 1);
  trace.forEach((t) => {
    const row = document.createElement("div");
    row.className = "trace-row";
    const pct = Math.max(2, Math.round((t.duration_ms / max) * 100));

    const label = document.createElement("span");
    label.className = "trace-label";
    label.textContent = t.step;
    const track = document.createElement("span");
    track.className = "trace-track";
    const fill = document.createElement("span");
    fill.className = "trace-fill";
    fill.style.width = pct + "%";
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "trace-value";
    value.textContent = `${t.duration_ms} ms`;

    row.append(label, track, value);
    container.appendChild(row);
  });
}

function showResult(report) {
  document.getElementById("rInvoiceNo").textContent = report.invoice_no ?? "-";
  document.getElementById("rSeller").textContent = report.seller_name ?? "-";
  document.getElementById("rTotal").textContent = report.grand_total != null ? `${report.grand_total} TL` : "-";

  const validEl = document.getElementById("rValid");
  validEl.innerHTML = report.is_valid
    ? '<span class="check-status pass">Geçerli</span>'
    : '<span class="check-status fail">Anomali var</span>';

  const badge = document.getElementById("rRiskBadge");
  const level = report.risk_level || "review_required";
  badge.className = `risk-badge risk-badge--${level}`;
  document.getElementById("rRiskScore").textContent = report.risk_score != null ? `${report.risk_score}/100` : "-";
  document.getElementById("rRiskLevel").textContent = RISK_LABELS[level] || level;

  document.getElementById("rExplanation").textContent = report.explanation || "";

  renderChecklist(report.validation_checklist);
  renderAnomalies(report.anomalies);
  renderPriceHistory(report.price_history);
  renderTrace(report.execution_trace);

  resultCard.hidden = false;
  newUploadBtn.hidden = false;
}

function showError(message) {
  errorMessage.textContent = message;
  errorCard.hidden = false;
  newUploadBtn.hidden = false;
}

async function uploadFile(file) {
  resetUI();
  dropzone.hidden = true;

  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch("/invoices/stream", { method: "POST", body: formData });
  } catch (err) {
    showError("Sunucuya ulaşılamadı: " + err.message);
    return;
  }

  if (!response.ok || !response.body) {
    showError(`Sunucu hatası (HTTP ${response.status})`);
    return;
  }

  for await (const payload of readSseEvents(response)) {
    if (payload.step === "error") {
      showError(payload.message || "Bilinmeyen hata");
    } else if (payload.step === "final") {
      showResult(payload.report);
    } else {
      markStepDone(payload.step);
    }
  }
}

/* ---------------- Ortak: SSE okuma ---------------- */
async function* readSseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (!rawEvent.startsWith("data:")) continue;
      try {
        yield JSON.parse(rawEvent.slice(5).trim());
      } catch {
        /* bozuk satır, atla */
      }
    }
  }
}

/* ---------------- Toplu Yükleme ----------------
   <input multiple> tek bir dialog oturumunda (Ctrl/Shift ile) secilen dosyalar icin doğru
   calisiyor - ama dialog HER YENIDEN ACILISINDA onceki secimin tamamini degistiriyor (bu
   native tarayici davranisi, canlica dogrulandi). Cozum: secimleri kendi JS state'imizde
   (selectedBulkFiles) biriktiriyoruz, native input'un FileList'ine degil ona guveniyoruz;
   her change'den sonra input.value sifirlanir ki ayni dosya sonradan tekrar secilebilsin. */
const bulkInput = document.getElementById("bulkInput");
const bulkWarning = document.getElementById("bulkWarning");
const bulkSubmitBtn = document.getElementById("bulkSubmitBtn");
const bulkDemoBtn = document.getElementById("bulkDemoBtn");
const bulkRows = document.getElementById("bulkRows");
const bulkSummary = document.getElementById("bulkSummary");

/* Sunucudaki hazir demo seti: frontend/samples/bulk/ altindaki 10 dosya (6 sentetik
   farkli sablon + 4 gercek dunya/Kaggle). Link paylasilan biri kendi dosyasi olmadan
   tek tiklamayla tam kapasiteli bir demo gorebilsin diye. */
const BULK_DEMO_FILES = [
  "sentetik_classic_1.png", "sentetik_classic_2.png",
  "sentetik_compact_1.png", "sentetik_compact_2.png",
  "sentetik_letterhead_1.png", "sentetik_letterhead_2.png",
  "gercek_batch1_0001.jpg", "gercek_batch1_0005.jpg",
  "gercek_batch1_0010.jpg", "gercek_batch1_0015.jpg",
];

bulkDemoBtn.addEventListener("click", async () => {
  bulkDemoBtn.disabled = true;
  try {
    const files = await Promise.all(
      BULK_DEMO_FILES.map(async (name) => {
        const response = await fetch(`/static/samples/bulk/${name}`);
        if (!response.ok) throw new Error(`${name} yüklenemedi (HTTP ${response.status})`);
        const blob = await response.blob();
        return new File([blob], name, { type: blob.type || "image/jpeg" });
      })
    );
    await uploadBulk(files);
  } catch (err) {
    bulkSummary.style.display = "block";
    bulkSummary.textContent = "Örnek dosyalar getirilemedi: " + err.message;
  } finally {
    bulkDemoBtn.disabled = false;
  }
});

let selectedBulkFiles = [];

function fileKey(file) {
  return `${file.name}::${file.size}::${file.lastModified}`;
}

function addBulkFiles(files) {
  const existingKeys = new Set(selectedBulkFiles.map(fileKey));
  for (const file of files) {
    const key = fileKey(file);
    if (!existingKeys.has(key)) {
      selectedBulkFiles.push(file);
      existingKeys.add(key);
    }
  }
  renderBulkRows({ removable: true });
  updateBulkSubmitState();
}

function updateBulkSubmitState() {
  const count = selectedBulkFiles.length;
  const tooMany = count > MAX_BULK_FILES;
  bulkWarning.style.display = tooMany ? "block" : "none";
  bulkSubmitBtn.disabled = count === 0 || tooMany;
}

bulkInput.addEventListener("change", () => {
  addBulkFiles(Array.from(bulkInput.files));
  bulkInput.value = "";
});

bulkSubmitBtn.addEventListener("click", () => uploadBulk(selectedBulkFiles));

function bulkRowId(index) {
  return `bulk-row-${index}`;
}

function renderBulkRows({ removable }) {
  bulkRows.innerHTML = "";
  bulkSummary.style.display = "none";
  selectedBulkFiles.forEach((file, i) => {
    const row = document.createElement("div");
    row.className = "bulk-row";
    row.id = bulkRowId(i + 1);
    const removeBtn = removable
      ? `<button type="button" class="bulk-remove" data-index="${i}" title="Kaldır">×</button>`
      : "";
    row.innerHTML = `
      <span class="bulk-file">${i + 1}. ${file.name}</span>
      <span class="bulk-status">Bekliyor ${removeBtn}</span>
    `;
    bulkRows.appendChild(row);
  });
  if (removable) {
    bulkRows.querySelectorAll(".bulk-remove").forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedBulkFiles.splice(Number(btn.dataset.index), 1);
        renderBulkRows({ removable: true });
        updateBulkSubmitState();
      });
    });
  }
}

function updateBulkRow(payload) {
  const row = document.getElementById(bulkRowId(payload.index));
  if (!row) return;
  const statusEl = row.querySelector(".bulk-status");

  if (payload.status === "done") {
    row.classList.add("done");
    const r = payload.report || {};
    statusEl.textContent = r.summary || "Tamamlandı";
  } else {
    row.classList.add("error");
    statusEl.textContent = payload.message || "Hata";
  }
}

function showBulkSummary(payload) {
  bulkSummary.style.display = "block";
  bulkSummary.textContent = `Toplam: ${payload.processed + payload.failed} · Başarılı: ${payload.processed} · Hatalı: ${payload.failed}`;
}

async function uploadBulk(files) {
  if (files.length === 0 || files.length > MAX_BULK_FILES) return;

  bulkSubmitBtn.disabled = true;
  bulkInput.disabled = true;
  selectedBulkFiles = files;
  renderBulkRows({ removable: false });

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  let response;
  try {
    response = await fetch("/invoices/bulk", { method: "POST", body: formData });
  } catch (err) {
    bulkSummary.style.display = "block";
    bulkSummary.textContent = "Sunucuya ulaşılamadı: " + err.message;
    bulkSubmitBtn.disabled = false;
    bulkInput.disabled = false;
    return;
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      /* govde JSON degilse HTTP kodu yeterli */
    }
    bulkSummary.style.display = "block";
    bulkSummary.textContent = "İşlem başarısız: " + detail;
    bulkSubmitBtn.disabled = false;
    bulkInput.disabled = false;
    return;
  }

  for await (const payload of readSseEvents(response)) {
    if (payload.step === "batch_complete") {
      showBulkSummary(payload);
    } else {
      updateBulkRow(payload);
    }
  }

  selectedBulkFiles = [];
  bulkInput.disabled = false;
  updateBulkSubmitState();
}
