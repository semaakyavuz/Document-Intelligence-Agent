/* home.js
   AI Pipeline / Dogrulama Motoru / Mimari bolumlerinin ACIKLAYICI (statik) etkilesimi.
   Nav'in kendisi duz <a href="#..."> - kaydirma CSS'teki "scroll-behavior: smooth" ile
   calisiyor, ayrica JS gerekmiyor.

   Buradaki tum icerik app/ altindaki gercek koddan alinmistir (dosya/sinif adlari asagida
   belirtildigi gibi); "ornek veri" diye isaretlenenler disinda hicbir sey uydurulmamistir.
*/

/* ---------------- AI Pipeline ---------------- */
const AGENT_DETAILS = {
  vision: {
    title: "Vision Agent",
    meta: "app/agents/vision_agent.py — VisionAgent",
    fields: [
      ["Girdi", "state.image_path (yüklenen görsel)"],
      ["Sağlayıcı", "Yapılandırılabilir (.env → LLM_PROVIDER) — şu an Gemini (gemini-3.5-flash-lite); Ollama/Groq'a da geçilebilir"],
      ["Süre", "Gerçek ölçümlerde görsel başına ortalama ~2-5 saniye (Gemini)"],
      ["Çıktı", "state.raw_extraction (JSON)"],
    ],
    extra: `<p>EXTRACTION_PROMPT modele tam olarak hangi şemayı, hangi formatta (tarih DD.MM.YYYY, KDV oranı ondalık kesir) döndüreceğini söylüyor; cevap üçlü-backtick'li bir kod bloğu içinde ya da düz metin arasında gelebiliyor, <code>JsonResponseParser</code> bunu ayıklıyor.</p>`,
    sample: {
      label: "Örnek veri — gerçek şemaya uygun, canlı bir koşudan değil",
      json: {
        invoice_no: "2025123456",
        invoice_date: "14.03.2025",
        seller_name: "Ege Ofis Malzemeleri A.Ş.",
        seller_tax_no: "1234567890",
        buyer_name: "Deniz Yazılım Ltd.",
        items: [
          { description: "Klavye", quantity: 2, unit_price: 450.0, vat_rate: 0.20, line_total: 900.0, vat_amount: 180.0 },
        ],
        subtotal: 900.0,
        vat_total: 180.0,
        grand_total: 1080.0,
      },
    },
  },
  rag: {
    title: "RAG Agent",
    meta: "app/agents/rag_agent.py — RAGAgent, app/mcp/rules_server.py",
    fields: [
      ["Girdi", "state.raw_extraction"],
      ["Protokol", "MCP (Model Context Protocol) — stdio alt süreç"],
      ["Çıktı", "state.retrieved_rules — list[{text, score, source}]"],
    ],
    flow: ["Fatura verisi", "RuleQueryBuilder", "MCP Rules Server", "Embedding Provider", "ChromaDB araması", "İlgili kurallar"],
    extra: `<p>RAG'ın embed+arama mantığı aynı süreçte bir fonksiyon değil — <code>python -m app.mcp.rules_server</code> olarak başlatılan <strong>bağımsız bir alt süreç</strong>; RAGAgent buna resmi <code>mcp</code> SDK'sının stdio client'ıyla bağlanıp <code>query_rules</code> aracını çağırıyor. Kaynak: <code>data/knowledge_base/</code> altındaki 12 statik metin dosyası (KDV oranları, format kuralları, 6 ürün kategorisi için tipik fiyat aralığı).</p>`,
  },
  validation: {
    title: "Validation Agent",
    meta: "app/agents/validation_agent.py — ValidationAgent",
    fields: [
      ["Girdi", "state.raw_extraction, state.retrieved_rules"],
      ["Model kullanımı", "Yok — LLM'e hiç sorulmuyor"],
      ["Çıktı", "state.anomalies, state.is_valid, state.validation_checklist"],
    ],
    list: [
      "MathConsistencyCheck — miktar×birim fiyat, KDV tutarı ve toplamların tutarlılığı (0.01 TL tolerans)",
      "VatRateCheck — KDV oranı %1/%10/%20'den biri mi",
      "FormatCheck — fatura/vergi no formatı (strict_tr ya da lenient profili)",
      "PriceRangeCheck — birim fiyat, RAG'ın getirdiği tipik aralıkta mı",
    ],
    extra: `<p>Dört sınıf da <code>Checker</code> arayüzünü uyguluyor, hepsi saf Python — regex ve aritmetik, dış servise bağımlılık yok. <code>validation_checklist</code>, sadece başarısız olanları değil <strong>4 kontrolün tamamının</strong> sonucunu taşıyor.</p>`,
  },
  report: {
    title: "Report Agent",
    meta: "app/agents/report_agent.py — ReportAgent",
    fields: [
      ["Girdi", "raw_extraction, anomalies, validation_checklist, execution_trace"],
      ["Model kullanımı", "Yok — kod-tabanlı özet, LLM'e tekrar sorulmuyor"],
      ["Çıktı", "state.final_report"],
    ],
    list: [
      "risk_score / risk_level — her anomali -15 puan, 100'den başlar (kod: _risk_score)",
      "explanation — anomali listesinden şablon bazlı Türkçe paragraf",
      "validation_checklist — Validation Agent'tan aynen aktarılır",
      "price_history — PostgreSQL'deki geçmiş kayıtlarda aynı ürün için görülen fiyatlar (RAG'dan değil, doğrudan DB sorgusundan)",
      "execution_trace — her ajanın (ve MCP çağrısının) süresi",
    ],
    extra: `<p><code>missing_extraction</code> (Vision hiç veri çıkaramadıysa) özel durumdur: puan otomatik 0, seviye "rejected" — hiçbir şey doğrulanamadığı için sıradan bir kural ihlali gibi sayılmaz.</p>`,
  },
};

function renderAgentDetail(key) {
  const data = AGENT_DETAILS[key];
  const panel = document.getElementById("agent-detail");
  if (!data) { panel.innerHTML = ""; return; }

  let html = `<h3>${data.title}</h3><p class="detail-meta">${data.meta}</p>`;

  if (data.flow) {
    html += '<div class="detail-flow">';
    data.flow.forEach((step, i) => {
      html += `<span class="step">${step}</span>`;
      if (i < data.flow.length - 1) html += '<span class="arrow">→</span>';
    });
    html += "</div>";
  }

  html += '<dl class="detail-fields">';
  data.fields.forEach(([label, value]) => {
    html += `<dt>${label}</dt><dd>${value}</dd>`;
  });
  html += "</dl>";

  if (data.list) {
    html += '<ul class="detail-list">' + data.list.map((item) => `<li>${item}</li>`).join("") + "</ul>";
  }

  html += data.extra || "";

  if (data.sample) {
    html += `<p class="sample-json-label">${data.sample.label}</p>`;
    html += `<pre class="sample-json">${JSON.stringify(data.sample.json, null, 2)}</pre>`;
  }

  panel.innerHTML = html;
}

const agentButtons = document.querySelectorAll(".agent-node");
agentButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const isOpen = btn.getAttribute("aria-expanded") === "true";
    agentButtons.forEach((b) => b.setAttribute("aria-expanded", "false"));
    if (isOpen) {
      document.getElementById("agent-detail").innerHTML = "";
      return;
    }
    btn.setAttribute("aria-expanded", "true");
    renderAgentDetail(btn.dataset.agent);
  });
});
if (agentButtons.length > 0) {
  agentButtons[0].setAttribute("aria-expanded", "true");
  renderAgentDetail(agentButtons[0].dataset.agent);
}

/* ---------------- Doğrulama Motoru: accordion ---------------- */
document.querySelectorAll(".checker-header").forEach((header) => {
  header.addEventListener("click", () => {
    const key = header.dataset.checker;
    const body = document.querySelector(`[data-checker-body="${key}"]`);
    const isOpen = header.getAttribute("aria-expanded") === "true";
    header.setAttribute("aria-expanded", String(!isOpen));
    if (body) body.hidden = isOpen;
  });
});

/* ---------------- MathConsistencyCheck demosu (sadece tarayicida) ----------------
   app/agents/validation_agent.py::MathConsistencyCheck ile AYNI formul/tolerans
   (miktar*birim_fiyat ~= kalem_toplami, 0.01 TL). Hicbir veri sunucuya gitmiyor. */
const MATH_TOLERANCE = 0.01;
function updateMathDemo() {
  const qty = parseFloat(document.getElementById("demoQty").value) || 0;
  const unitPrice = parseFloat(document.getElementById("demoUnitPrice").value) || 0;
  const lineTotal = parseFloat(document.getElementById("demoLineTotal").value) || 0;
  const expected = Math.round(qty * unitPrice * 100) / 100;
  const diff = Math.abs(expected - lineTotal);
  const result = document.getElementById("demoResult");
  if (diff <= MATH_TOLERANCE) {
    result.textContent = `Tutarlı (beklenen ${expected} TL)`;
    result.className = "demo-result ok";
  } else {
    result.textContent = `Tutarsız — beklenen ${expected} TL, girilen ${lineTotal} TL`;
    result.className = "demo-result bad";
  }
}
["demoQty", "demoUnitPrice", "demoLineTotal"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("input", updateMathDemo);
});
if (document.getElementById("demoResult")) updateMathDemo();

/* ---------------- Under the Hood ---------------- */
const ARCH_DETAILS = {
  fastapi: "FastAPI — app/api/main.py. HTTP katmanı; POST /invoices, POST /invoices/stream (SSE), POST /invoices/bulk, GET /invoices, GET /invoices/{id}.",
  langgraph: "LangGraph StateGraph(PipelineState) — app/graph.py. Dört ajanı sabit sırada zincirler (Vision→RAG→Validation→Report), koşullu geçiş/retry yok.",
  vision: "Vision Agent — app/agents/vision_agent.py. Görseli görsel-destekli LLM'e gönderir, JSON çıkarır.",
  rag: "RAG Agent — app/agents/rag_agent.py. MCP client olarak, kural sorgulamasını ayrı bir alt sürece devrediyor.",
  mcp: "MCP Rules Server — app/mcp/rules_server.py. Bağımsız stdio alt süreç, tek aracı query_rules.",
  embedding: "Embedding Provider — app/providers/. Sorgu metnini ve bilgi tabanı belgelerini vektöre çevirir (Ollama nomic-embed-text ya da Gemini embedding).",
  chromadb: "ChromaDB — yerel, diskte persist eden vektör veritabanı (data/chroma_db/). Benzerlik araması burada yapılır.",
  validation: "Validation Agent — app/agents/validation_agent.py. 4 deterministik kural sınıfı, LLM kullanmaz.",
  report: "Report Agent — app/agents/report_agent.py. final_report'u kurar: risk_score, explanation, price_history, execution_trace.",
  postgres: "PostgreSQL — app/db/models.py (InvoiceRecord). Her sonuç, tam final_report'uyla birlikte kalıcı olarak yazılır.",
};

document.querySelectorAll(".arch-node").forEach((btn) => {
  btn.addEventListener("click", () => {
    const isOpen = btn.getAttribute("aria-expanded") === "true";
    document.querySelectorAll(".arch-node").forEach((b) => b.setAttribute("aria-expanded", "false"));
    const panel = document.getElementById("arch-detail");
    if (isOpen) { panel.innerHTML = ""; return; }
    btn.setAttribute("aria-expanded", "true");
    const text = ARCH_DETAILS[btn.dataset.arch];
    panel.innerHTML = text ? `<h4>${btn.textContent}</h4><p>${text}</p>` : "";
  });
});
