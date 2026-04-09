const state = {
  results: [],
  targetId: "",
  comparisonIds: new Set(),
};

const els = {
  apiKeyInput: document.getElementById("apiKeyInput"),
  queryInput: document.getElementById("queryInput"),
  searchBtn: document.getElementById("searchBtn"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusLine: document.getElementById("statusLine"),
  searchResults: document.getElementById("searchResults"),
  analysisPanel: document.getElementById("analysisPanel"),
  scoreCards: document.getElementById("scoreCards"),
  overallStatement: document.getElementById("overallStatement"),
  metricStatements: document.getElementById("metricStatements"),
  legalReasons: document.getElementById("legalReasons"),
  actionFocus: document.getElementById("actionFocus"),
  conflictTableWrap: document.getElementById("conflictTableWrap"),
};

function setStatus(message, isError = false) {
  els.statusLine.textContent = message;
  els.statusLine.style.color = isError ? "#b4381a" : "";
}

function persistApiKey(value) {
  localStorage.setItem("law_issue_engine_oc", value || "");
}

function loadApiKey() {
  const saved = localStorage.getItem("law_issue_engine_oc") || "";
  els.apiKeyInput.value = saved;
}

function getApiKey() {
  return (els.apiKeyInput.value || "").trim();
}

function getSelectedTarget() {
  return state.results.find((item) => item.id === state.targetId) || null;
}

function getSelectedComparisons() {
  return state.results.filter((item) => state.comparisonIds.has(item.id));
}

function updateAnalyzeButton() {
  const hasTarget = !!state.targetId;
  const comparisonCount = state.comparisonIds.size;
  els.analyzeBtn.disabled = !hasTarget || comparisonCount < 1;
}

function renderSearchResults() {
  els.searchResults.innerHTML = "";

  if (!state.results.length) {
    const empty = document.createElement("p");
    empty.textContent = "검색 결과가 없습니다.";
    empty.className = "hint";
    els.searchResults.appendChild(empty);
    updateAnalyzeButton();
    return;
  }

  state.results.forEach((law) => {
    const item = document.createElement("article");
    item.className = "result-item";

    const rowTop = document.createElement("div");
    rowTop.className = "row-top";

    const targetLabel = document.createElement("label");
    const targetRadio = document.createElement("input");
    targetRadio.type = "radio";
    targetRadio.name = "targetLaw";
    targetRadio.checked = state.targetId === law.id;
    targetRadio.addEventListener("change", () => {
      state.targetId = law.id;
      if (state.comparisonIds.has(law.id)) {
        state.comparisonIds.delete(law.id);
      }
      renderSearchResults();
    });
    targetLabel.appendChild(targetRadio);
    targetLabel.append("대상");

    const compareLabel = document.createElement("label");
    const compareCheckbox = document.createElement("input");
    compareCheckbox.type = "checkbox";
    compareCheckbox.checked = state.comparisonIds.has(law.id);
    compareCheckbox.disabled = state.targetId === law.id;
    compareCheckbox.addEventListener("change", () => {
      if (compareCheckbox.checked) {
        if (state.targetId === law.id) {
          state.targetId = "";
        }
        state.comparisonIds.add(law.id);
      } else {
        state.comparisonIds.delete(law.id);
      }
      renderSearchResults();
    });
    compareLabel.appendChild(compareCheckbox);
    compareLabel.append("비교");

    rowTop.appendChild(targetLabel);
    rowTop.appendChild(compareLabel);

    const title = document.createElement("p");
    title.className = "law-name";
    title.textContent = law.name;

    const id = document.createElement("p");
    id.className = "law-id";
    id.textContent = `ID: ${law.id}`;

    item.appendChild(rowTop);
    item.appendChild(title);
    item.appendChild(id);
    els.searchResults.appendChild(item);
  });

  updateAnalyzeButton();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `요청 실패(status=${response.status})`);
  }
  return payload;
}

async function handleSearch() {
  const query = (els.queryInput.value || "").trim();
  const apiKey = getApiKey();

  persistApiKey(apiKey);

  if (!query) {
    setStatus("검색어를 입력하세요.", true);
    return;
  }
  if (!apiKey) {
    setStatus("Open API 키(OC)를 입력하세요.", true);
    return;
  }

  setStatus("법령을 검색 중입니다...");
  state.results = [];
  state.targetId = "";
  state.comparisonIds.clear();
  renderSearchResults();

  try {
    const params = new URLSearchParams({
      q: query,
      limit: "30",
      oc: apiKey,
    });
    const data = await fetchJson(`/api/search-laws?${params.toString()}`);

    state.results = data.items || [];
    setStatus(`검색 완료: ${data.count || state.results.length}건`);
    renderSearchResults();
  } catch (error) {
    setStatus(`검색 실패: ${error.message}`, true);
  }
}

function createScoreCard(label, value) {
  const card = document.createElement("article");
  card.className = "score-card";

  const labelEl = document.createElement("p");
  labelEl.className = "score-label";
  labelEl.textContent = label;

  const valueEl = document.createElement("p");
  valueEl.className = "score-value";
  valueEl.textContent = `${value}`;

  card.appendChild(labelEl);
  card.appendChild(valueEl);
  return card;
}

function fillList(ulElement, values) {
  ulElement.innerHTML = "";
  (values || []).forEach((value) => {
    const li = document.createElement("li");
    li.textContent = value;
    ulElement.appendChild(li);
  });
}

function renderConflictTable(conflicts) {
  if (!conflicts || !conflicts.length) {
    els.conflictTableWrap.textContent = "대표 충돌 근거가 없습니다.";
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  ["신규안 조문", "기존법 조문", "유사도", "규범 비교"].forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headRow.appendChild(th);
  });

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  conflicts.slice(0, 7).forEach((item) => {
    const row = document.createElement("tr");

    const proposed = document.createElement("td");
    proposed.textContent = item.proposed_article;

    const existing = document.createElement("td");
    existing.textContent = item.existing_article;

    const similarity = document.createElement("td");
    similarity.textContent = `${item.similarity}%`;

    const mode = document.createElement("td");
    const pm = (item.proposed_modes || []).join("/") || "-";
    const em = (item.existing_modes || []).join("/") || "-";
    mode.textContent = `${pm} ↔ ${em}`;

    row.appendChild(proposed);
    row.appendChild(existing);
    row.appendChild(similarity);
    row.appendChild(mode);
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  els.conflictTableWrap.innerHTML = "";
  els.conflictTableWrap.appendChild(table);
}

function renderAnalysis(report, warnings = []) {
  const { scores, evidence, assessment } = report;

  els.analysisPanel.hidden = false;
  els.scoreCards.innerHTML = "";
  els.scoreCards.appendChild(createScoreCard("중복 점수", scores.overlap_score));
  els.scoreCards.appendChild(createScoreCard("충돌 점수", scores.conflict_score));
  els.scoreCards.appendChild(createScoreCard("헌법 위험 점수", scores.constitutional_risk_score));
  els.scoreCards.appendChild(createScoreCard("총 위험 점수", scores.total_risk_score));

  els.overallStatement.textContent = `${assessment.overall_statement} (${assessment.disclaimer})`;

  fillList(els.metricStatements, [
    `중복: ${assessment.metric_statements.overlap}`,
    `충돌: ${assessment.metric_statements.conflict}`,
    `헌법위험: ${assessment.metric_statements.constitutional}`,
  ]);

  fillList(els.legalReasons, assessment.legal_grounded_reasons || []);
  fillList(els.actionFocus, assessment.recommended_revision_focus || []);
  renderConflictTable(evidence.conflict_top_matches || []);

  if (warnings.length) {
    setStatus(`분석 완료(일부 경고 ${warnings.length}건): ${warnings.join(" | ")}`);
  } else {
    setStatus("분석 완료");
  }

  els.analysisPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handleAnalyze() {
  const apiKey = getApiKey();
  persistApiKey(apiKey);

  if (!apiKey) {
    setStatus("Open API 키(OC)를 입력하세요.", true);
    return;
  }

  const target = getSelectedTarget();
  const comparisons = getSelectedComparisons();

  if (!target || !comparisons.length) {
    setStatus("대상 1개와 비교 1개 이상을 선택하세요.", true);
    return;
  }

  setStatus("선택한 법령 본문을 불러와 분석 중입니다...");
  els.analyzeBtn.disabled = true;

  try {
    const payload = await fetchJson("/api/analyze-selected", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        oc: apiKey,
        target: {
          id: target.id,
          name: target.name,
        },
        comparisons: comparisons.map((law) => ({ id: law.id, name: law.name })),
      }),
    });

    renderAnalysis(payload.report, payload.warnings || []);
  } catch (error) {
    setStatus(`분석 실패: ${error.message}`, true);
  } finally {
    updateAnalyzeButton();
  }
}

els.searchBtn.addEventListener("click", handleSearch);
els.queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleSearch();
  }
});
els.analyzeBtn.addEventListener("click", handleAnalyze);
els.apiKeyInput.addEventListener("change", () => {
  persistApiKey(getApiKey());
});

loadApiKey();
updateAnalyzeButton();
