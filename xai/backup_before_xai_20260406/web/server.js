const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");
const { URL } = require("node:url");

const { analyzeLaws } = require("./riskEngine");
const { searchLaws, fetchLawTextById } = require("./lawApi");

const HOST = "0.0.0.0";
const PORT = Number(process.env.PORT || 5050);
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

function sendText(res, statusCode, text) {
  res.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(text);
}

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
    const size = chunks.reduce((acc, cur) => acc + cur.length, 0);
    if (size > 1_000_000) {
      throw new Error("요청 본문이 너무 큽니다.");
    }
  }
  return Buffer.concat(chunks).toString("utf-8");
}

async function serveStatic(req, res, pathname) {
  const safePath = pathname === "/" ? "/index.html" : pathname;
  const resolved = path.join(PUBLIC_DIR, path.normalize(safePath));
  if (!resolved.startsWith(PUBLIC_DIR)) {
    sendText(res, 403, "Forbidden");
    return;
  }

  try {
    const data = await fs.readFile(resolved);
    const ext = path.extname(resolved).toLowerCase();
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(data);
  } catch (_err) {
    sendText(res, 404, "Not Found");
  }
}

function normalizeSelection(raw) {
  if (!raw || typeof raw !== "object") return null;
  const id = String(raw.id || "").trim();
  const name = String(raw.name || id || "").trim();
  if (!id) return null;
  return { id, name };
}

async function handleSearch(reqUrl, res) {
  const query = (reqUrl.searchParams.get("q") || "").trim();
  const oc = (reqUrl.searchParams.get("oc") || "").trim();
  const limit = Number(reqUrl.searchParams.get("limit") || 20);

  if (!query) {
    sendJson(res, 400, { error: "검색어(q)가 필요합니다." });
    return;
  }

  try {
    const items = await searchLaws(query, { oc, limit });
    sendJson(res, 200, {
      query,
      count: items.length,
      items,
    });
  } catch (error) {
    sendJson(res, 500, {
      error: error.message,
    });
  }
}

async function handleAnalyze(req, res) {
  let body;
  try {
    const raw = await readRequestBody(req);
    body = raw ? JSON.parse(raw) : {};
  } catch (error) {
    sendJson(res, 400, { error: `요청 본문 파싱 실패: ${error.message}` });
    return;
  }

  const oc = String(body.oc || "").trim();
  const target = normalizeSelection(body.target);
  const comparisons = Array.isArray(body.comparisons) ? body.comparisons.map(normalizeSelection).filter(Boolean) : [];

  if (!target) {
    sendJson(res, 400, { error: "분석 대상 법령(target.id)이 필요합니다." });
    return;
  }
  if (!comparisons.length) {
    sendJson(res, 400, { error: "비교 법령(comparisons)이 최소 1개 필요합니다." });
    return;
  }

  try {
    const targetLaw = await fetchLawTextById(target.id, { oc });
    targetLaw.name = target.name || targetLaw.name;

    const comparisonFetches = await Promise.allSettled(
      comparisons.map((law) => fetchLawTextById(law.id, { oc })),
    );

    const comparisonLaws = [];
    const warnings = [];

    comparisonFetches.forEach((result, index) => {
      const selected = comparisons[index];
      if (result.status === "fulfilled") {
        const law = result.value;
        law.name = selected.name || law.name;
        comparisonLaws.push(law);
      } else {
        warnings.push(`${selected.name || selected.id}: ${result.reason.message}`);
      }
    });

    if (!comparisonLaws.length) {
      sendJson(res, 500, {
        error: "비교 법령 본문을 하나도 가져오지 못했습니다.",
        warnings,
      });
      return;
    }

    const report = analyzeLaws(targetLaw, comparisonLaws);
    sendJson(res, 200, {
      report,
      warnings,
      fetched_laws: {
        target: { id: targetLaw.id, name: targetLaw.name },
        comparisons: comparisonLaws.map((law) => ({ id: law.id, name: law.name })),
      },
    });
  } catch (error) {
    sendJson(res, 500, { error: error.message });
  }
}

const server = http.createServer(async (req, res) => {
  const reqUrl = new URL(req.url || "/", `http://${req.headers.host}`);
  const { pathname } = reqUrl;

  if (req.method === "GET" && pathname === "/api/health") {
    sendJson(res, 200, { ok: true, service: "law-issue-engine" });
    return;
  }

  if (req.method === "GET" && pathname === "/api/search-laws") {
    await handleSearch(reqUrl, res);
    return;
  }

  if (req.method === "POST" && pathname === "/api/analyze-selected") {
    await handleAnalyze(req, res);
    return;
  }

  if (req.method === "GET") {
    await serveStatic(req, res, pathname);
    return;
  }

  sendText(res, 405, "Method Not Allowed");
});

server.listen(PORT, HOST, () => {
  process.stdout.write(`Law issue engine is running on http://localhost:${PORT}\n`);
});
