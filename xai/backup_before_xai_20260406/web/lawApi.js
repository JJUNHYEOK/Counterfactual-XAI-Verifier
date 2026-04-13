const LAW_SEARCH_ENDPOINT = "https://www.law.go.kr/DRF/lawSearch.do";
const LAW_SERVICE_ENDPOINT = "https://www.law.go.kr/DRF/lawService.do";

const NAME_KEYS = ["법령명한글", "법령명", "법령약칭명", "법령명영문", "lawName", "name"];
const ID_KEYS = ["법령ID", "MST", "ID", "lawId", "id"];

function decodeXmlEntities(input) {
  return String(input || "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function stripTags(input) {
  return decodeXmlEntities(String(input || ""))
    .replace(/<!\[CDATA\[|\]\]>/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function pickFirstValue(obj, keys) {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, key) && obj[key] !== null && obj[key] !== undefined && obj[key] !== "") {
      return obj[key];
    }
  }
  return null;
}

function normalizeLawEntry(raw) {
  if (!raw || typeof raw !== "object") return null;
  const idValue = pickFirstValue(raw, ID_KEYS);
  const nameValue = pickFirstValue(raw, NAME_KEYS);

  if (!idValue && !nameValue) return null;

  const id = String(idValue || nameValue).trim();
  const name = String(nameValue || idValue).trim();
  if (!id || !name) return null;

  return {
    id,
    name,
    display: `${name} (${id})`,
  };
}

function walkAndCollectLawEntries(node, bag) {
  if (!node) return;

  if (Array.isArray(node)) {
    node.forEach((item) => walkAndCollectLawEntries(item, bag));
    return;
  }

  if (typeof node === "object") {
    const normalized = normalizeLawEntry(node);
    if (normalized) bag.push(normalized);
    Object.values(node).forEach((value) => walkAndCollectLawEntries(value, bag));
  }
}

function uniqueById(entries) {
  const seen = new Set();
  return entries.filter((entry) => {
    if (seen.has(entry.id)) return false;
    seen.add(entry.id);
    return true;
  });
}

function parseSearchXml(xml) {
  const entries = [];
  const blocks = [...xml.matchAll(/<(?:law|법령)>([\s\S]*?)<\/(?:law|법령)>/g)].map((m) => m[1]);

  if (blocks.length) {
    blocks.forEach((block) => {
      const id = stripTags((block.match(/<(?:법령ID|MST|ID|lawId)>([\s\S]*?)<\/(?:법령ID|MST|ID|lawId)>/) || [])[1]);
      const name = stripTags((block.match(/<(?:법령명한글|법령명|법령약칭명|lawName)>([\s\S]*?)<\/(?:법령명한글|법령명|법령약칭명|lawName)>/) || [])[1]);
      if (id && name) entries.push({ id, name, display: `${name} (${id})` });
    });
    return uniqueById(entries);
  }

  const ids = [...xml.matchAll(/<(?:법령ID|MST|ID|lawId)>([\s\S]*?)<\/(?:법령ID|MST|ID|lawId)>/g)].map((m) => stripTags(m[1]));
  const names = [...xml.matchAll(/<(?:법령명한글|법령명|법령약칭명|lawName)>([\s\S]*?)<\/(?:법령명한글|법령명|법령약칭명|lawName)>/g)].map((m) => stripTags(m[1]));

  for (let i = 0; i < Math.min(ids.length, names.length); i += 1) {
    if (ids[i] && names[i]) entries.push({ id: ids[i], name: names[i], display: `${names[i]} (${ids[i]})` });
  }

  return uniqueById(entries);
}

function parseApiErrorMessage(rawText) {
  const text = String(rawText || "");

  try {
    const parsed = JSON.parse(text);
    const candidates = [
      parsed?.error,
      parsed?.message,
      parsed?.msg,
      parsed?.resultMsg,
      parsed?.LawSearch?.message,
    ].filter(Boolean);
    if (candidates.length) return String(candidates[0]);

    const stringified = JSON.stringify(parsed);
    if (/인증|권한|OC|오류|error|invalid|forbidden|접근/i.test(stringified)) {
      return "API 권한 또는 인증값(OC) 관련 오류 응답이 반환되었습니다.";
    }
  } catch (_err) {
    // JSON parse 실패 시 XML/텍스트 검사
  }

  const xmlMsg = text.match(/<(?:message|msg|오류메시지|resultMsg)>([\s\S]*?)<\/(?:message|msg|오류메시지|resultMsg)>/i);
  if (xmlMsg && xmlMsg[1]) {
    return stripTags(xmlMsg[1]);
  }

  if (/인증|권한|OC|오류|error|invalid|forbidden|접근/i.test(text)) {
    return stripTags(text).slice(0, 180) || "API 권한 또는 인증값(OC) 관련 오류 응답이 반환되었습니다.";
  }

  return "";
}

async function fetchWithTimeout(url, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "Law-Issue-Engine/1.0" },
    });
    const text = await response.text();
    return { ok: response.ok, status: response.status, text };
  } finally {
    clearTimeout(timeoutId);
  }
}

function ensureApiKey(oc) {
  const key = String(oc || process.env.LAW_API_KEY || "").trim();
  if (!key) {
    const err = new Error("국가법령정보 Open API 키(OC)가 필요합니다.");
    err.code = "MISSING_API_KEY";
    throw err;
  }
  return key;
}

async function searchLaws(query, options = {}) {
  const q = String(query || "").trim();
  if (!q) return [];

  const apiKey = ensureApiKey(options.oc);
  const limit = Math.min(Number(options.limit || 20), 50);

  const jsonUrl = new URL(LAW_SEARCH_ENDPOINT);
  jsonUrl.searchParams.set("OC", apiKey);
  jsonUrl.searchParams.set("target", "law");
  jsonUrl.searchParams.set("type", "JSON");
  jsonUrl.searchParams.set("query", q);
  jsonUrl.searchParams.set("display", String(limit));

  const jsonRes = await fetchWithTimeout(jsonUrl);
  if (!jsonRes.ok) throw new Error(`법령 검색 요청 실패(status=${jsonRes.status})`);

  const jsonError = parseApiErrorMessage(jsonRes.text);
  if (jsonError) throw new Error(`법령 검색 API 오류: ${jsonError}`);

  let entries = [];
  try {
    const payload = JSON.parse(jsonRes.text);
    const bag = [];
    walkAndCollectLawEntries(payload, bag);
    entries = uniqueById(bag);
  } catch (_err) {
    entries = [];
  }

  if (entries.length) return entries.slice(0, limit);

  const xmlUrl = new URL(LAW_SEARCH_ENDPOINT);
  xmlUrl.searchParams.set("OC", apiKey);
  xmlUrl.searchParams.set("target", "law");
  xmlUrl.searchParams.set("type", "XML");
  xmlUrl.searchParams.set("query", q);
  xmlUrl.searchParams.set("display", String(limit));

  const xmlRes = await fetchWithTimeout(xmlUrl);
  if (!xmlRes.ok) throw new Error(`법령 검색(XML) 요청 실패(status=${xmlRes.status})`);

  const xmlError = parseApiErrorMessage(xmlRes.text);
  if (xmlError) throw new Error(`법령 검색 API 오류(XML): ${xmlError}`);

  const xmlEntries = parseSearchXml(xmlRes.text).slice(0, limit);
  if (xmlEntries.length) return xmlEntries;

  throw new Error("검색 결과를 파싱하지 못했습니다. API 신청 항목(대한민국 현행법령 목록/본문 JSON·XML)과 OC 키 권한을 확인하세요.");
}

function collectTextFieldsFromObject(node, bag) {
  if (!node) return;
  if (Array.isArray(node)) {
    node.forEach((item) => collectTextFieldsFromObject(item, bag));
    return;
  }
  if (typeof node === "object") {
    Object.entries(node).forEach(([key, value]) => {
      if (value === null || value === undefined) return;
      const normalizedKey = String(key);
      const isTextField = ["조문내용", "조문", "본문", "내용", "article"].some((fragment) => normalizedKey.includes(fragment));
      if (isTextField && typeof value === "string") {
        const cleaned = stripTags(value);
        if (cleaned.length >= 8) bag.push(cleaned);
      }
      collectTextFieldsFromObject(value, bag);
    });
  }
}

function parseLawXml(xml, fallbackName = "법령") {
  const titleMatch = xml.match(/<(?:법령명한글|법령명|lawName)>([\s\S]*?)<\/(?:법령명한글|법령명|lawName)>/);
  const name = stripTags(titleMatch ? titleMatch[1] : fallbackName) || fallbackName;

  const articleTexts = [...xml.matchAll(/<조문내용>([\s\S]*?)<\/조문내용>/g)]
    .map((match) => stripTags(match[1]))
    .filter((chunk) => chunk.length >= 8);

  let text = articleTexts.join("\n");
  if (!text) {
    text = [...xml.matchAll(/<(?:조문|본문|내용)>([\s\S]*?)<\/(?:조문|본문|내용)>/g)]
      .map((match) => stripTags(match[1]))
      .filter((chunk) => chunk.length >= 8)
      .join("\n");
  }
  return { name, text };
}

async function fetchLawTextById(lawId, options = {}) {
  const id = String(lawId || "").trim();
  if (!id) throw new Error("법령 ID가 필요합니다.");

  const apiKey = ensureApiKey(options.oc);

  const jsonUrl = new URL(LAW_SERVICE_ENDPOINT);
  jsonUrl.searchParams.set("OC", apiKey);
  jsonUrl.searchParams.set("target", "law");
  jsonUrl.searchParams.set("type", "JSON");
  jsonUrl.searchParams.set("ID", id);

  const jsonRes = await fetchWithTimeout(jsonUrl);
  if (!jsonRes.ok) throw new Error(`법령 본문 요청 실패(status=${jsonRes.status}, id=${id})`);

  const jsonErr = parseApiErrorMessage(jsonRes.text);
  if (jsonErr) throw new Error(`법령 본문 API 오류(id=${id}): ${jsonErr}`);

  try {
    const payload = JSON.parse(jsonRes.text);
    const titleCandidateBag = [];
    walkAndCollectLawEntries(payload, titleCandidateBag);
    const title = titleCandidateBag.length ? titleCandidateBag[0].name : id;

    const textBag = [];
    collectTextFieldsFromObject(payload, textBag);
    const text = textBag.join("\n").trim();
    if (text.length >= 30) return { id, name: title || id, text };
  } catch (_err) {
    // XML fallback
  }

  const xmlUrl = new URL(LAW_SERVICE_ENDPOINT);
  xmlUrl.searchParams.set("OC", apiKey);
  xmlUrl.searchParams.set("target", "law");
  xmlUrl.searchParams.set("type", "XML");
  xmlUrl.searchParams.set("ID", id);

  const xmlRes = await fetchWithTimeout(xmlUrl);
  if (!xmlRes.ok) throw new Error(`법령 본문(XML) 요청 실패(status=${xmlRes.status}, id=${id})`);

  const xmlErr = parseApiErrorMessage(xmlRes.text);
  if (xmlErr) throw new Error(`법령 본문 API 오류(XML, id=${id}): ${xmlErr}`);

  const parsed = parseLawXml(xmlRes.text, id);
  if (!parsed.text || parsed.text.length < 30) throw new Error(`법령 본문 파싱 실패(id=${id})`);

  return { id, name: parsed.name, text: parsed.text };
}

module.exports = {
  searchLaws,
  fetchLawTextById,
};
