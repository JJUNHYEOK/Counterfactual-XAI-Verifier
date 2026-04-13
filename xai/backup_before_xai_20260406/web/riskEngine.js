const ARTICLE_REGEX = /(제\s*\d+\s*조(?:의\s*\d+)?)/g;
const TOKEN_REGEX = /[가-힣A-Za-z0-9]{2,}/g;

const NORMATIVE_MARKERS = {
  prohibit: ["금지", "아니된다", "할 수 없다", "불가", "제한한다"],
  require: ["하여야 한다", "해야 한다", "의무", "신고하여야", "제출하여야"],
  allow: ["할 수 있다", "허용", "가능", "인정"],
  exempt: ["면제", "예외", "적용하지 아니한다", "적용하지 않는다"],
};

const MODE_LABELS = {
  prohibit: "금지",
  require: "의무",
  allow: "허용",
  exempt: "면제",
};

const TOPIC_KEYWORDS = {
  privacy: ["개인정보", "사생활", "통신", "위치정보", "감시", "추적"],
  property: ["재산", "수용", "보상", "부담금", "과징금", "세금"],
  expression: ["표현", "언론", "출판", "집회", "시위"],
  employment: ["근로자", "노동", "임금", "해고", "사업주"],
  criminal: ["처벌", "징역", "벌금", "형사", "구금"],
  administrative: ["행정", "허가", "인가", "등록", "신고"],
};

const VAGUE_TERMS = ["적절한", "필요한", "상당한", "공익상", "신속히", "원활히", "충분한", "현저한", "긴급한"];
const BROAD_SCOPE_TERMS = ["모든", "전부", "일체", "누구든지", "항상", "즉시"];
const DELEGATION_TERMS = ["대통령령으로 정한다", "총리령으로 정한다", "부령으로 정한다", "정하는 바에 따라"];
const RETROACTIVE_TERMS = ["소급", "이미 발생한", "종전의", "과거의 행위"];
const EQUALITY_RISK_TERMS = ["특정 집단", "일부 국민", "정치적 견해", "성별", "출신지역"];

const PRINCIPLE_META = {
  clarity_principle: {
    label: "명확성 원칙",
    basis: "헌법상 법치국가원리·적법절차 원칙",
    riskReason: "추상적 문구가 많으면 규율 대상과 요건의 예측가능성이 떨어집니다.",
  },
  proportionality_principle: {
    label: "비례성 원칙",
    basis: "헌법 제37조 제2항",
    riskReason: "적용 범위와 제한 강도가 넓으면 최소침해 심사에서 취약해집니다.",
  },
  non_delegation_principle: {
    label: "포괄위임금지 원칙",
    basis: "헌법 제75조",
    riskReason: "본질적 사항을 하위법령에 넘기는 구조로 해석될 수 있습니다.",
  },
  non_retroactivity_principle: {
    label: "소급입법 제한 원칙",
    basis: "헌법 제13조 제2항",
    riskReason: "이미 종료된 사실관계에 불이익을 부과하면 신뢰보호 문제가 큽니다.",
  },
  equality_principle: {
    label: "평등 원칙",
    basis: "헌법 제11조 제1항",
    riskReason: "집단 구분 기준이 불명확하면 자의적 차별 문제로 이어질 수 있습니다.",
  },
};

const OPPOSITION = new Set(["prohibit|allow", "allow|prohibit", "require|exempt", "exempt|require"]);

const STOPWORDS = new Set([
  "제",
  "조",
  "및",
  "또는",
  "그",
  "이",
  "수",
  "것",
  "등",
  "한다",
  "있다",
  "아니한다",
  "의",
  "에",
  "를",
  "으로",
]);

function normalizeSpace(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function splitArticles(lawName, text) {
  const clean = normalizeSpace(text);
  if (!clean) return [];

  const matches = [...clean.matchAll(ARTICLE_REGEX)];
  if (!matches.length) {
    return [
      {
        lawName,
        articleId: "본문",
        text: clean,
      },
    ];
  }

  const articles = [];
  for (let i = 0; i < matches.length; i += 1) {
    const start = matches[i].index;
    const end = i + 1 < matches.length ? matches[i + 1].index : clean.length;
    const articleText = clean.slice(start, end).trim();
    const articleId = normalizeSpace(matches[i][1]);
    articles.push({
      lawName,
      articleId,
      text: articleText,
    });
  }
  return articles;
}

function tokenize(text) {
  const matches = String(text || "").match(TOKEN_REGEX) || [];
  return new Set(
    matches
      .map((token) => token.toLowerCase())
      .filter((token) => !STOPWORDS.has(token)),
  );
}

function detectModalities(text) {
  const source = String(text || "");
  const modes = new Set();
  Object.entries(NORMATIVE_MARKERS).forEach(([mode, markers]) => {
    if (markers.some((marker) => source.includes(marker))) {
      modes.add(mode);
    }
  });
  return modes;
}

function detectTopics(text) {
  const source = String(text || "");
  const tags = new Set();
  Object.entries(TOPIC_KEYWORDS).forEach(([topic, words]) => {
    if (words.some((word) => source.includes(word))) {
      tags.add(topic);
    }
  });
  return tags;
}

function enrichArticles(articles) {
  return articles.map((article) => ({
    ...article,
    tokens: tokenize(article.text),
    modalities: detectModalities(article.text),
    topicTags: detectTopics(article.text),
  }));
}

function buildLawDocument({ id = "", name, text }) {
  const lawName = name || id || "법령";
  const articles = enrichArticles(splitArticles(lawName, text));
  return {
    id,
    name: lawName,
    text: String(text || ""),
    articles,
  };
}

function jaccard(setA, setB) {
  if (!setA.size || !setB.size) return 0;
  let intersection = 0;
  setA.forEach((value) => {
    if (setB.has(value)) intersection += 1;
  });
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function articleSimilarity(articleA, articleB) {
  const tokenSim = jaccard(articleA.tokens, articleB.tokens);
  const topicSim = articleA.topicTags.size || articleB.topicTags.size ? jaccard(articleA.topicTags, articleB.topicTags) : 0;
  return 0.75 * tokenSim + 0.25 * topicSim;
}

function computeOverlapScore(proposed, existing) {
  const existingArticles = existing.flatMap((law) => law.articles);
  if (!proposed.articles.length || !existingArticles.length) {
    return { score: 0, evidence: [] };
  }

  const pairs = [];
  const bestScores = [];

  proposed.articles.forEach((pArticle) => {
    let bestScore = 0;
    let bestMatch = null;

    existingArticles.forEach((eArticle) => {
      const sim = articleSimilarity(pArticle, eArticle);
      if (sim > bestScore) {
        bestScore = sim;
        bestMatch = eArticle;
      }
    });

    bestScores.push(bestScore);
    if (bestMatch) {
      pairs.push([bestScore, pArticle, bestMatch]);
    }
  });

  const overlapScore = (100 * bestScores.reduce((sum, score) => sum + score, 0)) / Math.max(bestScores.length, 1);
  const evidence = pairs
    .sort((a, b) => b[0] - a[0])
    .slice(0, 5)
    .map(([score, proposedArticle, existingArticle]) => ({
      proposed_article: `${proposedArticle.lawName} ${proposedArticle.articleId}`,
      existing_article: `${existingArticle.lawName} ${existingArticle.articleId}`,
      similarity: Number((score * 100).toFixed(2)),
    }));

  return { score: Number(overlapScore.toFixed(2)), evidence };
}

function hasNormativeConflict(modesA, modesB) {
  for (const modeA of modesA) {
    for (const modeB of modesB) {
      if (OPPOSITION.has(`${modeA}|${modeB}`)) {
        return true;
      }
    }
  }
  return false;
}

function computeConflictScore(proposed, existing) {
  const existingArticles = existing.flatMap((law) => law.articles);
  if (!proposed.articles.length || !existingArticles.length) {
    return { score: 0, evidence: [] };
  }

  let weightedConflict = 0;
  let weightedTotal = 0;
  const evidence = [];

  proposed.articles.forEach((pArticle) => {
    existingArticles.forEach((eArticle) => {
      const sim = articleSimilarity(pArticle, eArticle);
      if (sim < 0.18) return;

      weightedTotal += sim;
      const conflict = hasNormativeConflict(pArticle.modalities, eArticle.modalities);
      if (!conflict) return;

      weightedConflict += sim;
      evidence.push({
        proposed_article: `${pArticle.lawName} ${pArticle.articleId}`,
        existing_article: `${eArticle.lawName} ${eArticle.articleId}`,
        similarity: Number((sim * 100).toFixed(2)),
        proposed_modes: [...pArticle.modalities].sort(),
        existing_modes: [...eArticle.modalities].sort(),
      });
    });
  });

  if (weightedTotal === 0) {
    return { score: 0, evidence: [] };
  }

  const conflictScore = (100 * weightedConflict) / weightedTotal;
  return {
    score: Number(conflictScore.toFixed(2)),
    evidence: evidence.sort((a, b) => b.similarity - a.similarity).slice(0, 7),
  };
}

function countTerms(text, terms) {
  return terms.reduce((acc, term) => acc + (text.includes(term) ? 1 : 0), 0);
}

function clamp(value, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function calcConstitutionalRisk(proposed) {
  const text = normalizeSpace(proposed.text);
  const textLenFactor = Math.max(1, Math.log10(Math.max(text.length, 10)));

  const vagueCount = countTerms(text, VAGUE_TERMS);
  const broadCount = countTerms(text, BROAD_SCOPE_TERMS);
  const delegationCount = countTerms(text, DELEGATION_TERMS);
  const retroCount = countTerms(text, RETROACTIVE_TERMS);
  const equalityCount = countTerms(text, EQUALITY_RISK_TERMS);
  const criminalCount = countTerms(text, TOPIC_KEYWORDS.criminal);
  const rightsCount = countTerms(text, [...TOPIC_KEYWORDS.privacy, ...TOPIC_KEYWORDS.expression, ...TOPIC_KEYWORDS.property]);

  const clarityRisk = clamp((vagueCount / textLenFactor) * 11);
  const proportionalityRisk = clamp((broadCount * 8 + criminalCount * 4 + rightsCount * 3) / textLenFactor);
  const delegationRisk = clamp((delegationCount * 18 + vagueCount * 3) / textLenFactor);
  const retroactivityRisk = clamp((retroCount * 26) / textLenFactor);
  const equalityRisk = clamp((equalityCount * 22 + broadCount * 2) / textLenFactor);

  const breakdown = {
    clarity_principle: Number(clarityRisk.toFixed(2)),
    proportionality_principle: Number(proportionalityRisk.toFixed(2)),
    non_delegation_principle: Number(delegationRisk.toFixed(2)),
    non_retroactivity_principle: Number(retroactivityRisk.toFixed(2)),
    equality_principle: Number(equalityRisk.toFixed(2)),
  };

  const overall = Number(
    (
      0.22 * clarityRisk
      + 0.3 * proportionalityRisk
      + 0.2 * delegationRisk
      + 0.16 * retroactivityRisk
      + 0.12 * equalityRisk
    ).toFixed(2),
  );

  const keywords = {
    clarity_principle: VAGUE_TERMS.filter((term) => text.includes(term)).slice(0, 8),
    proportionality_principle: BROAD_SCOPE_TERMS.filter((term) => text.includes(term)).slice(0, 8),
    non_delegation_principle: DELEGATION_TERMS.filter((term) => text.includes(term)).slice(0, 8),
    non_retroactivity_principle: RETROACTIVE_TERMS.filter((term) => text.includes(term)).slice(0, 8),
    equality_principle: EQUALITY_RISK_TERMS.filter((term) => text.includes(term)).slice(0, 8),
  };

  return { score: overall, breakdown, keywords };
}

function modalitiesToText(modes) {
  if (!modes || !modes.length) return "규범표현 미검출";
  return [...modes].sort().map((mode) => MODE_LABELS[mode] || mode).join("/");
}

function joinKeywords(keywords) {
  if (!keywords || !keywords.length) return "관련 키워드 미검출";
  return keywords.join(", ");
}

function buildOverlapStatement(overlap, overlapEvidence) {
  if (!overlapEvidence.length) {
    return "입력된 비교 범위에서 직접적인 중복 근거가 적어, 현재로서는 중복 위험이 상대적으로 괜찮은 편입니다.";
  }

  const top = overlapEvidence[0];
  const pair = `${top.proposed_article} ↔ ${top.existing_article}(유사도 ${top.similarity}%)`;

  if (overlap >= 70) {
    return `기존 법령과 핵심 규율 영역이 크게 겹칩니다. ${pair} 근거가 확인되어 조문 목적과 적용 범위를 재설계하는 보완이 필요합니다.`;
  }
  if (overlap >= 40) {
    return `일부 핵심 조문이 기존 법과 중첩됩니다. ${pair} 근거가 있어 중복을 줄이는 문구 정리가 필요합니다.`;
  }
  return `중복 신호는 제한적이라 비교적 괜찮습니다. 가장 높은 중첩 근거는 ${pair} 수준입니다.`;
}

function buildConflictStatement(conflict, conflictEvidence) {
  if (!conflictEvidence.length) {
    if (conflict < 25) {
      return "동일 행위에 대한 정면 충돌 근거가 뚜렷하지 않아 현재 범위에서는 규범 정합성이 비교적 괜찮습니다.";
    }
    return "충돌 점수는 일부 있으나 명시적 반대 규범 근거가 제한적입니다. 유사 조문 추가 검토가 필요합니다.";
  }

  const top = conflictEvidence[0];
  const pair = `${top.proposed_article}(${modalitiesToText(top.proposed_modes)})와 ${top.existing_article}(${modalitiesToText(top.existing_modes)})`;

  if (conflict >= 70) {
    return `유사 행위를 두고 반대 규범이 병존합니다. ${pair}가 대표 근거라 입법 충돌 방지를 위한 조문 재정렬이 필요합니다.`;
  }
  if (conflict >= 40) {
    return `규범 충돌 신호가 확인됩니다. ${pair} 근거가 있어 예외/우선적용 관계를 명확히 하는 보완이 필요합니다.`;
  }
  return `충돌 신호는 제한적이라 비교적 괜찮지만, ${pair} 같은 경계 사례는 추가 점검이 바람직합니다.`;
}

function buildConstitutionalStatement(constitutional, constitutionalDetail, constitutionalEvidence) {
  const ranked = Object.entries(constitutionalDetail).sort((a, b) => b[1] - a[1]);
  const high = ranked.filter((entry) => entry[1] >= 18);

  if (!high.length) {
    if (constitutional < 30) {
      return "입력된 문구 기준으로는 중대한 위헌 위험 신호가 낮아 헌법원칙 측면에서 비교적 괜찮습니다.";
    }
    return "헌법 위험 점수는 일부 있으나 강한 트리거 문구가 제한적입니다. 근거 조문을 추가해 정밀 검토가 필요합니다.";
  }

  const [topKey, topScore] = high[0];
  const meta = PRINCIPLE_META[topKey];
  const keywordText = joinKeywords(constitutionalEvidence[topKey]);

  if (constitutional >= 70) {
    return `헌법 원칙 위험이 큽니다. 특히 ${meta.label}(${meta.basis}) 관련 점수가 ${topScore}점이며, '${keywordText}' 문구가 확인되어 조문 축소·명확화 보완이 필요합니다.`;
  }
  if (constitutional >= 40) {
    return `헌법 원칙상 취약점이 있습니다. ${meta.label}(${meta.basis}) 관련 점수가 ${topScore}점이고, '${keywordText}' 표현이 있어 보완이 필요합니다.`;
  }
  return `헌법 위험은 상대적으로 낮아 괜찮은 편이지만, ${meta.label}(${meta.basis}) 관련 '${keywordText}' 문구는 사전 정비가 권장됩니다.`;
}

function buildOverallStatement(total, overlap, conflict, constitutional) {
  const drivers = [
    ["중복", overlap],
    ["충돌", conflict],
    ["헌법위험", constitutional],
  ].sort((a, b) => b[1] - a[1]);

  const lead = drivers[0];
  const second = drivers[1];

  if (total >= 75) {
    return `${lead[0]}(${lead[1]}점)와 ${second[0]}(${second[1]}점) 신호가 동시에 커서 현 문안은 그대로 추진하기 어렵습니다. 근거 조문 기준으로 재설계가 필요합니다.`;
  }
  if (total >= 55) {
    return `${lead[0]}(${lead[1]}점) 중심의 리스크가 확인되어 핵심 조문을 정비하면 개선 가능하지만, 현재 상태로는 보완이 필요합니다.`;
  }
  if (total >= 35) {
    return `중대한 경고는 제한적이지만 ${lead[0]}(${lead[1]}점) 이슈가 남아 있어 목적·요건·예외 조문을 보완하는 것이 안전합니다.`;
  }
  return "현재 입력 범위에서는 중대한 충돌·위헌 신호가 낮아 상대적으로 괜찮습니다.";
}

function buildLegalExplanations(conflictEvidence, constitutionalDetail, constitutionalEvidence) {
  const reasons = [];

  conflictEvidence.slice(0, 3).forEach((item) => {
    reasons.push(
      `충돌 근거: ${item.proposed_article}(${modalitiesToText(item.proposed_modes)})와 ${item.existing_article}(${modalitiesToText(item.existing_modes)})가 유사도 ${item.similarity}% 구간에서 반대 규범으로 포착되었습니다.`,
    );
  });

  Object.entries(constitutionalDetail)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .forEach(([principleKey, score]) => {
      if (score < 12) return;
      const meta = PRINCIPLE_META[principleKey];
      const keywords = joinKeywords(constitutionalEvidence[principleKey]);
      reasons.push(
        `헌법 근거: ${meta.label}(${meta.basis}) 위험이 ${score}점입니다. 탐지 문구는 '${keywords}'이고, ${meta.riskReason}`,
      );
    });

  if (!reasons.length) {
    reasons.push("현재 입력 범위에서는 직접적인 충돌/위헌 트리거가 크게 포착되지 않았습니다. 다만 이는 법원의 확정 판단이 아닌 위험도 추정 결과입니다.");
  }

  return reasons;
}

function buildActionFocus(overlap, conflict, constitutionalDetail) {
  const actions = [];

  if (overlap >= 40) {
    actions.push("목적조항과 적용범위 조항에 기존법과의 관계(우선/보충/배제)를 명시해 중복을 줄이세요.");
  }
  if (conflict >= 40) {
    actions.push("의무·금지·면제 조문을 같은 행위 단위로 정렬하고, 상충 시 우선 적용 규칙을 추가하세요.");
  }
  if ((constitutionalDetail.clarity_principle || 0) >= 18) {
    actions.push("'필요한', '적절한' 같은 추상 표현을 정량 요건 또는 절차 요건으로 치환하세요.");
  }
  if ((constitutionalDetail.proportionality_principle || 0) >= 18) {
    actions.push("적용 대상을 좁히고 기간·목적·대상 한계를 넣어 최소침해 구조로 수정하세요.");
  }
  if ((constitutionalDetail.non_delegation_principle || 0) >= 18) {
    actions.push("본질적 사항(대상, 요건, 제재)을 법률 본문에 직접 규정하고 하위법령 위임을 축소하세요.");
  }
  if ((constitutionalDetail.non_retroactivity_principle || 0) >= 18) {
    actions.push("소급 적용 문구를 삭제하거나 경과규정으로 전환해 신뢰보호를 확보하세요.");
  }
  if ((constitutionalDetail.equality_principle || 0) >= 18) {
    actions.push("집단 구분 기준을 객관화하고 차등 취급 사유를 조문에 명시하세요.");
  }

  if (!actions.length) {
    actions.push("현재 구조는 비교적 안정적이지만, 최신 법령/판례 반영 여부를 정기 점검하세요.");
  }

  return actions;
}

function buildAssessment(overlap, conflict, constitutional, totalRisk, overlapEvidence, conflictEvidence, constitutionalDetail, constitutionalEvidence) {
  return {
    overall_statement: buildOverallStatement(totalRisk, overlap, conflict, constitutional),
    metric_statements: {
      overlap: buildOverlapStatement(overlap, overlapEvidence),
      conflict: buildConflictStatement(conflict, conflictEvidence),
      constitutional: buildConstitutionalStatement(constitutional, constitutionalDetail, constitutionalEvidence),
    },
    legal_grounded_reasons: buildLegalExplanations(conflictEvidence, constitutionalDetail, constitutionalEvidence),
    recommended_revision_focus: buildActionFocus(overlap, conflict, constitutionalDetail),
    disclaimer: "이 결과는 위헌 여부의 확정 판단이 아니라, 조문 텍스트 기반 위험도 추정입니다.",
  };
}

function analyzeLaws(proposedLaw, existingLaws) {
  const proposed = buildLawDocument(proposedLaw);
  const existing = existingLaws.map((law) => buildLawDocument(law));

  const overlap = computeOverlapScore(proposed, existing);
  const conflict = computeConflictScore(proposed, existing);
  const constitutional = calcConstitutionalRisk(proposed);

  const totalRisk = Number((0.35 * overlap.score + 0.35 * conflict.score + 0.3 * constitutional.score).toFixed(2));

  const assessment = buildAssessment(
    overlap.score,
    conflict.score,
    constitutional.score,
    totalRisk,
    overlap.evidence,
    conflict.evidence,
    constitutional.breakdown,
    constitutional.keywords,
  );

  return {
    input: {
      proposed_bill: {
        id: proposedLaw.id || "",
        name: proposedLaw.name,
      },
      existing_laws: existingLaws.map((law) => ({ id: law.id || "", name: law.name })),
    },
    scores: {
      overlap_score: overlap.score,
      conflict_score: conflict.score,
      constitutional_risk_score: constitutional.score,
      total_risk_score: totalRisk,
      formula: "0.35*overlap + 0.35*conflict + 0.30*constitutional",
    },
    evidence: {
      overlap_top_matches: overlap.evidence,
      conflict_top_matches: conflict.evidence,
      constitutional_keywords: constitutional.keywords,
    },
    constitutional_breakdown: constitutional.breakdown,
    assessment,
  };
}

module.exports = {
  analyzeLaws,
};
