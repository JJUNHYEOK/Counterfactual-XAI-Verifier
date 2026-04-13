#!/usr/bin/env python3
"""
Rule-based legal overlap/conflict/constitutional-risk analyzer (MVP).

Input:
- One proposed bill text file
- One or more existing law text files

Output:
- JSON report with overlap/conflict/constitutional-risk scores (0-100)
- Reasoned assessment sentences with legal grounds
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


ARTICLE_PATTERN = re.compile(r"(제\s*\d+\s*조(?:의\s*\d+)?)")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]{2,}")


NORMATIVE_MARKERS: Dict[str, Set[str]] = {
    "prohibit": {"금지", "아니된다", "할 수 없다", "불가", "제한한다"},
    "require": {"하여야 한다", "해야 한다", "의무", "신고하여야", "제출하여야"},
    "allow": {"할 수 있다", "허용", "가능", "인정"},
    "exempt": {"면제", "예외", "적용하지 아니한다", "적용하지 않는다"},
}


MODE_LABELS = {
    "prohibit": "금지",
    "require": "의무",
    "allow": "허용",
    "exempt": "면제",
}


TOPIC_KEYWORDS: Dict[str, Set[str]] = {
    "privacy": {"개인정보", "사생활", "통신", "위치정보", "감시", "추적"},
    "property": {"재산", "수용", "보상", "부담금", "과징금", "세금"},
    "expression": {"표현", "언론", "출판", "집회", "시위"},
    "employment": {"근로자", "노동", "임금", "해고", "사업주"},
    "criminal": {"처벌", "징역", "벌금", "형사", "구금"},
    "administrative": {"행정", "허가", "인가", "등록", "신고"},
}


VAGUE_TERMS = {
    "적절한",
    "필요한",
    "상당한",
    "공익상",
    "신속히",
    "원활히",
    "충분한",
    "현저한",
    "긴급한",
}

BROAD_SCOPE_TERMS = {"모든", "전부", "일체", "누구든지", "항상", "즉시"}
DELEGATION_TERMS = {"대통령령으로 정한다", "총리령으로 정한다", "부령으로 정한다", "정하는 바에 따라"}
RETROACTIVE_TERMS = {"소급", "이미 발생한", "종전의", "과거의 행위"}
EQUALITY_RISK_TERMS = {"특정 집단", "일부 국민", "정치적 견해", "성별", "출신지역"}


PRINCIPLE_META = {
    "clarity_principle": {
        "label": "명확성 원칙",
        "basis": "헌법상 법치국가원리·적법절차 원칙",
        "risk_reason": "추상적 문구가 많으면 규율 대상과 요건의 예측가능성이 떨어집니다.",
    },
    "proportionality_principle": {
        "label": "비례성 원칙",
        "basis": "헌법 제37조 제2항",
        "risk_reason": "적용 범위와 제한 강도가 넓으면 최소침해 심사에서 취약해집니다.",
    },
    "non_delegation_principle": {
        "label": "포괄위임금지 원칙",
        "basis": "헌법 제75조",
        "risk_reason": "본질적 사항을 하위법령에 넘기는 구조로 해석될 수 있습니다.",
    },
    "non_retroactivity_principle": {
        "label": "소급입법 제한 원칙",
        "basis": "헌법 제13조 제2항",
        "risk_reason": "이미 종료된 사실관계에 불이익을 부과하면 신뢰보호 문제가 큽니다.",
    },
    "equality_principle": {
        "label": "평등 원칙",
        "basis": "헌법 제11조 제1항",
        "risk_reason": "집단 구분 기준이 불명확하면 자의적 차별 문제로 이어질 수 있습니다.",
    },
}


OPPOSITION = {
    ("prohibit", "allow"),
    ("allow", "prohibit"),
    ("require", "exempt"),
    ("exempt", "require"),
}


STOPWORDS = {
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
}


@dataclass
class Article:
    law_name: str
    article_id: str
    text: str
    tokens: Set[str] = field(default_factory=set)
    modalities: Set[str] = field(default_factory=set)
    topic_tags: Set[str] = field(default_factory=set)


@dataclass
class LawDocument:
    name: str
    text: str
    articles: List[Article]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_articles(law_name: str, text: str) -> List[Article]:
    clean = text.strip()
    if not clean:
        return []

    matches = list(ARTICLE_PATTERN.finditer(clean))
    if not matches:
        return [Article(law_name=law_name, article_id="본문", text=normalize_space(clean))]

    articles: List[Article] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        segment = clean[start:end].strip()
        article_id = normalize_space(match.group(1))
        articles.append(Article(law_name=law_name, article_id=article_id, text=normalize_space(segment)))
    return articles


def tokenize(text: str) -> Set[str]:
    raw = TOKEN_PATTERN.findall(text)
    tokens = {tok.lower() for tok in raw if tok not in STOPWORDS}
    return tokens


def detect_modalities(text: str) -> Set[str]:
    result: Set[str] = set()
    for mode, markers in NORMATIVE_MARKERS.items():
        for marker in markers:
            if marker in text:
                result.add(mode)
                break
    return result


def detect_topics(text: str) -> Set[str]:
    tags: Set[str] = set()
    for topic, words in TOPIC_KEYWORDS.items():
        if any(word in text for word in words):
            tags.add(topic)
    return tags


def enrich_articles(articles: Iterable[Article]) -> None:
    for article in articles:
        article.tokens = tokenize(article.text)
        article.modalities = detect_modalities(article.text)
        article.topic_tags = detect_topics(article.text)


def read_law(path: Path) -> LawDocument:
    text = path.read_text(encoding="utf-8")
    articles = split_articles(path.stem, text)
    enrich_articles(articles)
    return LawDocument(name=path.stem, text=text, articles=articles)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def article_similarity(a: Article, b: Article) -> float:
    token_sim = jaccard(a.tokens, b.tokens)
    topic_sim = jaccard(a.topic_tags, b.topic_tags) if (a.topic_tags or b.topic_tags) else 0.0
    return 0.75 * token_sim + 0.25 * topic_sim


def compute_overlap_score(proposed: LawDocument, existing: Sequence[LawDocument]) -> Tuple[float, List[Dict[str, object]]]:
    all_existing_articles: List[Article] = [art for law in existing for art in law.articles]
    if not proposed.articles or not all_existing_articles:
        return 0.0, []

    pairs: List[Tuple[float, Article, Article]] = []
    per_article_best: List[float] = []
    for p_art in proposed.articles:
        best_score = 0.0
        best_match: Article | None = None
        for e_art in all_existing_articles:
            sim = article_similarity(p_art, e_art)
            if sim > best_score:
                best_score = sim
                best_match = e_art
        per_article_best.append(best_score)
        if best_match is not None:
            pairs.append((best_score, p_art, best_match))

    overlap = 100.0 * (sum(per_article_best) / max(len(per_article_best), 1))
    top_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)[:5]
    evidence = [
        {
            "proposed_article": f"{p.law_name} {p.article_id}",
            "existing_article": f"{e.law_name} {e.article_id}",
            "similarity": round(score * 100, 2),
        }
        for score, p, e in top_pairs
    ]
    return round(overlap, 2), evidence


def has_normative_conflict(a_modes: Set[str], b_modes: Set[str]) -> bool:
    for pair in OPPOSITION:
        if pair[0] in a_modes and pair[1] in b_modes:
            return True
    return False


def compute_conflict_score(proposed: LawDocument, existing: Sequence[LawDocument]) -> Tuple[float, List[Dict[str, object]]]:
    all_existing_articles: List[Article] = [art for law in existing for art in law.articles]
    if not proposed.articles or not all_existing_articles:
        return 0.0, []

    weighted_conflict = 0.0
    weighted_total = 0.0
    evidence: List[Dict[str, object]] = []

    for p_art in proposed.articles:
        for e_art in all_existing_articles:
            sim = article_similarity(p_art, e_art)
            if sim < 0.18:
                continue
            weight = sim
            weighted_total += weight

            conflict = has_normative_conflict(p_art.modalities, e_art.modalities)
            if conflict:
                weighted_conflict += weight
                evidence.append(
                    {
                        "proposed_article": f"{p_art.law_name} {p_art.article_id}",
                        "existing_article": f"{e_art.law_name} {e_art.article_id}",
                        "similarity": round(sim * 100, 2),
                        "proposed_modes": sorted(list(p_art.modalities)),
                        "existing_modes": sorted(list(e_art.modalities)),
                    }
                )

    if weighted_total == 0:
        return 0.0, []

    conflict_score = 100.0 * (weighted_conflict / weighted_total)
    top = sorted(evidence, key=lambda x: x["similarity"], reverse=True)[:7]
    return round(conflict_score, 2), top


def count_terms(text: str, terms: Iterable[str]) -> int:
    return sum(1 for t in terms if t in text)


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(value, hi))


def calc_constitutional_risk(proposed: LawDocument) -> Tuple[float, Dict[str, float], Dict[str, List[str]]]:
    text = normalize_space(proposed.text)
    text_len_factor = max(1.0, math.log10(max(len(text), 10)))

    vague_count = count_terms(text, VAGUE_TERMS)
    broad_count = count_terms(text, BROAD_SCOPE_TERMS)
    delegation_count = count_terms(text, DELEGATION_TERMS)
    retro_count = count_terms(text, RETROACTIVE_TERMS)
    equality_count = count_terms(text, EQUALITY_RISK_TERMS)
    criminal_count = count_terms(text, TOPIC_KEYWORDS["criminal"])
    rights_count = count_terms(text, TOPIC_KEYWORDS["privacy"] | TOPIC_KEYWORDS["expression"] | TOPIC_KEYWORDS["property"])

    clarity_risk = clamp((vague_count / text_len_factor) * 11.0)
    proportionality_risk = clamp((broad_count * 8.0 + criminal_count * 4.0 + rights_count * 3.0) / text_len_factor)
    delegation_risk = clamp((delegation_count * 18.0 + vague_count * 3.0) / text_len_factor)
    retroactivity_risk = clamp((retro_count * 26.0) / text_len_factor)
    equality_risk = clamp((equality_count * 22.0 + broad_count * 2.0) / text_len_factor)

    by_principle = {
        "clarity_principle": round(clarity_risk, 2),
        "proportionality_principle": round(proportionality_risk, 2),
        "non_delegation_principle": round(delegation_risk, 2),
        "non_retroactivity_principle": round(retroactivity_risk, 2),
        "equality_principle": round(equality_risk, 2),
    }
    overall = round(
        0.22 * clarity_risk
        + 0.30 * proportionality_risk
        + 0.20 * delegation_risk
        + 0.16 * retroactivity_risk
        + 0.12 * equality_risk,
        2,
    )

    evidence = {
        "clarity_principle": [t for t in VAGUE_TERMS if t in text][:8],
        "proportionality_principle": [t for t in BROAD_SCOPE_TERMS if t in text][:8],
        "non_delegation_principle": [t for t in DELEGATION_TERMS if t in text][:8],
        "non_retroactivity_principle": [t for t in RETROACTIVE_TERMS if t in text][:8],
        "equality_principle": [t for t in EQUALITY_RISK_TERMS if t in text][:8],
    }
    return overall, by_principle, evidence


def modalities_to_text(modes: Sequence[str] | Set[str]) -> str:
    if not modes:
        return "규범표현 미검출"
    if isinstance(modes, set):
        normalized = sorted(list(modes))
    else:
        normalized = sorted(list(modes))
    return "/".join(MODE_LABELS.get(mode, mode) for mode in normalized)


def join_keywords(keywords: Sequence[str]) -> str:
    if not keywords:
        return "관련 키워드 미검출"
    return ", ".join(keywords)


def build_overlap_statement(overlap: float, overlap_evidence: List[Dict[str, object]]) -> str:
    if not overlap_evidence:
        return "입력된 비교 범위에서 직접적인 중복 근거가 적어, 현재로서는 중복 위험이 상대적으로 괜찮은 편입니다."

    top = overlap_evidence[0]
    pair_text = f"{top['proposed_article']} ↔ {top['existing_article']}(유사도 {top['similarity']}%)"

    if overlap >= 70:
        return f"기존 법령과 핵심 규율 영역이 크게 겹칩니다. {pair_text} 근거가 확인되어 조문 목적과 적용 범위를 재설계하는 보완이 필요합니다."
    if overlap >= 40:
        return f"일부 핵심 조문이 기존 법과 중첩됩니다. {pair_text} 근거가 있어 중복을 줄이는 문구 정리가 필요합니다."
    return f"중복 신호는 제한적이라 비교적 괜찮습니다. 가장 높은 중첩 근거는 {pair_text} 수준입니다."


def build_conflict_statement(conflict: float, conflict_evidence: List[Dict[str, object]]) -> str:
    if not conflict_evidence:
        if conflict < 25:
            return "동일 행위에 대한 정면 충돌 근거가 뚜렷하지 않아 현재 범위에서는 규범 정합성이 비교적 괜찮습니다."
        return "충돌 점수는 일부 있으나 명시적 반대 규범 근거가 제한적입니다. 유사 조문 추가 검토가 필요합니다."

    top = conflict_evidence[0]
    proposed_modes = modalities_to_text(top.get("proposed_modes", []))
    existing_modes = modalities_to_text(top.get("existing_modes", []))
    pair_text = (
        f"{top['proposed_article']}({proposed_modes})와 "
        f"{top['existing_article']}({existing_modes})"
    )

    if conflict >= 70:
        return f"유사 행위를 두고 반대 규범이 병존합니다. {pair_text}가 대표 근거라 입법 충돌 방지를 위한 조문 재정렬이 필요합니다."
    if conflict >= 40:
        return f"규범 충돌 신호가 확인됩니다. {pair_text} 근거가 있어 예외/우선적용 관계를 명확히 하는 보완이 필요합니다."
    return f"충돌 신호는 제한적이라 비교적 괜찮지만, {pair_text} 같은 경계 사례는 추가 점검이 바람직합니다."


def build_constitutional_statement(
    constitutional: float,
    constitutional_detail: Dict[str, float],
    constitutional_evidence: Dict[str, List[str]],
) -> str:
    ranked = sorted(constitutional_detail.items(), key=lambda x: x[1], reverse=True)
    high = [item for item in ranked if item[1] >= 18]

    if not high:
        if constitutional < 30:
            return "입력된 문구 기준으로는 중대한 위헌 위험 신호가 낮아 헌법원칙 측면에서 비교적 괜찮습니다."
        return "헌법 위험 점수는 일부 있으나 강한 트리거 문구가 제한적입니다. 근거 조문을 추가해 정밀 검토가 필요합니다."

    top_key, top_score = high[0]
    top_meta = PRINCIPLE_META[top_key]
    top_keywords = join_keywords(constitutional_evidence.get(top_key, []))

    if constitutional >= 70:
        return (
            f"헌법 원칙 위험이 큽니다. 특히 {top_meta['label']}({top_meta['basis']}) 관련 점수가 {top_score}점이며, "
            f"'{top_keywords}' 문구가 확인되어 조문 축소·명확화 보완이 필요합니다."
        )
    if constitutional >= 40:
        return (
            f"헌법 원칙상 취약점이 있습니다. {top_meta['label']}({top_meta['basis']}) 관련 점수가 {top_score}점이고, "
            f"'{top_keywords}' 표현이 있어 보완이 필요합니다."
        )
    return (
        f"헌법 위험은 상대적으로 낮아 괜찮은 편이지만, {top_meta['label']}({top_meta['basis']}) 관련 "
        f"'{top_keywords}' 문구는 사전 정비가 권장됩니다."
    )


def build_overall_statement(total: float, overlap: float, conflict: float, constitutional: float) -> str:
    drivers = sorted(
        [
            ("중복", overlap),
            ("충돌", conflict),
            ("헌법위험", constitutional),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    lead_driver, lead_score = drivers[0]
    second_driver, second_score = drivers[1]

    if total >= 75:
        return (
            f"{lead_driver}({lead_score}점)와 {second_driver}({second_score}점) 신호가 동시에 커서 "
            "현 문안은 그대로 추진하기 어렵습니다. 근거 조문 기준으로 재설계가 필요합니다."
        )
    if total >= 55:
        return (
            f"{lead_driver}({lead_score}점) 중심의 리스크가 확인되어 "
            "핵심 조문을 정비하면 개선 가능하지만, 현재 상태로는 보완이 필요합니다."
        )
    if total >= 35:
        return (
            f"중대한 경고는 제한적이지만 {lead_driver}({lead_score}점) 이슈가 남아 있어 "
            "목적·요건·예외 조문을 보완하는 것이 안전합니다."
        )
    return "현재 입력 범위에서는 중대한 충돌·위헌 신호가 낮아 상대적으로 괜찮습니다."


def build_legal_explanations(
    conflict_evidence: List[Dict[str, object]],
    constitutional_detail: Dict[str, float],
    constitutional_evidence: Dict[str, List[str]],
) -> List[str]:
    reasons: List[str] = []

    for item in conflict_evidence[:3]:
        proposed_modes = modalities_to_text(item.get("proposed_modes", []))
        existing_modes = modalities_to_text(item.get("existing_modes", []))
        reasons.append(
            f"충돌 근거: {item['proposed_article']}({proposed_modes})와 "
            f"{item['existing_article']}({existing_modes})가 유사도 {item['similarity']}% 구간에서 반대 규범으로 포착되었습니다."
        )

    ranked = sorted(constitutional_detail.items(), key=lambda x: x[1], reverse=True)
    for principle_key, score in ranked[:3]:
        if score < 12:
            continue
        meta = PRINCIPLE_META[principle_key]
        keywords = join_keywords(constitutional_evidence.get(principle_key, []))
        reasons.append(
            f"헌법 근거: {meta['label']}({meta['basis']}) 위험이 {score}점입니다. "
            f"탐지 문구는 '{keywords}'이고, {meta['risk_reason']}"
        )

    if not reasons:
        reasons.append(
            "현재 입력 범위에서는 직접적인 충돌/위헌 트리거가 크게 포착되지 않았습니다. "
            "다만 이는 법원의 확정 판단이 아닌 위험도 추정 결과입니다."
        )

    return reasons


def build_action_focus(
    overlap: float,
    conflict: float,
    constitutional_detail: Dict[str, float],
) -> List[str]:
    actions: List[str] = []

    if overlap >= 40:
        actions.append("목적조항과 적용범위 조항에 기존법과의 관계(우선/보충/배제)를 명시해 중복을 줄이세요.")
    if conflict >= 40:
        actions.append("의무·금지·면제 조문을 같은 행위 단위로 정렬하고, 상충 시 우선 적용 규칙을 추가하세요.")

    if constitutional_detail.get("clarity_principle", 0.0) >= 18:
        actions.append("'필요한', '적절한' 같은 추상 표현을 정량 요건 또는 절차 요건으로 치환하세요.")
    if constitutional_detail.get("proportionality_principle", 0.0) >= 18:
        actions.append("적용 대상을 좁히고 기간·목적·대상 한계를 넣어 최소침해 구조로 수정하세요.")
    if constitutional_detail.get("non_delegation_principle", 0.0) >= 18:
        actions.append("본질적 사항(대상, 요건, 제재)을 법률 본문에 직접 규정하고 하위법령 위임을 축소하세요.")
    if constitutional_detail.get("non_retroactivity_principle", 0.0) >= 18:
        actions.append("소급 적용 문구를 삭제하거나 경과규정으로 전환해 신뢰보호를 확보하세요.")
    if constitutional_detail.get("equality_principle", 0.0) >= 18:
        actions.append("집단 구분 기준을 객관화하고 차등 취급 사유를 조문에 명시하세요.")

    if not actions:
        actions.append("현재 구조는 비교적 안정적이지만, 최신 법령/판례 반영 여부를 정기 점검하세요.")

    return actions


def build_assessment(
    overlap: float,
    conflict: float,
    constitutional: float,
    total_risk: float,
    overlap_evidence: List[Dict[str, object]],
    conflict_evidence: List[Dict[str, object]],
    constitutional_detail: Dict[str, float],
    constitutional_evidence: Dict[str, List[str]],
) -> Dict[str, object]:
    metric_statements = {
        "overlap": build_overlap_statement(overlap, overlap_evidence),
        "conflict": build_conflict_statement(conflict, conflict_evidence),
        "constitutional": build_constitutional_statement(
            constitutional,
            constitutional_detail,
            constitutional_evidence,
        ),
    }

    return {
        "overall_statement": build_overall_statement(total_risk, overlap, conflict, constitutional),
        "metric_statements": metric_statements,
        "legal_grounded_reasons": build_legal_explanations(
            conflict_evidence,
            constitutional_detail,
            constitutional_evidence,
        ),
        "recommended_revision_focus": build_action_focus(
            overlap,
            conflict,
            constitutional_detail,
        ),
        "disclaimer": "이 결과는 위헌 여부의 확정 판단이 아니라, 조문 텍스트 기반 위험도 추정입니다.",
    }


def analyze(proposed_path: Path, existing_paths: Sequence[Path]) -> Dict[str, object]:
    proposed = read_law(proposed_path)
    existing = [read_law(p) for p in existing_paths]

    overlap, overlap_evidence = compute_overlap_score(proposed, existing)
    conflict, conflict_evidence = compute_conflict_score(proposed, existing)
    constitutional, constitutional_detail, constitutional_evidence = calc_constitutional_risk(proposed)

    total_risk = round(0.35 * overlap + 0.35 * conflict + 0.30 * constitutional, 2)

    assessment = build_assessment(
        overlap,
        conflict,
        constitutional,
        total_risk,
        overlap_evidence,
        conflict_evidence,
        constitutional_detail,
        constitutional_evidence,
    )

    return {
        "input": {
            "proposed_bill": str(proposed_path),
            "existing_laws": [str(p) for p in existing_paths],
        },
        "scores": {
            "overlap_score": overlap,
            "conflict_score": conflict,
            "constitutional_risk_score": constitutional,
            "total_risk_score": total_risk,
            "formula": "0.35*overlap + 0.35*conflict + 0.30*constitutional",
        },
        "evidence": {
            "overlap_top_matches": overlap_evidence,
            "conflict_top_matches": conflict_evidence,
            "constitutional_keywords": constitutional_evidence,
        },
        "constitutional_breakdown": constitutional_detail,
        "assessment": assessment,
    }


def resolve_existing_paths(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted([p for p in target.glob("*.txt") if p.is_file()])
        return files
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze overlap/conflict/constitutional risk of a bill.")
    parser.add_argument("--proposed", required=True, help="Path to proposed bill text file.")
    parser.add_argument(
        "--existing",
        required=True,
        help="Path to existing law text file OR directory containing *.txt files.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    proposed_path = Path(args.proposed)
    existing_target = Path(args.existing)

    if not proposed_path.exists():
        raise SystemExit(f"Proposed file not found: {proposed_path}")
    existing_paths = resolve_existing_paths(existing_target)
    if not existing_paths:
        raise SystemExit(f"No existing law files found in: {existing_target}")

    report = analyze(proposed_path, existing_paths)
    payload = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
