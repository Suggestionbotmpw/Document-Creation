import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from itertools import product

# ================= CONFIG =================

INPUT_DIR = Path("raw_docs1")     # cleaned JSONs
OUTPUT_FILE = Path("training_data.jsonl")

MAX_RECS = 3
MAX_REASONS = 3
MAX_EVIDENCE = 2

MIN_LEN = 40
MAX_LEN = 280

BAD_PATTERNS = [
    r"decoded html",
    r"http",
    r"<",
    r">",
    r"figure",
    r"exhibit",
    r"table",
    r"reaction at",
]

# ================= HELPERS =================

def strip_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "img"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text()).strip()


def clean_sentence(text: str) -> str:
    if not text:
        return ""

    text = strip_html(text)

    for p in BAD_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    text = re.sub(r"[:;]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < MIN_LEN:
        return ""

    if len(text) > MAX_LEN:
        text = text[:MAX_LEN].rsplit(".", 1)[0] + "."

    return text


def trim_structure(solutions, nr, ns, ne):
    trimmed = []

    for sol in solutions[:nr]:
        reasons = []

        for r in sol["SupportingReasons"][:ns]:
            evidence = r["Evidence"][:ne]
            if not evidence:
                continue

            reasons.append({
                "SupportingReason": r["SupportingReason"],
                "Evidence": evidence
            })

        if reasons:
            trimmed.append({
                "Recommendation": sol["Recommendation"],
                "Result": sol["Result"],
                "SupportingReasons": reasons
            })

    return trimmed

# ================= CORE =================

def document_to_samples(doc: dict):
    base = []

    for sol in doc.get("Solutions", []):
        rec = clean_sentence(sol.get("Recommendation"))
        res = clean_sentence(sol.get("Result"))

        if not rec or not res:
            continue

        reasons = []

        for r in sol.get("SupportingReasons", []):
            sr = clean_sentence(r.get("SupportingReason"))
            if not sr:
                continue

            ev_list = []

            for ev in r.get("Evidence", []):
                ev_text = clean_sentence(ev.get("EvidenceContent"))
                if ev_text:
                    ev_list.append({
                        "EvidenceTitle": "",
                        "EvidenceContent": ev_text
                    })

            if ev_list:
                reasons.append({
                    "SupportingReason": sr,
                    "Evidence": ev_list
                })

        if reasons:
            base.append({
                "Recommendation": rec,
                "Result": res,
                "SupportingReasons": reasons
            })

    if not base:
        return []

    samples = []

    for nr, ns, ne in product(
        range(1, min(len(base), MAX_RECS) + 1),
        range(1, MAX_REASONS + 1),
        range(1, MAX_EVIDENCE + 1),
    ):
        trimmed = trim_structure(base, nr, ns, ne)
        if not trimmed:
            continue

        samples.append({
            "instruction": "Generate a professional technical recommendation document. Strictly follow the numeric requirements.",
            "input": {
                "Heading": doc.get("Heading", ""),
                "CustomerNeed": doc.get("CustomerNeed", ""),
                "Impact": doc.get("Impact", ""),
                "ServicePerformed": doc.get("ServicePerformed", ""),
                "ServiceSummaryTitle": doc.get("ServiceSummaryTitle", ""),
                "ServiceSummaryContent": doc.get("ServiceSummaryContent", ""),
                "Language": doc.get("Language", "en-US"),
                "NumRecommendations": nr,
                "NumSupportingReasons": ns,
                "NumEvidence": ne
            },
            "output": {
                "Solutions": trimmed
            }
        })

    return samples

# ================= RUN =================

def main():
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for file in INPUT_DIR.glob("*.json"):
            doc = json.loads(file.read_text(encoding="utf-8"))
            for sample in document_to_samples(doc):
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    print(f"✅ Training samples created: {total}")
    print(f"📄 Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
