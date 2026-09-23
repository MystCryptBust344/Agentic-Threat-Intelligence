"""
triplet_extractor.py  —  Phase 1 v2
=====================================
Fixed NER + real per-triplet model confidence.

Key v2 fixes:
  1. NER uses two tiers:
       Tier 1: Explicit entity lists (30+ APTs, 30+ malware) -> confidence cap 1.0
       Tier 2: Capitalized spans that co-occur with relation verbs (novel entities)
               -> labelled "Candidate-Entity", confidence capped at 0.65
       Stopword guard: "used", "against", "Government", etc. never tagged.
  2. extract_from_tire_sample() calls predict_proba() per triplet -> real varying c_NLP.
     v1 used TIRE_SOURCE_CONFIDENCE = 0.85 (constant) — removed entirely.
  3. Adds diagnose_confidence_distribution() method for collapse verification.
"""

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
FAST_MODEL_PATH = "models/fast_extractor.pkl"
LABEL_MAP_FILE  = "models/label_map.json"


# ══════════════════════════════════════════════════════════════════════════════
#  NER  —  Conservative, two-tier entity recognition
# ══════════════════════════════════════════════════════════════════════════════

# ── Tier 1 explicit lists ──────────────────────────────────────────────────────
_KNOWN_APT = [
    "APT28", "APT29", "APT30", "APT32", "APT33", "APT34", "APT38", "APT41",
    "APT10", "APT15", "APT16", "APT17", "APT18", "APT19",
    "admin@338", "Lazarus Group", "Cozy Bear", "Fancy Bear", "Sandworm",
    "Kimsuky", "OceanLotus", "Turla", "Winnti", "MuddyWater", "Comment Crew",
    "menuPass", "Charming Kitten", "Equation Group", "The Shadow Brokers",
    "Gamaredon", "TA505", "FIN7", "FIN6", "FIN4", "Carbanak", "DarkHydrus",
    "Dragonfly", "Energetic Bear", "Machete", "Sidewinder", "Bitter",
    "Transparent Tribe", "Gorgon Group", "Cobalt Group",
]

_KNOWN_MALWARE = [
    "WannaCry", "NotPetya", "Emotet", "Stuxnet", "BlackEnergy", "Gh0stRAT",
    "PlugX", "Cobalt Strike", "Mimikatz", "Metasploit", "Poison Ivy",
    "DarkComet", "njRAT", "AsyncRAT", "QuasarRAT", "NanoCore", "NetWire",
    "AgentTesla", "FormBook", "Lokibot", "Remcos", "Ursnif", "Trickbot",
    "BazarLoader", "Ryuk", "Conti", "REvil", "DarkSide", "BlackMatter",
    "Dridex", "ZLoader", "IcedID", "SolarWinds", "SUNBURST", "TEARDROP",
    "Havoc", "Brute Ratel", "Sliver", "CobaltStrike", "Meterpreter",
    "PoisonIvy", "FinFisher", "Regin", "Duqu", "Flame",
    "Industroyer", "CRASHOVERRIDE", "Petya", "Bad Rabbit",
    "GandCrab", "Sodinokibi", "Mailto", "Snake",
]

_KNOWN_ATTACK_TECHNIQUES = [
    "spear phishing", "spearphishing", "credential dumping", "credential harvesting",
    "lateral movement", "privilege escalation", "command and control",
    "watering hole", "supply chain attack", "supply chain compromise",
    "man-in-the-middle", "SQL injection", "buffer overflow",
    "DDoS", "distributed denial of service", "brute force",
    "pass the hash", "golden ticket", "Kerberoasting",
    "living off the land", "fileless malware", "fileless attack",
    "drive-by download", "typosquatting", "domain spoofing",
    "phishing", "vishing", "smishing", "pretexting",
]

def _alt(*names: List[str]) -> str:
    """Build regex alternation from a list, longest-first to avoid prefix shadowing."""
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))

_RE_APT       = re.compile(r"\b(?:" + _alt(*_KNOWN_APT) + r"|APT\s*\d+)\b", re.IGNORECASE)
_RE_MALWARE   = re.compile(r"\b(?:" + _alt(*_KNOWN_MALWARE) + r")\b", re.IGNORECASE)
_RE_TECHNIQUE = re.compile(
    r"\b(?:T\d{4}(?:\.\d{3})?|" + _alt(*_KNOWN_ATTACK_TECHNIQUES) + r")\b",
    re.IGNORECASE
)
_RE_CVE    = re.compile(r"\bCVE-\d{4}-\d{4,7}\b")
_RE_DOMAIN = re.compile(
    r"\b(?:https?://)?(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+(?:com|net|org|io|gov|edu|mil|ru|cn|ir|kp|int)\b"
)
_RE_IDENTITY = re.compile(
    r"\b(?:(?:financial|healthcare|energy|defense|government|banking|retail|"
    r"aerospace|telecommunications|critical\s+infrastructure|media|research|"
    r"pharmaceutical)\s+(?:sector|industry|organization(?:s)?|company|companies|"
    r"agencies?|entities?))\b",
    re.IGNORECASE
)
_RE_SECTEAM = re.compile(
    r"\b(?:CERT|CSIRT|SOC|NOC|Red\s+Team|Blue\s+Team|Security\s+Operations|"
    r"Incident\s+Response)\b",
    re.IGNORECASE
)

# ── Tier 2: candidate entities from relation-verb context ──────────────────────
# Matches: "[relation verb] [Capitalized Multi-Word Span]"
# These are novel/emerging entities not in the explicit lists.
_RE_CANDIDATE = re.compile(
    r"(?:attributed\s+to|used\s+by|deployed\s+by|operated\s+by|"
    r"associated\s+with|linked\s+to|believed\s+to\s+be|identified\s+as|"
    r"tracked\s+(?:as|by)|known\s+as|dubbed|called|carried\s+out\s+by|"
    r"conducted\s+by)\s+([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3})"
)

# ── Common English words that should NEVER be tagged as entities ───────────────
_STOPWORDS = frozenset({
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","not","no","nor","so","yet","both","either","neither",
    "also","just","as","than","if","then","because","since","while","when",
    "where","which","who","whom","what","that","these","those","this","it",
    "its","used","against","using","through","via","into","onto","about",
    "during","after","before","between","among","government","actor","access",
    "system","systems","network","networks","data","information","attack",
    "attacks","target","targets","threat","activity","activities","campaign",
    "campaigns","operation","operations","group","groups","organization",
    "organizations","company","companies","sector","industry","infrastructure",
    "malware","ransomware","backdoor","trojan","worm","exploit","vulnerability",
    "zero","day","code","execution","remote","local","privilege","escalation",
    "lateral","movement","persistence","discovery","collection","exfiltration",
})


def _is_valid_span(text: str) -> bool:
    """Return False if the span is a stopword or contains only stopwords."""
    words = text.lower().split()
    if not words:
        return False
    if all(w in _STOPWORDS for w in words):
        return False
    if len(text.strip()) <= 2:
        return False
    return True


def extract_entities_heuristic(text: str) -> List[Tuple[str, str, float]]:
    """
    Two-tier conservative NER.

    Returns list of (entity_text, entity_label, tier_confidence):
      - Tier 1 explicit matches: tier_confidence = 1.0
      - Tier 2 candidate matches: tier_confidence = 0.65
        These will cap c_NLP at 0.65 (FLAGGED at most) to reflect entity uncertainty.

    v2 fix: "used", "against", "Government", etc. are NEVER tagged.
    """
    results: List[Tuple[str, str, float]] = []
    seen: set = set()

    def _add(pattern: re.Pattern, label: str, conf: float, group: int = 0) -> None:
        for m in pattern.finditer(text):
            span = m.group(group).strip()
            key  = span.lower()
            if span and key not in seen and _is_valid_span(span):
                results.append((span, label, conf))
                seen.add(key)

    # Tier 1 — explicit lists
    _add(_RE_APT,       "Threat-Actor",    1.0)
    _add(_RE_MALWARE,   "Malware",         1.0)
    _add(_RE_TECHNIQUE, "Attack-Pattern",  1.0)
    _add(_RE_CVE,       "Vulnerability",   1.0)
    _add(_RE_DOMAIN,    "Domain",          1.0)
    _add(_RE_IDENTITY,  "Identity",        1.0)
    _add(_RE_SECTEAM,   "Security-Team",   1.0)

    # Tier 2 — candidate entities from relation-verb context
    for m in _RE_CANDIDATE.finditer(text):
        span = m.group(1).strip()
        key  = span.lower()
        if span and key not in seen and _is_valid_span(span):
            results.append((span, "Candidate-Entity", 0.65))
            seen.add(key)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Data class
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExtractedTriplet:
    subject:              str
    subject_label:        str
    relation:             str
    object_:              str
    object_label:         str
    c_nlp:                float   # RAW classifier confidence [0,1] — NOT weighted
    threshold_weight:     float   # Policy weight: 1.0 / raw_conf / 0.3 / 0.0
    effective_confidence: float   # After temporal decay — computed by TKG
    status:               str     # ACCEPTED | FLAGGED | REJECTED
    source_text:          str
    entity_confidence:    float = field(default=1.0)  # Tier confidence of involved entities


# ══════════════════════════════════════════════════════════════════════════════
#  TripletExtractor
# ══════════════════════════════════════════════════════════════════════════════
class TripletExtractor:
    """
    Phase 1 v2 extractor.

    Model: TF-IDF + LogisticRegression loaded from fast_extractor.pkl.
    Confidence: predict_proba() max probability per triplet — VARIES per input.
    NER: explicit lists + candidate-entity tier — no garbage tokens.
    """

    def __init__(self, model_path: str = FAST_MODEL_PATH,
                 label_map_path: str = LABEL_MAP_FILE):
        self._pipeline   = None
        self._label_map: Dict[str, int] = {}
        self._classes:   List[str] = []
        self._model_loaded = False

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(
                f"[TripletExtractor] No model at '{model_path}'.\n"
                f"  -> Run: python -X utf8 train_relation_extractor.py\n"
                f"  -> Falling back to heuristic-only mode (c_nlp=source confidence)."
            )

    def _load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self._pipeline   = saved["pipeline"]
        self._label_map  = saved["label_map"]
        self._classes    = saved["classes"]
        self._model_loaded = True
        val_acc      = saved.get("val_acc", "?")
        maj_acc      = saved.get("majority_acc", "?")
        guard        = saved.get("guard", {})
        print(f"[TripletExtractor] Model loaded: {path}")
        print(f"  val_acc={val_acc:.4f}  majority_baseline={maj_acc:.4f}  "
              f"conf_std={guard.get('std', '?'):.4f}  "
              f"unique_confs={guard.get('unique', '?')}")

    # ── Inference ──────────────────────────────────────────────────────────────
    def _build_feature_str(self, text: str, e1: str, e1_type: str,
                            e2: str, e2_type: str) -> str:
        marked = text
        if e1 and e1 in marked:
            marked = marked.replace(e1, f"[E1] {e1} [/E1]", 1)
        if e2 and e2 in marked and e1 != e2:
            marked = marked.replace(e2, f"[E2] {e2} [/E2]", 1)
        return f"{e1_type} REL {e2_type} | {marked}"

    def _classify_relation(self, feature_str: str) -> Tuple[str, float]:
        """
        Returns (predicted_relation, confidence).
        Confidence = max(predict_proba) — varies uniquely per input.
        """
        if not self._model_loaded:
            return "unknown", 0.0
        proba    = self._pipeline.predict_proba([feature_str])[0]
        pred_idx = int(np.argmax(proba))
        return self._classes[pred_idx], float(proba[pred_idx])

    @staticmethod
    def _apply_threshold(c: float) -> Tuple[Optional[float], str]:
        if c >= 0.90: return 1.0,  "ACCEPTED"
        if c >= 0.70: return c,    "ACCEPTED"
        if c >= 0.50: return 0.3,  "FLAGGED"
        return None, "REJECTED"

    # ── Public API ─────────────────────────────────────────────────────────────
    def extract(self, text: str) -> List[ExtractedTriplet]:
        """Extract triplets from unstructured/live CTI text."""
        entities = extract_entities_heuristic(text)
        triplets: List[ExtractedTriplet] = []
        for i, (e1_text, e1_label, e1_conf) in enumerate(entities):
            for j, (e2_text, e2_label, e2_conf) in enumerate(entities):
                if i == j:
                    continue
                feat     = self._build_feature_str(text, e1_text, e1_label,
                                                    e2_text, e2_label)
                relation, c_nlp = self._classify_relation(feat)
                if relation in ("noRelation", "unknown"):
                    continue
                # Candidate-entity cap: if either entity is Tier 2, cap c_nlp at 0.65
                ent_conf = min(e1_conf, e2_conf)
                c_nlp    = min(c_nlp, ent_conf)
                weight, status = self._apply_threshold(c_nlp)
                triplets.append(ExtractedTriplet(
                    subject=e1_text, subject_label=e1_label,
                    relation=relation,
                    object_=e2_text, object_label=e2_label,
                    c_nlp=c_nlp,               # Store RAW confidence
                    threshold_weight=weight or 0.0,  # Policy weight separate
                    effective_confidence=0.0,
                    status=status,
                    source_text=text[:200],
                    entity_confidence=ent_conf,
                ))
        return triplets

    def extract_from_tire_sample(self, sample: Dict) -> List[ExtractedTriplet]:
        """
        Extract from a labeled TIRE sample.

        v2 FIX: Uses model predict_proba() per triplet — NOT the constant
        TIRE_SOURCE_CONFIDENCE = 0.85.

        v2 CORRECTION (double-penalty fix): c_nlp stores the RAW classifier
        confidence. threshold_weight is stored separately so ingest_triplet
        does not re-apply the threshold to an already-weighted value.
        """
        text     = sample.get("text", "")
        entities = sample.get("entities", [])
        triplets: List[ExtractedTriplet] = []

        for rel in sample.get("relations", []):
            rel_type        = rel[0]
            ent_idx1, ent_idx2 = rel[1], rel[2]
            if rel_type == "noRelation":
                continue
            if ent_idx1 >= len(entities) or ent_idx2 >= len(entities):
                continue

            e1 = entities[ent_idx1]
            e2 = entities[ent_idx2]
            e1_name  = e1[2] if len(e1) > 2 else ""
            e1_label = e1[3] if len(e1) > 3 else "Unknown"
            e2_name  = e2[2] if len(e2) > 2 else ""
            e2_label = e2[3] if len(e2) > 3 else "Unknown"

            if self._model_loaded:
                # Real model inference — varies per (text, e1, e2) triple
                feat = self._build_feature_str(text, e1_name, e1_label,
                                               e2_name, e2_label)
                _, c_nlp = self._classify_relation(feat)
            else:
                # No model: use conservative source confidence (fallback only)
                c_nlp = 0.72   # deliberately below 0.90 so FLAGGED branch can trigger

            weight, status = self._apply_threshold(c_nlp)
            triplets.append(ExtractedTriplet(
                subject=e1_name, subject_label=e1_label,
                relation=rel_type,
                object_=e2_name, object_label=e2_label,
                c_nlp=c_nlp,               # Store RAW confidence (not raw * weight)
                threshold_weight=weight or 0.0,  # Policy weight separate
                effective_confidence=0.0,
                status=status,
                source_text=text[:200],
            ))

        return triplets

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def diagnose_confidence_distribution(
        self,
        tire_path: str = "../TIRE/dnrti_aug_stix2_je.json",
        n_samples: int = 300,
    ) -> Dict:
        """
        Run inference on the last n_samples of TIRE (test split) and report:
          - Global confidence distribution
          - Per-class confidence table
          - Bucket histogram (ACCEPTED/FLAGGED/REJECTED)
          - Collapse detection (unique values, std)
        """
        if not self._model_loaded:
            print("[Diagnostics] No model loaded.")
            return {}

        if not os.path.exists(tire_path):
            print(f"[Diagnostics] TIRE dataset not found at {tire_path}.")
            return {}

        with open(tire_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        test_samples = raw[-n_samples:]

        all_confs:    List[float] = []
        all_rels:     List[str]   = []
        bucket_counts = {"ACCEPTED": 0, "FLAGGED": 0, "REJECTED": 0}

        for sample in test_samples:
            text  = sample.get("text", "")
            ents  = sample.get("entities", [])
            for rel in sample.get("relations", []):
                rt, i1, i2 = rel[0], rel[1], rel[2]
                if rt == "noRelation" or i1 >= len(ents) or i2 >= len(ents):
                    continue
                e1 = ents[i1]; e2 = ents[i2]
                feat = self._build_feature_str(
                    text,
                    e1[2] if len(e1) > 2 else "",
                    e1[3] if len(e1) > 3 else "UNK",
                    e2[2] if len(e2) > 2 else "",
                    e2[3] if len(e2) > 3 else "UNK",
                )
                _, conf = self._classify_relation(feat)
                all_confs.append(conf)
                all_rels.append(rt)
                _, status = self._apply_threshold(conf)
                bucket_counts[status] += 1

        confs  = np.array(all_confs)
        unique = len(set(confs.round(3)))
        std    = float(confs.std())

        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  DIAGNOSTIC — Confidence Distribution  ({len(confs)} triplets)")
        print(sep)
        print(f"  Min     : {confs.min():.4f}")
        print(f"  Max     : {confs.max():.4f}")
        print(f"  Mean    : {confs.mean():.4f}")
        print(f"  Std     : {std:.4f}   (threshold > {MIN_CONF_STD if False else 0.03})")
        print(f"  Median  : {np.median(confs):.4f}")
        print(f"  P25     : {np.percentile(confs, 25):.4f}")
        print(f"  P75     : {np.percentile(confs, 75):.4f}")
        print(f"  Unique values (3dp): {unique}")

        collapsed = (std < 0.03) or (unique < 20)
        print(f"\n  Model Collapse: {'YES -- WARNING' if collapsed else 'NO -- model is discriminating'}")

        # Bucket histogram
        print(f"\n  Threshold bucket distribution:")
        buckets_def = [
            (0.0,  0.30, "< 0.30",    "REJECTED"),
            (0.30, 0.50, "0.30-0.50", "REJECTED"),
            (0.50, 0.70, "0.50-0.70", "FLAGGED"),
            (0.70, 0.90, "0.70-0.90", "ACCEPTED"),
            (0.90, 1.01, ">= 0.90",   "ACCEPTED"),
        ]
        for lo, hi, label, policy in buckets_def:
            mask  = (confs >= lo) & (confs < hi)
            count = int(mask.sum())
            pct   = 100 * count / max(len(confs), 1)
            bar   = "█" * min(35, int(35 * count / max(len(confs), 1)))
            print(f"  [{label:10s} -> {policy:8s}] {bar:<35s}  {pct:5.1f}%  ({count:,})")

        # Per-relation-class confidence table
        print(f"\n  Per-class confidence breakdown:")
        print(f"  {'Relation':<25} {'N':>5}  {'Mean':>7}  {'Std':>7}  "
              f"{'Min':>7}  {'Max':>7}")
        print(f"  {'-'*25} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        class_data: Dict[str, List[float]] = {}
        for rt, c in zip(all_rels, all_confs):
            class_data.setdefault(rt, []).append(c)
        for rt, cs in sorted(class_data.items(),
                              key=lambda x: np.mean(x[1]), reverse=True):
            ca = np.array(cs)
            print(f"  {rt:<25} {len(ca):>5}  {ca.mean():>7.3f}  "
                  f"{ca.std():>7.3f}  {ca.min():>7.3f}  {ca.max():>7.3f}")

        print(f"\n  Bucket counts:  {bucket_counts}")
        all_buckets_populated = all(v > 0 for v in bucket_counts.values())
        print(f"  All buckets populated: "
              f"{'YES -- UQ is discriminating' if all_buckets_populated else 'NO -- some buckets empty'}")

        return {
            "n": len(confs), "min": float(confs.min()), "max": float(confs.max()),
            "mean": float(confs.mean()), "std": std, "unique": unique,
            "collapsed": collapsed, "buckets": bucket_counts,
            "all_buckets_populated": all_buckets_populated,
        }
