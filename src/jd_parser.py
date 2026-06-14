"""
JD Parser Module
Handles multi-format JD parsing (.docx, .pdf, .txt) with structured signal extraction.
Built specifically for the Redrob AI Senior AI Engineer JD and similar roles.
All configuration data is loaded from the static/ folder — no hardcoded arrays.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JDParser:
    """
    Multi-format JD parser with structured signal extraction.
    Supports .docx, .pdf, and .txt files.
    All data is loaded from static JSON files in static/.
    """

    def __init__(self, static_dir: Optional[Path] = None):
        if static_dir is None:
            static_dir = Path(__file__).parent.parent / "static"
        self.static_dir = static_dir
        self._load_vocabularies()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_json(self, filename: str) -> Optional[Any]:
        filepath = self.static_dir / filename
        if not filepath.exists():
            logger.warning(f"{filename} not found — no fallback available")
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not data:
                    return None
                return data
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None

    def _load_vocabularies(self):
        """Load all configuration data from static JSON files."""

        # skill_vocabulary.json → flat list from all categories
        skill_vocab_raw = self._load_json("skill_vocabulary.json")
        if skill_vocab_raw and "categories" in skill_vocab_raw:
            self.skill_vocab: List[str] = [
                skill
                for skills in skill_vocab_raw["categories"].values()
                for skill in skills
            ]
        else:
            self.skill_vocab = []
            logger.warning("skill_vocabulary.json missing or malformed — skill vocab is empty")

        # skill_aliases.json → aliases dict
        skill_aliases_raw = self._load_json("skill_aliases.json")
        if skill_aliases_raw and "aliases" in skill_aliases_raw:
            self.skill_aliases: Dict[str, str] = skill_aliases_raw["aliases"]
        else:
            self.skill_aliases = {}
            logger.warning("skill_aliases.json missing or malformed — alias expansion disabled")

        # title_hierarchy.json → seniority_scores dict
        title_raw = self._load_json("title_hierarchy.json")
        if title_raw and "seniority_scores" in title_raw:
            self.seniority_map: Dict[str, float] = title_raw["seniority_scores"]
        else:
            self.seniority_map = {}
            logger.warning("title_hierarchy.json missing or malformed — seniority scoring disabled")

        # consulting_firms.json → list of company names
        firms_raw = self._load_json("consulting_firms.json")
        if firms_raw and "consulting_companies" in firms_raw:
            self.disqualifying_companies: List[str] = firms_raw["consulting_companies"]
        else:
            self.disqualifying_companies = []
            logger.warning("consulting_firms.json missing or malformed — company disqualification disabled")

        # section_headers.json → section header patterns
        section_raw = self._load_json("section_headers.json")
        if section_raw and "section_headers" in section_raw:
            self.section_headers: Dict[str, List[str]] = section_raw["section_headers"]
        else:
            self.section_headers = {}
            logger.warning("section_headers.json missing or malformed — section segmentation disabled")

        # disqualifier_patterns.json → {name: [patterns]}
        disq_raw = self._load_json("disqualifier_patterns.json")
        if disq_raw and "disqualifiers" in disq_raw:
            self.disqualifier_patterns: Dict[str, List[str]] = {
                name: entry.get("patterns", [])
                for name, entry in disq_raw["disqualifiers"].items()
            }
        else:
            self.disqualifier_patterns = {}
            logger.warning("disqualifier_patterns.json missing or malformed — disqualifier detection disabled")

        # implied_signals.json
        implied_raw = self._load_json("implied_signals.json")
        if implied_raw and "implied_signals" in implied_raw:
            self.implied_signals_patterns: Dict[str, List[str]] = implied_raw["implied_signals"]
        else:
            self.implied_signals_patterns = {}
            logger.warning("implied_signals.json missing or malformed")

        # culture_signals.json
        culture_raw = self._load_json("culture_signals.json")
        if culture_raw and "culture_signals" in culture_raw:
            self.culture_signals_patterns: Dict[str, List[str]] = culture_raw["culture_signals"]
        else:
            self.culture_signals_patterns = {}
            logger.warning("culture_signals.json missing or malformed")

        # critical_skills.json
        critical_raw = self._load_json("critical_skills.json")
        if critical_raw and "critical_skills" in critical_raw:
            self.critical_skills_for_role: List[str] = critical_raw["critical_skills"]
        else:
            self.critical_skills_for_role = []
            logger.warning("critical_skills.json missing or malformed — critical skill detection disabled")

        # skill_functional_equivalents.json
        equiv_raw = self._load_json("skill_functional_equivalents.json")
        if equiv_raw and "functional_equivalents" in equiv_raw:
            self.skill_functional_equivalents: Dict[str, List[str]] = equiv_raw["functional_equivalents"]
        else:
            self.skill_functional_equivalents = {}
            logger.warning("skill_functional_equivalents.json missing or malformed")

        # skill_canonical_groups.json → {specific_skill: canonical_group}
        canonical_raw = self._load_json("skill_canonical_groups.json")
        if canonical_raw and "canonical_groups" in canonical_raw:
            self.skill_to_canonical_group: Dict[str, str] = canonical_raw["canonical_groups"]
        else:
            self.skill_to_canonical_group = {}
            logger.warning("skill_canonical_groups.json missing or malformed — canonical grouping disabled")

        logger.info("All vocabularies loaded from static files")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, jd_path: str) -> Dict:
        """
        Parse a JD file and return structured signals for candidate comparison.
        text_embedding is left null — rank.py sets it at runtime.
        """
        jd_path = Path(jd_path)
        if not jd_path.exists():
            raise FileNotFoundError(f"JD file not found: {jd_path}")

        raw_text = self._extract_text(jd_path)
        if not raw_text.strip():
            raise ValueError(f"No text extracted from {jd_path}")

        logger.info(f"Extracted {len(raw_text)} chars from {jd_path.name}")

        sections = self._segment_sections(raw_text)
        required_skills, critical_skills = self._extract_required_skills(raw_text, sections)
        nice_to_have = self._extract_preferred_skills(raw_text, sections)

        # Remove overlap
        nice_to_have = [s for s in nice_to_have if s not in required_skills]

        seniority_level, seniority_range = self._extract_seniority(raw_text)
        disqualifiers = self._detect_disqualifiers(raw_text)
        production_signals = self._extract_production_signals(raw_text)
        experience_type = self._extract_experience_type(raw_text)
        implied_signals = self._detect_implied_signals(raw_text)
        culture_signals = self._detect_culture_signals(raw_text)
        skill_weights = self._build_skill_weights(required_skills, critical_skills, nice_to_have)
        confidence = self._compute_confidence(required_skills, critical_skills, seniority_level, sections)

        result = {
            "jd_id": str(uuid.uuid4()),
            "source_file": str(jd_path),
            "parsed_at": datetime.utcnow().isoformat(),

            "required_skills": required_skills,
            "critical_skills": critical_skills,
            "nice_to_have_skills": nice_to_have,
            "skill_weights": skill_weights,
            "skill_functional_equivalents": self.skill_functional_equivalents,

            "seniority_level": seniority_level,
            "seniority_range": seniority_range,

            "domain": self._infer_domain(required_skills),
            "sub_domains": self._infer_sub_domains(required_skills),
            "role_archetype": self._infer_role_archetype(raw_text, implied_signals),

            "disqualifiers": disqualifiers,
            "disqualifying_company_patterns": {
                "company_names": self.disqualifying_companies,
                "disqualify_if": "only-employer",
                "exception": "has_prior_product_company_experience",
            },
            "title_chaser_signals": {
                "max_avg_tenure_months": 18,
                "min_company_switches": 3,
                "requires_title_escalation": True,
            },

            "soft_penalties": self._detect_soft_penalties(raw_text),

            "production_signals": production_signals,
            "experience_type": experience_type,

            "behavioral_availability_signals": {
                "penalize_inactive": True,
                "inactive_threshold_days": 180,
                "response_rate_threshold": 0.30,
                "availability_weight_in_score": 0.10,
            },

            "location_preference": self._extract_locations(raw_text),
            "allows_relocation": self._extract_relocation(raw_text),
            "notice_period_preference": self._extract_notice_period(raw_text),

            "implied_signals": implied_signals,
            "culture_signals": culture_signals,

            "text_embedding": None,  # Set by rank.py at runtime
            "raw_jd_text": raw_text,

            "extraction_confidence": confidence,
        }

        logger.info(
            f"Parsed JD: {len(required_skills)} required skills, "
            f"{len(critical_skills)} critical, "
            f"{len(disqualifiers)} disqualifiers, "
            f"confidence={confidence['overall_confidence']:.2f}"
        )
        return result

    def parse_to_json(self, jd_path: str, output_path: str = None) -> Dict:
        result = self.parse(jd_path)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved parsed JD to {output_path}")
        return result

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, jd_path: Path) -> str:
        suffix = jd_path.suffix.lower()
        if suffix == ".docx":
            return self._extract_docx(jd_path)
        elif suffix == ".pdf":
            return self._extract_pdf(jd_path)
        else:
            return self._extract_txt(jd_path)

    def _extract_docx(self, path: Path) -> str:
        if not HAS_DOCX:
            logger.warning("python-docx not installed")
            return ""
        try:
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"docx extraction failed: {e}")
            return ""

    def _extract_pdf(self, path: Path) -> str:
        if not HAS_PDF:
            logger.warning("pdfplumber not installed")
            return ""
        try:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except Exception as e:
            logger.error(f"pdf extraction failed: {e}")
            return ""

    def _extract_txt(self, path: Path) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return ""

    # ------------------------------------------------------------------
    # Section segmentation
    # ------------------------------------------------------------------

    def _segment_sections(self, text: str) -> Dict[str, str]:
        """
        Split JD text into labeled sections.
        Returns dict of section_type -> section_text.
        """
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {
            "required": [], "preferred": [], "disqualifiers": [],
            "responsibilities": [], "company": [], "body": [],
        }
        current_section = "body"

        for line in lines:
            line_stripped = line.strip()
            matched_section = None

            for section_type, patterns in self.section_headers.items():
                for pattern in patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        matched_section = section_type
                        break
                if matched_section:
                    break

            if matched_section:
                current_section = matched_section
            else:
                sections[current_section].append(line)

        return {k: "\n".join(v) for k, v in sections.items()}

    # ------------------------------------------------------------------
    # Skill extraction
    # ------------------------------------------------------------------

    def _normalize_skill(self, skill: str) -> str:
        s = skill.lower().strip().rstrip(".,;:-")
        return self.skill_aliases.get(s, s)

    def _extract_skills_from_text(self, text: str) -> Set[str]:
        """
        Match skills from vocabulary against free text.
        Uses whole-word regex matching + alias normalization.
        """
        found = set()
        text_lower = text.lower()

        for skill in self.skill_vocab:
            skill_lower = skill.lower()
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.add(self._normalize_skill(skill_lower))

        # Also catch alias variants not in vocabulary
        for alias, canonical in self.skill_aliases.items():
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.add(canonical)

        return found

    def _collapse_to_canonical(self, skills: Set[str]) -> Set[str]:
        """
        Collapse specific tool names into canonical skill groups.
        Prevents elasticsearch + faiss + pinecone from counting as 3 required skills
        when they're all examples of the same capability: vector database.
        """
        canonical = set()
        for skill in skills:
            canonical.add(self.skill_to_canonical_group.get(skill, skill))
        return canonical

    def _extract_required_skills(
        self, full_text: str, sections: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract required and critical skills.
        Priority: required section > responsibilities (implied) > body.
        Collapses tool variants into canonical capability groups.
        Critical skills are from critical_skills.json filtered to what's found.
        """
        required: Set[str] = set()

        # Primary: explicit required section
        if sections.get("required"):
            required.update(self._extract_skills_from_text(sections["required"]))

        # Secondary: responsibilities section has implied required skills
        if sections.get("responsibilities"):
            implied = self._extract_skills_from_text(sections["responsibilities"])
            required.update(implied)

        # Fallback: if still empty, scan full text
        if not required:
            required.update(self._extract_skills_from_text(full_text))
            logger.warning("No required section found — scanned full text for skills")

        # Collapse tool variants into canonical capability groups
        required = self._collapse_to_canonical(required)

        required_list = sorted(required)

        # Critical = intersection of configured critical list with what's found
        # If none match, use top 4 from required
        critical = [s for s in self.critical_skills_for_role if s in required]
        if not critical:
            critical = required_list[:4]

        return required_list, critical

    def _extract_preferred_skills(
        self, full_text: str, sections: Dict[str, str]
    ) -> List[str]:
        preferred: Set[str] = set()
        if sections.get("preferred"):
            preferred.update(self._extract_skills_from_text(sections["preferred"]))
        preferred = self._collapse_to_canonical(preferred)
        return sorted(preferred)

    def _build_skill_weights(
        self,
        required: List[str],
        critical: List[str],
        preferred: List[str],
    ) -> Dict[str, float]:
        weights = {}
        for skill in critical:
            weights[skill] = 1.0
        for skill in required:
            if skill not in weights:
                weights[skill] = 0.7
        for skill in preferred:
            if skill not in weights:
                weights[skill] = 0.3
        return weights

    # Seniority

    def _extract_seniority(self, text: str) -> Tuple[str, List[int]]:
        text_lower = text.lower()

        # Year range
        year_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", text_lower)
        if year_match:
            seniority_range = [int(year_match.group(1)), int(year_match.group(2))]
        else:
            single_match = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", text_lower)
            min_yr = int(single_match.group(1)) if single_match else 5
            seniority_range = [min_yr, min_yr + 4]

        # Level from keywords — take highest match
        best_score = -1
        for keyword, score in self.seniority_map.items():
            if keyword.lower().replace("-", " ") in text_lower and score > best_score:
                best_score = score

        if best_score < 0:
            level = "mid"
        elif best_score <= 1:
            level = "junior"
        elif best_score <= 2:
            level = "mid"
        elif best_score <= 3:
            level = "senior"
        elif best_score <= 4:
            level = "lead"
        else:
            level = "principal"

        # Confidence boost: explicit title in JD text
        if re.search(r"senior\s+ai\s+engineer|senior\s+ml\s+engineer", text_lower):
            level = "senior"

        return level, seniority_range

    # Domain inference

    DOMAIN_CLUSTERS = {
        "ranking-and-retrieval": [
            "embeddings", "vector database", "ranking evaluation",
            "bm25", "hybrid search", "semantic search", "information retrieval",
            "approximate nearest neighbour", "learning to rank",
        ],
        "ml-engineering": [
            "pytorch", "tensorflow", "machine learning", "deep learning",
            "mlops", "feature engineering", "llm fine-tuning",
        ],
        "data-engineering": [
            "spark", "kafka", "airflow", "dbt", "etl", "pipeline",
        ],
        "backend-engineering": [
            "python", "java", "golang", "microservices", "distributed systems",
        ],
        "llm-ai": [
            "large language models", "generative ai", "rag", "llm framework",
        ],
    }

    def _infer_domain(self, required_skills: List[str]) -> str:
        scores = {domain: 0 for domain in self.DOMAIN_CLUSTERS}
        for skill in required_skills:
            for domain, domain_skills in self.DOMAIN_CLUSTERS.items():
                if skill in domain_skills:
                    scores[domain] += 1
        return max(scores, key=scores.get) if any(scores.values()) else "general"

    def _infer_sub_domains(self, required_skills: List[str]) -> List[str]:
        sub = []
        for domain, domain_skills in self.DOMAIN_CLUSTERS.items():
            overlap = sum(1 for s in required_skills if s in domain_skills)
            if overlap >= 1:
                sub.append(domain)
        return sub

    # Role archetype

    def _infer_role_archetype(self, text: str, implied_signals: List[str]) -> str:
        """
        Determine whether this role fundamentally wants a builder, researcher,
        operator, or generalist. Drives trajectory scoring.
        """
        text_lower = text.lower()

        builder_signals = [
            r"ship", r"deploy", r"production", r"scrappy",
            r"working.*suboptimal", r"learn from real user",
            r"tilt.*shipper", r"founding team",
        ]
        researcher_signals = [
            r"research", r"paper", r"publication", r"academic",
            r"novel", r"state.of.the.art",
        ]
        operator_signals = [
            r"operate", r"maintain", r"sre", r"reliability",
            r"on.call", r"incident",
        ]

        builder_count = sum(
            1 for p in builder_signals if re.search(p, text_lower)
        )
        researcher_count = sum(
            1 for p in researcher_signals if re.search(p, text_lower)
        )
        operator_count = sum(
            1 for p in operator_signals if re.search(p, text_lower)
        )

        if "scrappy-executor" in implied_signals:
            builder_count += 2
        if "pre-llm-era-ml-experience" in implied_signals:
            builder_count += 1

        counts = {
            "builder": builder_count,
            "researcher": researcher_count,
            "operator": operator_count,
        }
        winner = max(counts, key=counts.get)
        if counts[winner] == 0:
            return "generalist"
        return winner

    # Disqualifiers and penalties

    def _detect_disqualifiers(self, text: str) -> List[str]:
        found = []
        for disq_name, patterns in self.disqualifier_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        found.append(disq_name)
                        break
                except re.error:
                    logger.warning(f"Bad regex in {disq_name}: {pattern}")
        return found

    def _detect_soft_penalties(self, text: str) -> List[str]:
        penalties = []

        checks = [
            (r"framework.*enthusiast|langchain.*tutorial|hot\s+framework", "framework-enthusiast"),
            (r"poc|proof.of.concept|demo\s+only", "poc-only-experience"),
            (r"30\+\s+day.*notice|90\s+day.*notice|3\s+month.*notice", "long-notice-period"),
            (r"visa\s+sponsor", "requires-visa-sponsorship"),
        ]
        for pattern, label in checks:
            if re.search(pattern, text, re.IGNORECASE):
                penalties.append(label)

        return penalties

    # Production signals and experience type

    def _extract_production_signals(self, text: str) -> Dict:
        requires_shipped = bool(
            re.search(r"ship|deploy|production|real\s+user|real.world", text, re.IGNORECASE)
        )
        user_facing = bool(
            re.search(r"user.facing|end.user|recruiter|candidate", text, re.IGNORECASE)
        )

        scale_level = "meaningful"
        if re.search(r"large\s+scale|massive|millions", text, re.IGNORECASE):
            scale_level = "large"
        elif re.search(r"medium\s+scale", text, re.IGNORECASE):
            scale_level = "medium"
        elif re.search(r"small\s+scale|poc|prototype", text, re.IGNORECASE):
            scale_level = "small"

        code_recency_months = 18
        recency_match = re.search(
            r"(?:production\s+code|code).*?(\d+)\s+months?", text, re.IGNORECASE
        )
        if recency_match:
            code_recency_months = int(recency_match.group(1))

        return {
            "requires_shipped_system": requires_shipped,
            "scale_level": scale_level,
            "user_facing": user_facing,
            "code_recency_months": code_recency_months,
        }

    def _extract_experience_type(self, text: str) -> Dict:
        requires_product = bool(
            re.search(r"product\s+company|product.driven|not\s+a\s+services", text, re.IGNORECASE)
        )
        allows_consulting = not bool(
            re.search(
                r"consulting.only.*disqualif|services.only.*not|"
                r"tcs|infosys|wipro.*entire\s+career",
                text, re.IGNORECASE,
            )
        )
        startup_valued = bool(
            re.search(r"startup|early.stage|series\s+[ab]", text, re.IGNORECASE)
        )
        research_disqualifies = bool(
            re.search(
                r"pure\s+research.*not\s+move\s+forward|"
                r"research.only.*didn.t\s+work|academic.*disqualif",
                text, re.IGNORECASE,
            )
        )

        min_product_years = 4

        return {
            "requires_product_company": requires_product,
            "allows_consulting_only": allows_consulting,
            "startup_experience_valued": startup_valued,
            "research_only_disqualifies": research_disqualifies,
            "min_product_company_years": min_product_years,
        }

    # Implied and culture signals

    def _detect_implied_signals(self, text: str) -> List[str]:
        found = []
        for signal, patterns in self.implied_signals_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found.append(signal)
                    break
        return found

    def _detect_culture_signals(self, text: str) -> List[str]:
        found = []
        for signal, patterns in self.culture_signals_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found.append(signal)
                    break
        return found

    # Location, notice period, relocation

    LOCATION_PATTERNS = {
        "pune": [r"pune", r"maharashtra"],
        "noida": [r"noida", r"delhi\s+ncr", r"new\s+delhi"],
        "bangalore": [r"bangalore", r"bengaluru"],
        "hyderabad": [r"hyderabad", r"telangana"],
        "mumbai": [r"mumbai", r"bombay"],
        "delhi": [r"\bdelhi\b"],
    }

    def _extract_locations(self, text: str) -> List[str]:
        found = []
        for loc, patterns in self.LOCATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found.append(loc)
                    break
        return found if found else ["flexible"]

    def _extract_relocation(self, text: str) -> bool:
        return bool(re.search(r"relocat", text, re.IGNORECASE))

    def _extract_notice_period(self, text: str) -> str:
        if re.search(r"sub.?30|less\s+than\s+30", text, re.IGNORECASE):
            return "sub-30-days"
        if re.search(r"30.?60|1.?2\s+month", text, re.IGNORECASE):
            return "30-60-days"
        return "flexible"

    # Confidence scoring

    def _compute_confidence(
        self,
        required_skills: List[str],
        critical_skills: List[str],
        seniority_level: str,
        sections: Dict[str, str],
    ) -> Dict:
        has_required_section = bool(sections.get("required", "").strip())
        skills_found = len(required_skills)
        critical_found = len(critical_skills)

        if has_required_section and critical_found >= 3:
            skills_conf = 0.95
        elif has_required_section and skills_found >= 3:
            skills_conf = 0.80
        elif has_required_section and skills_found > 0:
            skills_conf = 0.65
        elif not has_required_section and skills_found >= 3:
            skills_conf = 0.50
        else:
            skills_conf = 0.25

        body_text = sections.get("body", "") + sections.get("required", "")
        has_explicit_title = bool(
            re.search(r"\bsenior\b|\blead\b|\bprincipal\b|\bstaff\b", body_text, re.IGNORECASE)
        )
        has_year_range = seniority_level != "mid"

        if has_year_range and has_explicit_title:
            seniority_conf = 0.90
        elif has_year_range or has_explicit_title:
            seniority_conf = 0.70
        else:
            seniority_conf = 0.45

        overall = round(skills_conf * 0.6 + seniority_conf * 0.4, 3)
        return {
            "required_skills_confidence": round(skills_conf, 3),
            "seniority_confidence": round(seniority_conf, 3),
            "overall_confidence": overall,
        }


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python jd_parser.py <path_to_jd>")
        sys.exit(1)

    parser = JDParser()
    result = parser.parse(sys.argv[1])
    print(json.dumps(result, indent=2, default=str))