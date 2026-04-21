# backend/app/ai/domain_concepts.py

DOMAIN_CONCEPTS = {
    "legal": {
        "description": "Legal documents, contracts, policies, constitutional texts",
        "concept_hints": [
            "articles, clauses, sections, provisions, acts, amendments",
            "rights, duties, obligations, liabilities, penalties",
            "plaintiff, defendant, court, judgment, tribunal",
            "fundamental rights, constitutional, statutory, regulatory",
        ],
        "query_expansion_examples": [
            ("rights of citizens", ["fundamental rights", "constitutional rights", "article 14", "equality", "freedom", "protection"]),
            ("punishment", ["penalty", "fine", "imprisonment", "sentence", "liable", "offence"]),
            ("agreement", ["contract", "clause", "obligation", "party", "terms", "covenant"]),
        ]
    },

    "ecommerce": {
        "description": "Product catalogs, return policies, shipping, support FAQs",
        "concept_hints": [
            "products, pricing, SKU, inventory, catalog, stock",
            "shipping, delivery, courier, tracking, dispatch, logistics",
            "refund, return, exchange, replacement, cancellation",
            "payment, invoice, COD, EMI, discount, coupon, offer",
        ],
        "query_expansion_examples": [
            ("return", ["refund", "exchange", "replacement", "cancel", "money back", "policy"]),
            ("shipping", ["delivery", "dispatch", "courier", "tracking", "logistics", "days"]),
            ("payment", ["COD", "UPI", "EMI", "invoice", "price", "cost", "charge"]),
        ]
    },

    "education": {
        "description": "Academic reports, student records, course content, research",
        "concept_hints": [
            "courses, subjects, syllabus, curriculum, semester, credits",
            "internship, project, thesis, research, publication",
            "volunteer, ISR, social responsibility, community service, event",
            "games, tournament, sports, competition, cultural, technical fest",
        ],
        "query_expansion_examples": [
            ("games", ["tournament", "esports", "Valorant", "FIFA", "cricket", "sport", "competition", "match", "gaming"]),
            ("volunteering", ["volunteer", "community service", "ISR", "social work", "event management", "NGO"]),
            ("project", ["thesis", "dissertation", "research", "implementation", "development", "system"]),
            ("activity", ["event", "workshop", "seminar", "fest", "competition", "participation"]),
        ]
    },

    "hr": {
        "description": "Employee handbooks, HR policies, job descriptions, appraisals",
        "concept_hints": [
            "employee, staff, team, department, designation, role",
            "leave, attendance, WFH, remote, office, shift, timing",
            "salary, CTC, compensation, bonus, increment, appraisal",
            "policy, guidelines, code of conduct, compliance, rules",
        ],
        "query_expansion_examples": [
            ("salary", ["CTC", "compensation", "pay", "increment", "bonus", "appraisal", "package"]),
            ("leave", ["vacation", "PTO", "sick leave", "WFH", "work from home", "attendance"]),
            ("performance", ["appraisal", "KPI", "review", "rating", "evaluation", "target"]),
        ]
    },

    "general": {
        "description": "General purpose — no domain-specific expansion",
        "concept_hints": [],
        "query_expansion_examples": []
    }
}


def get_domain_hints(domain: str) -> str:
    """Returns a hint string to inject into the concept expansion prompt."""
    info = DOMAIN_CONCEPTS.get(domain, DOMAIN_CONCEPTS["general"])
    if not info["concept_hints"]:
        return ""
    hints = "\n".join(f"- {h}" for h in info["concept_hints"])
    return f"\nDomain context ({domain}):\n{hints}"


def get_domain_examples(domain: str) -> str:
    """Returns few-shot examples for this domain."""
    info = DOMAIN_CONCEPTS.get(domain, DOMAIN_CONCEPTS["general"])
    if not info["query_expansion_examples"]:
        return ""
    lines = []
    for query, expansions in info["query_expansion_examples"]:
        lines.append(f'"{query}" → {expansions}')
    return "\n".join(lines)