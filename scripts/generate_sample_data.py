"""
Generate synthetic candidates.jsonl for testing the pipeline.
Creates 200 realistic candidate profiles with varied signals.
Run: python scripts/generate_sample_data.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "candidates.jsonl"

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------
FIRST_NAMES = ["Arjun","Priya","Rahul","Anjali","Vikram","Sneha","Rohan","Divya",
                "Karthik","Meera","Aditya","Nisha","Suresh","Pooja","Manish","Riya",
                "Nikhil","Shalini","Deepak","Kavya","Amit","Tanya","Sanjay","Isha",
                "Gaurav","Swati","Rajesh","Prerna","Vivek","Simran"]
LAST_NAMES  = ["Sharma","Patel","Kumar","Singh","Reddy","Gupta","Verma","Joshi",
                "Nair","Rao","Mehta","Iyer","Malhotra","Bose","Kapoor","Shah",
                "Pandey","Sinha","Das","Trivedi"]
TITLES_BY_LEVEL = {
    "junior": ["Junior ML Engineer","Software Engineer","Data Analyst","ML Associate"],
    "mid":    ["ML Engineer","Data Scientist","Software Engineer II","AI Engineer"],
    "senior": ["Senior ML Engineer","Senior Data Scientist","Senior AI Engineer","Lead ML Engineer"],
    "lead":   ["ML Tech Lead","Staff ML Engineer","Principal Data Scientist","Engineering Lead"],
}
COMPANIES_PRODUCT = ["Google","Meta","Amazon","Flipkart","Razorpay","Zepto","Meesho",
                      "CRED","Groww","PhonePe","Swiggy","Zomato","Postman","Freshworks",
                      "Chargebee","MoEngage","InMobi","Sigmoid","Darwinbox","Paytm"]
COMPANIES_CONSULTING = ["TCS","Infosys","Wipro","Cognizant","HCL","Tech Mahindra",
                         "Capgemini","Accenture","Mphasis","Mindtree"]
SKILL_POOL = {
    "core_ai": ["python","pytorch","tensorflow","scikit-learn","numpy","pandas"],
    "ranking": ["embeddings","faiss","vector database","bm25","ranking evaluation",
                "learning to rank","semantic search","ndcg","mrr"],
    "llm":     ["large language models","rag","fine-tuning","transformer","lora","peft"],
    "infra":   ["docker","kubernetes","aws","gcp","spark","kafka","airflow"],
    "other":   ["sql","git","java","golang","c++","microservices","distributed systems"],
}
LOCATIONS = ["Bangalore","Mumbai","Pune","Hyderabad","Delhi","Noida","Chennai","Remote"]
DEGREES   = ["Bachelor of Technology","Bachelor of Engineering","B.Sc. Computer Science",
              "M.Tech Computer Science","M.S. Machine Learning","MBA"]
INSTITUTIONS = ["IIT Bombay","IIT Delhi","IIT Madras","IIT Kharagpur","NIT Trichy",
                 "BITS Pilani","VIT","SRM University","Anna University","Pune University"]
PROFICIENCIES = ["beginner","intermediate","intermediate","expert","expert"]


def _rand_date(start_year: int, end_year: int) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    rand_days = random.randint(0, delta.days)
    return (start + timedelta(days=rand_days)).strftime("%Y-%m-%d")


def _make_candidate(idx: int) -> dict:
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    level = random.choices(
        ["junior","mid","senior","lead"],
        weights=[15, 35, 35, 15]
    )[0]
    title = random.choice(TITLES_BY_LEVEL[level])

    # Career timeline
    career_start_year = {
        "junior": random.randint(2021, 2023),
        "mid":    random.randint(2018, 2021),
        "senior": random.randint(2015, 2019),
        "lead":   random.randint(2012, 2016),
    }[level]

    # Education
    grad_year = career_start_year  # graduated same year career started
    edu_end_year = grad_year
    degree = random.choice(DEGREES)

    education = [{
        "degree": degree,
        "institution": random.choice(INSTITUTIONS),
        "start_date": f"{grad_year - 4}-06-01",
        "end_date": f"{edu_end_year}-06-01",
        "gpa": round(random.uniform(6.5, 9.5), 1),
    }]
    if "M." in degree or "MBA" in degree:
        education.append({
            "degree": "Bachelor of Technology",
            "institution": random.choice(INSTITUTIONS),
            "start_date": f"{grad_year - 7}-06-01",
            "end_date": f"{grad_year - 3}-06-01",
        })

    # Work history
    num_roles = {"junior": 1, "mid": 2, "senior": 3, "lead": 4}[level]
    consulting_heavy = random.random() < 0.25
    work_history = []
    current_year = 2025
    role_end = current_year
    seniority_order = {"junior": ["junior"], "mid": ["junior","mid"],
                       "senior": ["junior","mid","senior"],
                       "lead":   ["junior","mid","senior","lead"]}[level]

    for r_idx in range(num_roles - 1, -1, -1):
        role_level = seniority_order[min(r_idx, len(seniority_order)-1)]
        role_title = random.choice(TITLES_BY_LEVEL[role_level])
        duration_years = random.randint(1, 3)
        role_start = max(career_start_year, role_end - duration_years)

        if consulting_heavy:
            company = random.choice(COMPANIES_CONSULTING)
        elif r_idx == 0 and random.random() < 0.6:
            company = random.choice(COMPANIES_PRODUCT)
        else:
            company = random.choice(COMPANIES_PRODUCT + COMPANIES_CONSULTING)

        is_current = (r_idx == num_roles - 1)
        work_history.append({
            "company": company,
            "title": role_title,
            "start_date": f"{role_start}-01-01",
            "end_date": "present" if is_current else f"{role_end}-12-31",
            "description": (
                f"Built {random.choice(['ML pipelines','ranking systems','embedding models'])} "
                f"using {random.choice(['Python','PyTorch','TensorFlow'])} at scale. "
                f"Worked on {random.choice(['NLP','CV','ranking','recommendation'])} projects."
            ),
        })
        role_end = role_start

    work_history.reverse()

    # Skills
    num_core = random.randint(2, 5)
    num_ranking = random.randint(0, 4) if level in ("senior","lead") else random.randint(0, 2)
    num_llm = random.randint(0, 3)
    num_infra = random.randint(0, 3)
    num_other = random.randint(0, 3)

    selected_skills = (
        random.sample(SKILL_POOL["core_ai"], min(num_core, len(SKILL_POOL["core_ai"]))) +
        random.sample(SKILL_POOL["ranking"], min(num_ranking, len(SKILL_POOL["ranking"]))) +
        random.sample(SKILL_POOL["llm"],     min(num_llm,     len(SKILL_POOL["llm"]))) +
        random.sample(SKILL_POOL["infra"],   min(num_infra,   len(SKILL_POOL["infra"]))) +
        random.sample(SKILL_POOL["other"],   min(num_other,   len(SKILL_POOL["other"])))
    )
    selected_skills = list(dict.fromkeys(selected_skills))  # dedup preserve order

    skills = []
    for skill in selected_skills:
        years_used = max(1, random.randint(1, min(6, current_year - career_start_year)))
        skills.append({
            "name": skill,
            "proficiency": random.choice(PROFICIENCIES),
            "years_used": years_used,
        })

    # Certifications
    num_certs = random.choices([0, 1, 2, 3], weights=[50, 25, 15, 10])[0]
    certifications = []
    for _ in range(num_certs):
        cert_year = random.randint(career_start_year + 1, 2025)
        certifications.append({
            "name": random.choice(["AWS ML Specialty","GCP Professional ML","Azure AI","Coursera DL"]),
            "year": cert_year,
            "date": f"{cert_year}-{random.randint(1,12):02d}-01",
        })

    current_company = work_history[-1]["company"] if work_history else "Unknown"
    years_exp = current_year - career_start_year

    return {
        "candidate_id": f"cand_{idx:06d}",
        "full_name": name,
        "current_title": title,
        "current_company": current_company,
        "current_employer": current_company,
        "current_location": random.choice(LOCATIONS),
        "total_years_experience": years_exp + random.randint(-1, 1),  # slight noise
        "summary": (
            f"{level.capitalize()} AI/ML engineer with {years_exp} years of experience "
            f"in {random.choice(['building production ML systems','NLP and ranking systems','MLOps and deployment'])}."
        ),
        "skills": skills,
        "work_history": work_history,
        "education": education,
        "certifications": certifications,
        "platform_last_active_days": random.choices(
            [random.randint(1,30), random.randint(30,180), random.randint(180,400)],
            weights=[50, 30, 20]
        )[0],
        "recruiter_response_rate": round(random.uniform(0.1, 1.0), 2),
        "open_source": random.random() < 0.2,
        "github": f"github.com/{name.lower().replace(' ','.')}" if random.random() < 0.3 else None,
        "publications": random.random() < 0.1,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 200
    print(f"Generating {n} synthetic candidates -> {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i in range(n):
            candidate = _make_candidate(i)
            f.write(json.dumps(candidate) + "\n")
    print(f"Done. File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Run: python precompute/run_all.py --candidates {OUTPUT_PATH} --jd data/job_description.docx")


if __name__ == "__main__":
    main()
