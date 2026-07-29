"""
generate_dataset.py
Generates a realistic synthetic employee salary dataset for the project.

Salary is built from a base value per job role, then adjusted by:
  - experience (diminishing-returns growth curve)
  - education level multiplier
  - location cost-of-living multiplier
  - industry multiplier
  - random noise (to simulate real-world variance)

Run:
    python generate_dataset.py
Produces:
    data/salary_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 1500

job_roles = ["Software Engineer", "Data Scientist", "Analyst", "Manager", "Business Analyst", "DevOps Engineer"]
education_levels = ["High School", "Bachelors", "Masters", "PhD"]
locations = ["Bangalore", "Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Pune"]
industries = ["IT", "Finance", "Healthcare", "Education", "Retail", "Manufacturing"]

# Base annual salary (INR) per job role
base_salary = {
    "Software Engineer": 600000,
    "Data Scientist": 750000,
    "Analyst": 450000,
    "Manager": 900000,
    "Business Analyst": 500000,
    "DevOps Engineer": 650000,
}

# Multiplier per education level
education_multiplier = {
    "High School": 0.80,
    "Bachelors": 1.00,
    "Masters": 1.20,
    "PhD": 1.35,
}

# Cost-of-living / market multiplier per city
location_multiplier = {
    "Bangalore": 1.15,
    "Delhi": 1.10,
    "Mumbai": 1.20,
    "Kolkata": 0.90,
    "Chennai": 1.00,
    "Hyderabad": 1.05,
    "Pune": 1.02,
}

# Multiplier per industry
industry_multiplier = {
    "IT": 1.15,
    "Finance": 1.20,
    "Healthcare": 1.00,
    "Education": 0.80,
    "Retail": 0.85,
    "Manufacturing": 0.95,
}


def experience_growth(exp: int) -> float:
    """Diminishing-returns growth curve: fast growth early career, slower later."""
    return 1 + 0.09 * np.log1p(exp) * exp ** 0.35


rows = []
for _ in range(N_SAMPLES):
    experience = np.random.randint(0, 26)
    education = np.random.choice(education_levels, p=[0.10, 0.45, 0.35, 0.10])
    role = np.random.choice(job_roles)
    location = np.random.choice(locations)
    industry = np.random.choice(industries)

    salary = (
        base_salary[role]
        * experience_growth(experience)
        * education_multiplier[education]
        * location_multiplier[location]
        * industry_multiplier[industry]
    )

    # Add realistic noise (+/- ~8%)
    noise = np.random.normal(loc=1.0, scale=0.08)
    salary = max(150000, salary * noise)  # floor salary

    rows.append({
        "experience": experience,
        "education_level": education,
        "job_role": role,
        "location": location,
        "industry": industry,
        "salary": round(salary, -3),  # round to nearest 1000
    })

df = pd.DataFrame(rows)
df.to_csv("data/salary_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/salary_data.csv")
print(df.describe(include="all"))
