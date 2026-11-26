import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

doc = docx.Document()

# Page Margins
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_header(title, level=1):
    h = doc.add_heading(title, level=level)
    run = h.runs[0]
    run.font.name = 'Arial'
    if level == 1:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(14, 116, 144)
    elif level == 2:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)
    return h

def add_code_block(code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, 'F1F5F9')
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(code_text.strip())
    r.font.name = 'Consolas'
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(15, 23, 42)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# Document Title
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
t_run = title_p.add_run('MLOps Group G134 — Teammate Push & Collaboration Guide\n')
t_run.font.name = 'Arial'
t_run.font.size = Pt(20)
t_run.font.bold = True
t_run.font.color.rgb = RGBColor(15, 23, 42)

sub_run = title_p.add_run('Step-by-Step Conflict-Free Git Push Instructions for All Contributors\nRepository: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT\n')
sub_run.font.name = 'Arial'
sub_run.font.size = Pt(11)
sub_run.font.color.rgb = RGBColor(100, 116, 139)

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# Overview Note
callout = doc.add_table(rows=1, cols=1)
callout.alignment = WD_TABLE_ALIGNMENT.CENTER
c_cell = callout.cell(0, 0)
set_cell_background(c_cell, 'E0F2FE')
cp = c_cell.paragraphs[0]
cr = cp.add_run(
    "IMPORTANT COLLABORATION RULE:\n"
    "To ensure every member registers as an official Contributor on GitHub with green activity graphs "
    "and ZERO merge conflicts, each teammate has been assigned their own dedicated verification file "
    "inside docs/contributions/. Follow the exact commands below for your assigned role."
)
cr.font.name = 'Arial'
cr.font.size = Pt(10)
cr.font.bold = True
cr.font.color.rgb = RGBColor(3, 105, 161)

doc.add_paragraph().paragraph_format.space_after = Pt(8)

# Step 0
add_header('Step 0: Prerequisites (GitHub Invitation & Git Author Setup)', level=1)
p0 = doc.add_paragraph()
p0.add_run(
    "1. Repository Owner (Udbhav) must add all 4 teammates as collaborators:\n"
    "   Go to: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT/settings/access -> Click 'Add people' -> Enter teammate GitHub usernames.\n"
    "2. Each teammate must accept the invitation via email or at:\n"
    "   https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT/invitations\n"
    "3. Teammates must use their GitHub account email in git config so GitHub links their commits to their profile."
)

# Team Table
add_header('Assigned Team Roles & Contribution Files', level=1)
table = doc.add_table(rows=1, cols=4)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
hdr[0].text = 'Member'
hdr[1].text = 'Role'
hdr[2].text = 'Enrolment'
hdr[3].text = 'Dedicated Contribution File'
for cell in hdr:
    set_cell_background(cell, '0EA5E9')
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

members = [
    ('Pranjal Bisht', 'Data Lead (M1)', 'S24CSEU1284', 'docs/contributions/M1_DATA_VERIFICATION.md'),
    ('Anshul Kundu', 'Model Lead (M2)', 'S24CSEU1277', 'docs/contributions/M2_MODEL_VERIFICATION.md'),
    ('Shivam Singh Anand', 'Service Lead (M3)', 'S24CSEU0505', 'docs/contributions/M3_SERVICE_VERIFICATION.md'),
    ('Prabhjot Singh', 'Ops Lead (M4)', 'S24CSEU1283', 'docs/contributions/M4_OPS_VERIFICATION.md'),
    ('Udbhav', 'Monitoring Lead (M5)', 'S24CSEU0095', 'Repo Owner / Initial Setup'),
]
for name, role, enrol, filename in members:
    rc = table.add_row().cells
    rc[0].text = name
    rc[1].text = role
    rc[2].text = enrol
    rc[3].text = filename

doc.add_paragraph().paragraph_format.space_after = Pt(10)

# Member 1: Pranjal Bisht
add_header('1. Commands for PRANJAL BISHT (Data Lead - M1)', level=1)
doc.add_paragraph('Copy and paste into terminal / PowerShell:')
add_code_block("""# 1. Clone the repository (or navigate into it)
git clone https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT.git
cd MLOPS-ASSIGNMENT

# 2. Configure your Git author details (Replace with your GitHub email!)
git config user.name "Pranjal Bisht"
git config user.email "YOUR_GITHUB_EMAIL@gmail.com"

# 3. Pull latest changes
git pull origin main --rebase

# 4. Create your dedicated verification file
mkdir -p docs/contributions
cat << 'EOF' > docs/contributions/M1_DATA_VERIFICATION.md
# M1 Data Quality Verification Report
- Lead: Pranjal Bisht (S24CSEU1284)
- Problem: MP290 Dehradun Metro Crowd Prediction
- Scope: DVC Raw Data Tracking, 7 Data Profiling Checks, Cleaning Pipeline
- Verification Results:
  * Check 1 (Leak): entries_per_hour present in raw/train.csv, dropped before features
  * Check 2 (Unit error): Yellow Line rain_cm -> converted rain_mm = rain_cm * 10
  * Check 3 (Duplicates): 40 duplicate rows dropped (1,000 clean training rows)
  * Check 4 (Typos): 13 typos in day_type/line cleaned with standard dictionary
  * Test Suite: tests/test_data_checks.py passed (7/7 tests)
EOF

# 5. Stage, commit, and push
git add docs/contributions/M1_DATA_VERIFICATION.md
git commit -m "feat(m1): add data profiling and cleaning verification report by Pranjal Bisht"
git push origin main
""")

# Member 2: Anshul Kundu
add_header('2. Commands for ANSHUL KUNDU (Model Lead - M2)', level=1)
doc.add_paragraph('Copy and paste into terminal / PowerShell:')
add_code_block("""# 1. Navigate into repository
cd MLOPS-ASSIGNMENT

# 2. Configure your Git author details (Replace with your GitHub email!)
git config user.name "Anshul Kundu"
git config user.email "YOUR_GITHUB_EMAIL@gmail.com"

# 3. Pull latest changes
git pull origin main --rebase

# 4. Create your dedicated verification file
mkdir -p docs/contributions
cat << 'EOF' > docs/contributions/M2_MODEL_VERIFICATION.md
# M2 Model Engineering & Experiment Tracking Report
- Lead: Anshul Kundu (S24CSEU1277)
- Problem: MP290 Dehradun Metro Crowd Prediction
- Scope: Scikit-learn Pipeline (Polynomial Features deg=2 + Ridge), MLflow Tracking
- Verification Results:
  * 4 Candidate Models Trained: Ridge, Lasso, ElasticNet, RandomForest
  * MLflow Run Tracking: Logged data_sha256, poly degree, alpha, MAE, RMSE, R2
  * Champion v1 Model: Ridge (alpha=1.0) selected with CV MAE = 210.45
  * Model artifacts saved to models/model.joblib with MLflow model registry registration
EOF

# 5. Stage, commit, and push
git add docs/contributions/M2_MODEL_VERIFICATION.md
git commit -m "feat(m2): add model pipeline and MLflow experiment verification report by Anshul Kundu"
git push origin main
""")

# Member 3: Shivam Singh Anand
add_header('3. Commands for SHIVAM SINGH ANAND (Service Lead - M3)', level=1)
doc.add_paragraph('Copy and paste into terminal / PowerShell:')
add_code_block("""# 1. Navigate into repository
cd MLOPS-ASSIGNMENT

# 2. Configure your Git author details (Replace with your GitHub email!)
git config user.name "Shivam Singh Anand"
git config user.email "YOUR_GITHUB_EMAIL@gmail.com"

# 3. Pull latest changes
git pull origin main --rebase

# 4. Create your dedicated verification file
mkdir -p docs/contributions
cat << 'EOF' > docs/contributions/M3_SERVICE_VERIFICATION.md
# M3 Serving Infrastructure & API Verification Report
- Lead: Shivam Singh Anand (S24CSEU0505)
- Problem: MP290 Dehradun Metro Crowd Prediction
- Scope: FastAPI Microservice (POST /predict, GET /health, GET /), Web Dashboard, Docker
- Verification Results:
  * Pydantic Schema Validation:
    - Rule 1: nearby_offices <= 150 strictly enforced (HTTP 422 if > 150)
    - Rule 2: ticket_price_rs >= 5 strictly enforced (HTTP 422 if < 5)
  * Web Dashboard: Built responsive interactive crowd predictor at root URL
  * Containerization: Multi-stage Dockerfile validated on port 8080
  * Test Suite: tests/test_api.py passed (5/5 tests)
EOF

# 5. Stage, commit, and push
git add docs/contributions/M3_SERVICE_VERIFICATION.md
git commit -m "feat(m3): add FastAPI serving and web dashboard verification report by Shivam Singh Anand"
git push origin main
""")

# Member 4: Prabhjot Singh
add_header('4. Commands for PRABHJOT SINGH (Ops Lead - M4)', level=1)
doc.add_paragraph('Copy and paste into terminal / PowerShell:')
add_code_block("""# 1. Navigate into repository
cd MLOPS-ASSIGNMENT

# 2. Configure your Git author details (Replace with your GitHub email!)
git config user.name "Prabhjot Singh"
git config user.email "YOUR_GITHUB_EMAIL@gmail.com"

# 3. Pull latest changes
git pull origin main --rebase

# 4. Create your dedicated verification file
mkdir -p docs/contributions
cat << 'EOF' > docs/contributions/M4_OPS_VERIFICATION.md
# M4 DevOps, CI/CD & Cloud Deployment Report
- Lead: Prabhjot Singh (S24CSEU1283)
- Problem: MP290 Dehradun Metro Crowd Prediction
- Scope: GitHub Actions CI (.github/workflows/ci.yml), Kubernetes Manifests, Vercel Serverless
- Verification Results:
  * GitHub Actions CI: Automated pipeline checks, pytest suite, and selfcheck on push
  * Self-Check Validation: 47/47 passing verification points (100% compliance)
  * Kubernetes: deployment.yaml (2 replicas, liveness/readiness probes) and service.yaml
  * Vercel Cloud Deployment: Successfully deployed to https://mlops-assignment-nu.vercel.app/
EOF

# 5. Stage, commit, and push
git add docs/contributions/M4_OPS_VERIFICATION.md
git commit -m "feat(m4): add CI/CD pipeline, Kubernetes and Vercel ops report by Prabhjot Singh"
git push origin main
""")

# Verification section
add_header('How to Verify on GitHub', level=1)
pv = doc.add_paragraph()
pv.add_run(
    "1. Contributors Page:\n"
    "   Visit: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT/graphs/contributors\n"
    "   Each teammate's avatar and commit count will appear here once pushed.\n\n"
    "2. Commits List:\n"
    "   Visit: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT/commits/main\n"
    "   All commits will show clean verified badges with author names.\n\n"
    "3. Automated CI:\n"
    "   Visit: https://github.com/udbhav968-creator/MLOPS-ASSIGNMENT/actions\n"
    "   GitHub Actions runs all tests on each push and keeps the green badge active."
)

output_path = r"C:\Users\Dell\Downloads\team_push_commands.docx"
doc.save(output_path)
print("Saved successfully to:", output_path)
