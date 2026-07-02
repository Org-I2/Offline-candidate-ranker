# run_all.ps1 — Full setup and run for the Offline Candidate Ranker
# Usage: powershell -ExecutionPolicy Bypass -File run_all.ps1 -CandidatesFile "./data/candidates.jsonl"

param(
    [string]$CandidatesFile = "./data/candidates.jsonl",
    [string]$JdFile = "./data/job_description.docx"
)

Set-Location $PSScriptRoot

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

Write-Host "`n[1/8] Installing Streamlit..." -ForegroundColor Cyan
pip install streamlit

Write-Host "`n[2/8] Downloading MiniLM model (skip if already done)..." -ForegroundColor Cyan
python -c "
from pathlib import Path
if Path('models/minilm').exists():
    print('  models/minilm already exists, skipping download.')
else:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('models/minilm')
    print('  Model saved to models/minilm')
"

Write-Host "`n[3/8] Checking if CandidatesFile exists..." -ForegroundColor Cyan
if (-not (Test-Path $CandidatesFile)) {
    Write-Host "ERROR: Candidates file not found at $CandidatesFile. Please provide a valid file." -ForegroundColor Red
    exit 1
}

if ($CandidatesFile -match "\.normalized\.jsonl$") {
    $NormalizedFile = $CandidatesFile
    Write-Host "`n[4/8] Skipping normalization (file already normalized)..." -ForegroundColor Cyan
} else {
    $NormalizedFile = $CandidatesFile -replace "\.jsonl$", ".normalized.jsonl"
    if ($NormalizedFile -eq $CandidatesFile) {
        $NormalizedFile = "$CandidatesFile.normalized.jsonl"
    }
    Write-Host "`n[4/8] Normalizing candidates..." -ForegroundColor Cyan
    python scripts/normalize_candidates_for_local.py --input $CandidatesFile --output $NormalizedFile
}

Write-Host "`n[5/8] Pre-computing artifacts..." -ForegroundColor Cyan
python precompute/run_all.py --candidates $NormalizedFile --jd $JdFile

Write-Host "`n[6/8] Running ranker..." -ForegroundColor Cyan
python rank.py --candidates $NormalizedFile --jd $JdFile --out ./submission.csv

Write-Host "`n[7/8] Validating output..." -ForegroundColor Cyan
python validate_output.py --submission ./submission.csv --candidates $NormalizedFile

Write-Host "`n[8/8] Running tests..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "`nAll done! Output -> submission.csv" -ForegroundColor Green
