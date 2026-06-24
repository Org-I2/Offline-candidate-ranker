# run_all.ps1 — Full setup and run for the Offline Candidate Ranker
# Usage: powershell -ExecutionPolicy Bypass -File run_all.ps1

Set-Location "C:\D drive content\Offline-candidate-ranker"
.\.venv\Scripts\Activate.ps1

Write-Host "`n[1/7] Installing Streamlit..." -ForegroundColor Cyan
pip install streamlit

Write-Host "`n[2/7] Downloading MiniLM model (skip if already done)..." -ForegroundColor Cyan
python -c "
from pathlib import Path
if Path('models/minilm').exists():
    print('  models/minilm already exists, skipping download.')
else:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('models/minilm')
    print('  Model saved to models/minilm')
"

Write-Host "`n[3/7] Generating sample candidates..." -ForegroundColor Cyan
python scripts/generate_sample_data.py

Write-Host "`n[4/7] Pre-computing artifacts..." -ForegroundColor Cyan
python precompute/run_all.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx

Write-Host "`n[5/7] Running ranker..." -ForegroundColor Cyan
python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./submission.csv

Write-Host "`n[6/7] Validating output..." -ForegroundColor Cyan
python validate_output.py --submission ./submission.csv --candidates ./data/candidates.jsonl

Write-Host "`n[7/7] Running tests..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "`nAll done! Output -> submission.csv" -ForegroundColor Green
