Command to test jd_parser:
python -c "
from src.jd_parser import JDParser
import json

parser = JDParser()
result = parser.parse('data/job_description.docx')
print(json.dumps(result, indent=2, default=str))
"
