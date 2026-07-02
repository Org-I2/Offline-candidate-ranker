import json
import argparse
import sys
import os

def normalize_candidate(c):
    # Preserve the original fields, but we will add/modify keys in the dictionary itself
    normalized = c.copy()

    profile = c.get('profile', {})
    if profile:
        if 'anonymized_name' in profile:
            normalized['full_name'] = normalized.get('full_name', profile['anonymized_name'])
            normalized['name'] = normalized.get('name', profile['anonymized_name'])
        
        if 'current_title' in profile:
            normalized['current_title'] = normalized.get('current_title', profile['current_title'])
            
        if 'current_company' in profile:
            normalized['current_company'] = normalized.get('current_company', profile['current_company'])
            normalized['current_employer'] = normalized.get('current_employer', profile['current_company'])
            
        if 'location' in profile:
            normalized['current_location'] = normalized.get('current_location', profile['location'])
            normalized['location'] = normalized.get('location', profile['location'])
            
        if 'years_of_experience' in profile:
            normalized['total_years_experience'] = normalized.get('total_years_experience', profile['years_of_experience'])
            normalized['years_of_experience'] = normalized.get('years_of_experience', profile['years_of_experience'])
            
        if 'summary' in profile:
            normalized['summary'] = normalized.get('summary', profile['summary'])

    if 'career_history' in c:
        normalized['work_history'] = normalized.get('work_history', c['career_history'])

    signals = c.get('redrob_signals', {})
    if signals:
        if 'recruiter_response_rate' in signals:
            normalized['recruiter_response_rate'] = normalized.get('recruiter_response_rate', signals['recruiter_response_rate'])
        if 'last_active_date' in signals:
            normalized['platform_last_active_days'] = normalized.get('platform_last_active_days', signals['last_active_date'])
        if 'github_activity_score' in signals:
            normalized['github_activity_score'] = normalized.get('github_activity_score', signals['github_activity_score'])
        if 'notice_period_days' in signals:
            normalized['notice_period_days'] = normalized.get('notice_period_days', signals['notice_period_days'])
        if 'profile_completeness_score' in signals:
            normalized['profile_completeness_score'] = normalized.get('profile_completeness_score', signals['profile_completeness_score'])

    if 'skills' in c and isinstance(c['skills'], list):
        for s in normalized['skills']:
            if 'duration_months' in s:
                if 'years_used' not in s and 'years' not in s:
                    try:
                        s['years_used'] = float(s['duration_months']) / 12.0
                    except (ValueError, TypeError):
                        pass

    if 'education' in c and isinstance(c['education'], list):
        for e in normalized['education']:
            if 'end_year' in e:
                if 'graduation_year' not in e:
                    e['graduation_year'] = e['end_year']
                if 'end' not in e:
                    e['end'] = e['end_year']

    # candidate_id should naturally be preserved as we did a copy.

    return normalized

def main():
    parser = argparse.ArgumentParser(description="Normalize candidates for local CLI pipeline")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    
    candidates = []
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            # Check if it's a JSON array by trying to parse the whole file
            content = f.read().strip()
            if content.startswith('['):
                candidates = json.loads(content)
            else:
                # Process as JSONL
                for line in content.split('\n'):
                    if line.strip():
                        candidates.append(json.loads(line))
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
        
    normalized_candidates = [normalize_candidate(c) for c in candidates]
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for c in normalized_candidates:
                f.write(json.dumps(c) + '\n')
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully normalized {len(normalized_candidates)} candidates.")
    
if __name__ == "__main__":
    main()
