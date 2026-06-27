import pandas as pd
import json

def is_honeypot(row: pd.Series) -> bool:
    try:
        if float(row.get('profile', {}).get('years_of_experience', 0)) > 45:
            return True
    except:
        pass
        
    career = row.get('career_history', [])
    if isinstance(career, list):
        for job in career:
            start = job.get('start_date')
            end = job.get('end_date')
            if start and end and start > end:
                return True
    return False

def is_keyword_stuffer(row: pd.Series) -> bool:
    skills = row.get('skills', [])
    if isinstance(skills, list) and len(skills) > 40:
        return True
    return False

def is_unrelated_title(row: pd.Series) -> bool:
    title = row.get('profile', {}).get('current_title', '').lower()
    if not title:
        return False
        
    invalid_keywords = ['civil', 'mechanical', 'electrical', 'chemical', 'hr ', 'human resources', 'marketing', 'sales', 'accountant', 'designer', 'support', 'writer', 'operations']
    if any(k in title for k in invalid_keywords):
        return True
        
    valid_keywords = ['software', 'ml', 'ai', 'data', 'backend', 'machine learning', 'programmer', 'developer', 'scientist']
    if not any(k in title for k in valid_keywords):
        return True
        
    return False

def is_pure_consulting(row: pd.Series) -> bool:
    consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree'}
    career = row.get('career_history', [])
    if not isinstance(career, list) or len(career) == 0:
        return False
        
    for job in career:
        company = job.get('company', '').lower()
        is_consulting = any(cf in company for cf in consulting_firms)
        if not is_consulting:
            return False
    return True

def is_title_chaser(row: pd.Series) -> bool:
    career = row.get('career_history', [])
    if not isinstance(career, list) or len(career) <= 1:
        return False
        
    total_months = 0
    valid_jobs = 0
    for job in career:
        dur = job.get('duration_months')
        if dur:
            total_months += int(dur)
            valid_jobs += 1
            
    if valid_jobs > 1 and (total_months / valid_jobs) < 18:
        return True
    return False

def is_pure_research(row: pd.Series) -> bool:
    """Trap: Candidate has only worked in Research or Academia."""
    career = row.get('career_history', [])
    if not isinstance(career, list) or len(career) == 0:
        return False
    
    for job in career:
        ind = job.get('industry', '').lower()
        if 'research' not in ind and 'academia' not in ind and 'education' not in ind:
            return False
    return True

def is_langchain_enthusiast(row: pd.Series) -> bool:
    """Trap: Heavy LangChain use without older ML experience like PyTorch or NLP."""
    skills = row.get('skills', [])
    if not isinstance(skills, list):
        return False
        
    has_langchain = False
    has_core_ml = False
    
    core_ml_terms = ['pytorch', 'tensorflow', 'nlp', 'scikit', 'machine learning', 'deep learning']
    
    for s in skills:
        name = s.get('name', '').lower()
        if 'langchain' in name:
            has_langchain = True
        if any(c in name for c in core_ml_terms):
            has_core_ml = True
            
    if has_langchain and not has_core_ml:
        return True
    return False

def calculate_behavioral_score(row: pd.Series) -> float:
    signals = row.get('redrob_signals', {})
    if not isinstance(signals, dict):
        return 1.0
        
    multiplier = 1.0
    
    notice_period = signals.get('notice_period_days', 90)
    if notice_period <= 30:
        multiplier *= 1.1
    elif notice_period >= 90:
        multiplier *= 0.8
        
    loc = row.get('profile', {}).get('location', '').lower()
    good_locations = ['pune', 'noida', 'delhi', 'ncr', 'mumbai', 'hyderabad']
    willing_relocate = signals.get('willing_to_relocate', False)
    
    if any(l in loc for l in good_locations) or willing_relocate:
        multiplier *= 1.15
        
    if not signals.get('open_to_work_flag', True):
        multiplier *= 0.5
        
    response_rate = signals.get('recruiter_response_rate', 1.0)
    if response_rate < 0.2:
        multiplier *= 0.2
    elif response_rate < 0.5:
        multiplier *= 0.7
        
    gh = signals.get('github_activity_score', -1)
    if gh > 50:
        multiplier *= 1.2
        
    if signals.get('verified_email') and signals.get('verified_phone'):
        multiplier *= 1.05
        
    return min(multiplier, 2.0)
