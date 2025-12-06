from flask import Flask, render_template, request, jsonify
import pickle
import re
import PyPDF2
import io

app = Flask(__name__)

# Load model and vectorizer
with open('resume_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Skill database
SKILL_DB = [
    "python", "java", "c", "c++", "html", "css", "javascript",
    "machine learning", "deep learning", "sql", "mongodb",
    "react", "node", "data analysis", "nlp", "ai", "ml",
    "tensorflow", "pytorch", "docker", "kubernetes", "aws",
    "azure", "git", "flask", "django", "fastapi", "pandas",
    "numpy", "scikit-learn", "data science", "analytics"
]

EXPERIENCE_KEYWORDS = [
    "developed", "managed", "led", "created", "designed",
    "implemented", "built", "architected", "optimized",
    "years of experience", "work experience", "internship"
]

PROJECT_KEYWORDS = [
    "project", "portfolio", "built", "created", "developed",
    "implemented", "designed", "application", "system", "website"
]

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return None

def extract_skills(text):
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in SKILL_DB:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return found_skills

def detect_experience(text):
    """Detect experience in text"""
    text_lower = text.lower()
    experience_count = 0
    
    for keyword in EXPERIENCE_KEYWORDS:
        if keyword in text_lower:
            experience_count += 1
    
    years_pattern = r'(\d+)\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)'
    years_match = re.search(years_pattern, text_lower)
    years = int(years_match.group(1)) if years_match else 0
    
    return {
        'experience_keywords': experience_count,
        'years': years
    }

def detect_projects(text):
    """Detect projects in text"""
    text_lower = text.lower()
    project_count = 0
    
    for keyword in PROJECT_KEYWORDS:
        if keyword in text_lower:
            project_count += 1
    
    return project_count

def calculate_fit_score(resume_text, job_description):
    """Calculate fit score"""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))
    
    if len(job_skills) > 0:
        skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
    else:
        skill_match = 0
    
    exp_data = detect_experience(resume_text)
    exp_score = min(exp_data['years'] / 10, 1.0) * 0.5 + min(exp_data['experience_keywords'] / 5, 1.0) * 0.5
    
    project_count = detect_projects(resume_text)
    project_score = min(project_count / 5, 1.0)
    
    fit_score = (skill_match * 0.4 + exp_score * 0.3 + project_score * 0.3) * 100
    
    return {
        'fit_score': round(fit_score, 2),
        'matched_skills': list(resume_skills.intersection(job_skills)),
        'total_skills': list(resume_skills),
        'experience_years': exp_data['years'],
        'project_count': project_count
    }

def analyze_resume(resume_text, job_description):
    """Analyze resume"""
    resume_vector = tfidf_vectorizer.transform([resume_text])
    category = model.predict(resume_vector)[0]
    category_prob = model.predict_proba(resume_vector)[0]
    
    fit_data = calculate_fit_score(resume_text, job_description)
    
    return {
        'category': category,
        'category_confidence': round(max(category_prob) * 100, 2),
        'fit_score': fit_data['fit_score'],
        'matched_skills': fit_data['matched_skills'],
        'total_skills': fit_data['total_skills'],
        'experience_years': fit_data['experience_years'],
        'project_count': fit_data['project_count']
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description', '')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        results = []
        
        # Handle multiple resume files
        files = request.files.getlist('resumes')
        
        for file in files:
            if file.filename == '':
                continue
            
            # Extract text from PDF or plain text
            if file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
                if not resume_text:
                    continue
            else:
                resume_text = file.read().decode('utf-8')
            
            # Analyze resume
            result = analyze_resume(resume_text, job_description)
            result['filename'] = file.filename
            results.append(result)
        
        # Rank by fit score
        results.sort(key=lambda x: x['fit_score'], reverse=True)
        
        # Add rank
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
