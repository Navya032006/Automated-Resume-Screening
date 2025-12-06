# 🧠 Automated Resume Screening

An AI-powered web application that helps recruiters and HR teams **screen and rank multiple resumes automatically** based on a given job description.

The app uses:

- A **machine learning model** trained on resume text
- **TF-IDF vectorization** for feature extraction
- Rule-based logic for **skills, experience, and project analysis**
- A **Flask** backend and a simple web UI (HTML + JavaScript)

---

## 🚀 Features

- ✅ Upload **multiple resumes at once** (PDF or text)
- ✅ Paste a **job description** and analyze all resumes against it
- ✅ Extract and match **technical skills** from resumes
- ✅ Detect **years of experience** and experience-related phrases
- ✅ Count **projects** and project-related keywords
- ✅ Compute an overall **Fit Score (0–100%)** for each resume
- ✅ Classify resumes into a **category** (from your trained ML model)
- ✅ Return **ranked results** in JSON for display on the front-end

---

## 🧱 Tech Stack

- **Backend:** Flask (Python)
- **Model Training:** scikit-learn, TF-IDF, Logistic Regression (from `resume.ipynb`)
- **NLP:** basic regex, keyword matching, TF-IDF vectorizer
- **File Handling:** PyPDF2 (for PDF text extraction)
- **Front-end:** HTML templates (`templates/index.html`) + JavaScript
- **Serialization:** Pickle (`resume_classifier_model.pkl`, `tfidf_vectorizer.pkl`)

---

## 📁 Project Structure

```text
Automated-Resume-Screening/
│
├── app.py                     # Flask web application
├── resume.ipynb               # Notebook for training the classifier & TF-IDF
├── requirements.txt           # Python dependencies
├── resume_classifier_model.pkl  # Trained ML model
├── tfidf_vectorizer.pkl       # Trained TF-IDF vectorizer
└── templates/
    └── index.html             # Front-end UI

⚙️ Local Setup
1. Clone the Repository
git clone https://github.com/Navya032006/Automated-Resume-Screening.git
cd Automated-Resume-Screening

2. Create & Activate a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Dependencies

Make sure requirements.txt is in the project root, then run:

pip install -r requirements.txt

4. Run the Flask App
python app.py


By default, the app runs on:

http://0.0.0.0:8000


or

http://localhost:8000

💻 Using the Web App

Open the URL in your browser.

Paste the Job Description in the text box.

Upload one or more resumes:

.pdf (text-extractable PDFs)

.txt (plain text)

Click on the Analyze button.

The app:

Processes each resume

Shows category, confidence, fit score, matched skills, experience and project count

Ranks resumes from best to worst match based on fit score.
