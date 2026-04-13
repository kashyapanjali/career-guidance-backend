# Career Guidance App — Backend

Flask-based REST API that powers the AI Career Guidance System. Uses a Random Forest ML model, resume parsing, roadmap.sh integration, and Gemini AI chatbot.

---

## Features

- **Career Prediction** — ML-based career recommendation using skills and CGPA
- **Resume Analysis** — Extracts skills, CGPA, and name from PDF/DOCX/TXT resumes
- **Career Roadmap** — Fetches interactive learning roadmaps from roadmap.sh
- **AI Chatbot** — Gemini-powered career counselor with fallback keyword responses
- **9 Career Profiles** — Data Scientist, Software Engineer, Web Developer, ML Engineer, Frontend Developer, Data Analyst, AI Researcher, Cybersecurity Analyst, Business Analyst

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Flask | Web framework |
| scikit-learn | ML model (Random Forest) |
| pandas / numpy | Data processing |
| joblib | Model serialization |
| PyPDF2 / python-docx | Resume text extraction |
| Google Generative AI | Gemini chatbot |
| python-dotenv | Environment variable management |

---

## Installation

```bash
# Go to the backend folder
cd career-guidance-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the ML model
python train_model.py

# Run the server
python app.py
```

The server runs at `http://localhost:5000`

---

## Environment Variables

Create a `.env` file in the backend root:

```
PORT=5000
FLASK_DEBUG=True
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Note:** If `GEMINI_API_KEY` is not set, the chatbot will use keyword-based fallback responses.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check (basic) |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Get career recommendations from form data |
| `/analyze-resume` | POST | Upload resume and get career recommendations |
| `/careers` | GET | List all available career paths |
| `/career/<name>` | GET | Get details of a specific career |
| `/roadmap/<career_name>` | GET | Fetch interactive roadmap for a career |
| `/roadmap/<career_name>/content/<node_id>` | GET | Fetch resources for a roadmap node |
| `/chat` | POST | AI chatbot (Gemini-powered) |

---

## API Usage Examples

### Predict Career (Form)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John",
    "cgpa": 8.5,
    "skills": ["Python", "Machine Learning", "Data Analysis"]
  }'
```

### Analyze Resume

```bash
curl -X POST http://localhost:5000/analyze-resume \
  -F "resume=@my_resume.pdf"
```

### Chat

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What skills do I need for data science?",
    "history": []
  }'
```

---

## Project Structure

```
career-guidance-backend/
├── app.py               # Flask app (all routes and logic)
├── train_model.py       # ML model training script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── models/              # Trained ML model files
│   ├── career_guidance_pipeline.joblib
│   ├── label_encoder.joblib
│   └── feature_names.joblib
├── test_api.py          # API test script
└── README.md
```

---

## ML Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Accuracy | ~99% on test dataset |
| Features | 51 engineered features (skills, interests, CGPA) |
| Dataset | 450 synthetic samples across 9 career paths |
| Cross-validation | 5-fold CV with ~98.5% average score |

---

## How It Works

1. User submits skills + CGPA (via form or resume upload)
2. Backend extracts and processes features
3. ML model predicts the best career match
4. Scoring engine ranks all 9 careers by weighted match (skills 40%, interests 30%, CGPA 30%)
5. Returns primary career, alternatives, courses, companies, and learning roadmap

---
