# AI-Powered Career Guidance System

## Problem Statement

Many students are confused about choosing the right career path (higher studies, jobs, entrepreneurship, etc.). Traditional career counseling is manual, time-consuming, and biased.
A web app with AI integration can provide personalized, data-driven career suggestions.

## Objective

* Help students decide their career path
* Recommend courses, job roles, or higher study options based on:
  * Skills
  * Academic performance (CGPA)
  * Interests
  * Personality traits

## ⚙ System Architecture

```
[ User Input ]
     ↓
(Web Form: Skills, CGPA, Interests, Personality Test)
     ↓
[ Data Preprocessing Layer ]
   - Convert skills & interests into feature vectors
   - Normalize academic scores
     ↓
[ AI/ML Model ]
   - Classification model (Random Forest)
   - Suggests Career Path (e.g., Software Engineer, Data Scientist, MBA, Research)
     ↓
[ Recommendation Engine ]
   - Uses ML + NLP to suggest:
       - Online Courses (Coursera, Udemy APIs)
       - Job Roles (LinkedIn/Indeed APIs)
       - Higher Study programs
     ↓
[ Web Application Dashboard ]
   - Displays Recommended Career Path
   - Graphs of skill-match percentage
   - Suggested Learning Roadmap
```

## 🛠 Tech Stack

### Backend
* **Flask** → Python backend to integrate AI models
* **scikit-learn** → ML classification & recommendation
* **pandas, numpy** → Data processing
* **joblib** → Model serialization
* **REST API** → To communicate between frontend & ML model

### Frontend
* **React.js** → Modern, interactive UI
* **TailwindCSS** → Styling
* **Lucide React** → Icons

### AI/ML Components
* **Random Forest Classifier** → ML classification
* **Feature Engineering** → Skills and interests vectorization
* **Match Scoring** → Personalized career recommendations

## Workflow Example

1. **User Registration & Input**
   * Students fill form: Name, CGPA, Skills (Python, Java, IoT), Interests (AI, Web Dev)

2. **Data Processing**
   * Convert skills & interests into numeric vectors
   * Normalize CGPA/marks

3. **ML Prediction**
   * Model trained on dataset of students → career outcomes
   * Predicts best-fit career path (e.g., "Data Scientist – 85% match")

4. **Recommendation Engine**
   * Suggests:
     * Top 5 online courses (Coursera, Udemy)
     * Job roles & companies hiring
     * Higher study programs

5. **Result Dashboard**
   * Graph of career fit scores
   * Career Roadmap (e.g., Learn TensorFlow → Do Internship → Apply for AI roles)

## Example Output (For a Student)

* **Predicted Career Path**: Data Scientist (85% match)
* **Alternative Suggestions**: Software Engineer (75%), AI Researcher (70%)
* **Recommended Courses**:
  * "Machine Learning" – Coursera
  * "Deep Learning Specialization" – Andrew Ng
* **Suggested Companies**: TCS, Infosys, Google AI
* **Learning Roadmap**:
  * Learn TensorFlow → Kaggle Projects → Internship → Apply for DS roles

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd career-guidance-backend
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train_model.py
```

5. **Run the backend server**
```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd ../career-guidance-frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 📡 API Endpoints

### Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Get career recommendations |
| `/careers` | GET | List all available careers |
| `/career/<name>` | GET | Get specific career details |

### Example API Usage

**Get Career Recommendations:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "cgpa": 8.5,
    "skills": ["Python", "Machine Learning", "Data Analysis"],
    "interests": ["Artificial Intelligence", "Data Science"]
  }'
```

**Response:**
```json
{
  "career": "Data Scientist",
  "match_percentage": 85,
  "info": {
    "companies": [...],
    "salary": "₹10–18 LPA",
    "skills": [...],
    "courses": [...],
    "roadmap": [...]
  },
  "alternatives": [
    {
      "career": "ML Engineer",
      "match_percentage": 75
    }
  ]
}
```

## Features

### Completed Features
- **AI-Powered Analysis**: Advanced machine learning algorithms analyze user profile
- **Personalized Paths**: Career suggestions tailored to unique skills, interests, and goals
- **Learning Roadmap**: Curated course recommendations and step-by-step learning paths
- **Career Growth**: Opportunities in top companies and emerging industries
- **Multiple Career Profiles**: 9 comprehensive career paths with detailed information
- **Match Scoring**: Intelligent scoring based on skills, interests, and academic performance
- **Alternative Suggestions**: Multiple career options with match percentages
- **Comprehensive Data**: Courses, companies, salaries, and learning roadmaps

### Future Enhancements
- **Chatbot Integration**: Interactive career counseling
- **LinkedIn API Integration**: Real-time job opportunities
- **Resume Analysis**: AI-powered resume optimization
- **Personality Assessment**: MBTI integration for better matching
- **Industry Trends**: Real-time market analysis
- **Mentorship Network**: Connect with industry professionals

## Model Performance

- **Accuracy**: 99.1% on test dataset
- **Model Type**: Random Forest Classifier
- **Features**: 51 engineered features (skills, interests, CGPA)
- **Dataset**: 450 synthetic samples across 9 career paths
- **Cross-validation**: 5-fold CV with 98.5% average score

## Project Structure

```
career-guidance-backend/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── models/                # Trained models
│   ├── career_guidance_pipeline.joblib
│   ├── label_encoder.joblib
│   └── feature_names.joblib
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Why This Project Stands Out

**Real-world use case** (career guidance is in demand)  
**Combines Web + AI + Recommendation Engine**  
**Extendable** (you can add chatbot, LinkedIn scraping, or resume analysis later)  
**Looks amazing on resume/LinkedIn** (shows full-stack + ML integration)  
**Production-ready** with comprehensive error handling and validation  
**Scalable architecture** with modular design  

## Deployment

### Backend Deployment (Heroku)
```bash
echo "web: python app.py" > Procfile

# Deploy
git add .
git commit -m "Deploy backend"
git push heroku main
```

### Frontend Deployment (Netlify/Vercel)
```bash
npm run build
# Deploy build folder to your preferred platform
```

## Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with for students seeking their perfect career path!**
