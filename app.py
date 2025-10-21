# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Paths to saved model and encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODEL_DIR, 'career_guidance_pipeline.joblib')
encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Try to load real model/encoder; otherwise use mocks so server still runs
if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(model_path)
    le_interest = joblib.load(encoder_path)
    print(f"Loaded model from {model_path} and encoder from {encoder_path}")
else:
    print("Model files not found! Using mock model and encoder. Run train_model.py to create real models.")
    class MockModel:
        def predict(self, X): 
            # Always returns a sample career path
            return ["Software Engineer"]
    class MockEncoder:
        def transform(self, X): 
            # Returns dummy encoding (0)
            return [0]
    model = MockModel()
    le_interest = MockEncoder()

# Note: Removed unused mappings and ML/NLP imports to keep the backend minimal

# Comprehensive career profiles with courses, roadmaps, and detailed information
career_profiles = {
    "Data Scientist": {
        "companies": [
            {"name": "Google AI", "website": "https://careers.google.com"},
            {"name": "Microsoft Research", "website": "https://careers.microsoft.com"},
            {"name": "Amazon ML", "website": "https://www.amazon.jobs"},
            {"name": "TCS Innovation Labs", "website": "https://www.tcs.com/careers"},
            {"name": "Infosys", "website": "https://www.infosys.com/careers"}
        ],
        "salary": "₹10–18 LPA",
        "skills": ["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow", "Pandas"],
        "courses": [
            {"title": "Machine Learning Specialization", "platform": "Coursera", "instructor": "Andrew Ng"},
            {"title": "Deep Learning A-Z", "platform": "Udemy", "instructor": "Kirill Eremenko"},
            {"title": "Python for Data Science", "platform": "DataCamp", "instructor": "Various"}
        ],
        "roadmap": [
            "Learn Python fundamentals and data manipulation",
            "Master statistics and probability",
            "Complete 3-5 Kaggle projects",
            "Learn TensorFlow and PyTorch",
            "Apply for ML internships",
            "Build a strong portfolio",
            "Apply for Data Scientist roles"
        ]
    },
    "Software Engineer": {
        "companies": [
            {"name": "Google", "website": "https://careers.google.com"},
            {"name": "Microsoft", "website": "https://careers.microsoft.com"},
            {"name": "Amazon", "website": "https://www.amazon.jobs"},
            {"name": "Infosys", "website": "https://www.infosys.com/careers"},
            {"name": "TCS", "website": "https://www.tcs.com/careers"}
        ],
        "salary": "₹6–12 LPA",
        "skills": ["Java", "C++", "OOPs", "Data Structures", "Algorithms", "System Design"],
        "courses": [
            {"title": "CS50: Introduction to Computer Science", "platform": "edX", "instructor": "David J. Malan"},
            {"title": "System Design Primer", "platform": "GitHub", "instructor": "Community"},
            {"title": "Java Programming Masterclass", "platform": "Udemy", "instructor": "Tim Buchalka"}
        ],
        "roadmap": [
            "Master Data Structures and Algorithms",
            "Learn a programming language deeply",
            "Build full-stack projects",
            "Contribute to open-source projects",
            "Learn system design principles",
            "Practice coding interviews",
            "Apply for SDE roles"
        ]
    },
    "Web Developer": {
        "companies": [
            {"name": "Cognizant", "website": "https://careers.cognizant.com"},
            {"name": "Adobe", "website": "https://www.adobe.com/careers.html"},
            {"name": "Zoho", "website": "https://www.zoho.com/careers.html"},
            {"name": "Freshworks", "website": "https://www.freshworks.com/careers"},
            {"name": "Wipro", "website": "https://careers.wipro.com"}
        ],
        "salary": "₹4–9 LPA",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "MongoDB"],
        "courses": [
            {"title": "The Complete Web Developer Bootcamp", "platform": "Udemy", "instructor": "Colt Steele"},
            {"title": "React - The Complete Guide", "platform": "Udemy", "instructor": "Maximilian Schwarzmüller"},
            {"title": "Full Stack Web Development", "platform": "Coursera", "instructor": "JHU"}
        ],
        "roadmap": [
            "Learn HTML, CSS, and JavaScript fundamentals",
            "Master a frontend framework (React/Vue/Angular)",
            "Learn backend development (Node.js/Python)",
            "Understand databases (SQL/NoSQL)",
            "Build full-stack projects",
            "Learn deployment and DevOps basics",
            "Apply for web developer positions"
        ]
    },
    "ML Engineer": {
        "companies": [
            {"name": "NVIDIA", "website": "https://www.nvidia.com/en-in/about-nvidia/careers"},
            {"name": "Amazon", "website": "https://www.amazon.jobs"},
            {"name": "OpenAI", "website": "https://openai.com/careers"},
            {"name": "Tesla", "website": "https://www.tesla.com/careers"},
            {"name": "Uber", "website": "https://www.uber.com/careers"}
        ],
        "salary": "₹12–20 LPA",
        "skills": ["Python", "TensorFlow", "Deep Learning", "Data Analysis", "MLOps", "Docker"],
        "courses": [
            {"title": "Deep Learning Specialization", "platform": "Coursera", "instructor": "Andrew Ng"},
            {"title": "Machine Learning Engineering for Production", "platform": "Coursera", "instructor": "Andrew Ng"},
            {"title": "MLOps Fundamentals", "platform": "Coursera", "instructor": "Google Cloud"}
        ],
        "roadmap": [
            "Master machine learning fundamentals",
            "Learn deep learning frameworks (TensorFlow/PyTorch)",
            "Understand MLOps and model deployment",
            "Learn cloud platforms (AWS/GCP/Azure)",
            "Build end-to-end ML projects",
            "Learn containerization (Docker/Kubernetes)",
            "Apply for ML Engineer positions"
        ]
    },
    "Frontend Developer": {
        "companies": [
            {"name": "Adobe", "website": "https://www.adobe.com/careers.html"},
            {"name": "Figma", "website": "https://www.figma.com/careers"},
            {"name": "Canva", "website": "https://www.canva.com/careers"},
            {"name": "Shopify", "website": "https://www.shopify.com/careers"},
            {"name": "Stripe", "website": "https://stripe.com/jobs"}
        ],
        "salary": "₹5–10 LPA",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Vue.js", "TypeScript"],
        "courses": [
            {"title": "Advanced React and Redux", "platform": "Udemy", "instructor": "Stephen Grider"},
            {"title": "JavaScript: The Complete Guide", "platform": "Udemy", "instructor": "Maximilian Schwarzmüller"},
            {"title": "Frontend Web Development", "platform": "Coursera", "instructor": "Meta"}
        ],
        "roadmap": [
            "Master HTML, CSS, and JavaScript",
            "Learn a modern frontend framework",
            "Understand state management",
            "Learn testing frameworks",
            "Master responsive design",
            "Learn performance optimization",
            "Apply for frontend developer roles"
        ]
    },
    "Data Analyst": {
        "companies": [
            {"name": "Microsoft", "website": "https://careers.microsoft.com"},
            {"name": "Amazon", "website": "https://www.amazon.jobs"},
            {"name": "Flipkart", "website": "https://www.flipkartcareers.com"},
            {"name": "Swiggy", "website": "https://careers.swiggy.com"},
            {"name": "Zomato", "website": "https://www.zomato.com/careers"}
        ],
        "salary": "₹6–12 LPA",
        "skills": ["Python", "SQL", "Excel", "Tableau", "Power BI", "Statistics"],
        "courses": [
            {"title": "Google Data Analytics Certificate", "platform": "Coursera", "instructor": "Google"},
            {"title": "Tableau Desktop Specialist", "platform": "Udemy", "instructor": "Kyle Pew"},
            {"title": "SQL for Data Analysis", "platform": "Udemy", "instructor": "Jose Portilla"}
        ],
        "roadmap": [
            "Learn SQL and database fundamentals",
            "Master Excel and data manipulation",
            "Learn data visualization tools",
            "Understand statistical concepts",
            "Learn Python for data analysis",
            "Build analytical projects",
            "Apply for data analyst positions"
        ]
    },
    "AI Researcher": {
        "companies": [
            {"name": "OpenAI", "website": "https://openai.com/careers"},
            {"name": "DeepMind", "website": "https://deepmind.com/careers"},
            {"name": "Google Research", "website": "https://research.google/careers"},
            {"name": "Facebook AI Research", "website": "https://ai.facebook.com/careers"},
            {"name": "Microsoft Research", "website": "https://careers.microsoft.com"}
        ],
        "salary": "₹15–25 LPA",
        "skills": ["Python", "TensorFlow", "PyTorch", "Research", "Mathematics", "Statistics"],
        "courses": [
            {"title": "Deep Learning Specialization", "platform": "Coursera", "instructor": "Andrew Ng"},
            {"title": "CS229: Machine Learning", "platform": "Stanford", "instructor": "Andrew Ng"},
            {"title": "Advanced Deep Learning", "platform": "Coursera", "instructor": "DeepLearning.AI"}
        ],
        "roadmap": [
            "Master advanced mathematics and statistics",
            "Learn deep learning frameworks",
            "Read research papers regularly",
            "Contribute to open-source AI projects",
            "Publish research work",
            "Attend AI conferences and workshops",
            "Apply for research positions"
        ]
    },
    "Cybersecurity Analyst": {
        "companies": [
            {"name": "CrowdStrike", "website": "https://www.crowdstrike.com/careers"},
            {"name": "Palo Alto Networks", "website": "https://www.paloaltonetworks.com/careers"},
            {"name": "Cisco", "website": "https://www.cisco.com/c/en/us/about/careers"},
            {"name": "Wipro", "website": "https://careers.wipro.com"},
            {"name": "HCL", "website": "https://www.hcl.com/careers"}
        ],
        "salary": "₹8–16 LPA",
        "skills": ["Network Security", "Ethical Hacking", "SIEM", "Risk Assessment", "Python", "Linux"],
        "courses": [
            {"title": "Cybersecurity Specialization", "platform": "Coursera", "instructor": "University of Maryland"},
            {"title": "Practical Ethical Hacking", "platform": "Udemy", "instructor": "Heath Adams"},
            {"title": "Network Security", "platform": "edX", "instructor": "Various"}
        ],
        "roadmap": [
            "Learn networking fundamentals",
            "Study cybersecurity basics",
            "Practice on platforms like TryHackMe",
            "Get Security+ certification",
            "Learn incident response",
            "Build security projects",
            "Apply for SOC analyst roles"
        ]
    },
    "Business Analyst": {
        "companies": [
            {"name": "McKinsey", "website": "https://www.mckinsey.com/careers"},
            {"name": "BCG", "website": "https://www.bcg.com/careers"},
            {"name": "Deloitte", "website": "https://www2.deloitte.com/careers"},
            {"name": "Accenture", "website": "https://www.accenture.com/in-en/careers"},
            {"name": "EY", "website": "https://www.ey.com/en_in/careers"}
        ],
        "salary": "₹7–14 LPA",
        "skills": ["Business Analysis", "Data Analysis", "SQL", "Excel", "Power BI", "Communication"],
        "courses": [
            {"title": "Business Analysis Fundamentals", "platform": "Coursera", "instructor": "University of Virginia"},
            {"title": "SQL for Business Intelligence", "platform": "Udemy", "instructor": "Jose Portilla"},
            {"title": "Power BI Desktop", "platform": "Udemy", "instructor": "Maven Analytics"}
        ],
        "roadmap": [
            "Learn business analysis fundamentals",
            "Master data analysis tools",
            "Understand business processes",
            "Learn stakeholder management",
            "Build analytical projects",
            "Get business analysis certification",
            "Apply for business analyst roles"
        ]
    }
}

@app.route('/')
def home():
    return "AI Career Guidance Backend Running"

@app.route('/careers', methods=['GET'])
def get_careers():
    """Get all available career paths with basic info"""
    try:
        careers = []
        for career_name, career_info in career_profiles.items():
            careers.append({
                'name': career_name,
                'salary': career_info['salary'],
                'skills': career_info['skills'][:3],  # Top 3 skills
                'companies_count': len(career_info['companies'])
            })
        
        return jsonify({
            'careers': careers,
            'total': len(careers)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get careers: {str(e)}'}), 500

@app.route('/career/<career_name>', methods=['GET'])
def get_career_details(career_name):
    """Get detailed information about a specific career"""
    try:
        if career_name not in career_profiles:
            return jsonify({'error': 'Career not found'}), 404
        
        career_info = career_profiles[career_name]
        return jsonify({
            'career': career_name,
            'info': career_info
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get career details: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': os.path.exists(model_path),
            'encoder_loaded': os.path.exists(encoder_path),
            'careers_available': len(career_profiles)
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

def calculate_skill_match_score(user_skills, career_skills):
    """Calculate match score based on skills overlap"""
    if not user_skills or not career_skills:
        return 0
    
    user_skills_lower = [skill.lower() for skill in user_skills]
    career_skills_lower = [skill.lower() for skill in career_skills]
    
    matches = sum(1 for skill in user_skills_lower if skill in career_skills_lower)
    return (matches / len(career_skills_lower)) * 100

def calculate_interest_match_score(user_interests, career_name):
    """Calculate match score based on interests and career alignment"""
    if not user_interests:
        return 50  # Default score if no interests provided
    
    # Map interests to career categories
    interest_to_career_mapping = {
        'Artificial Intelligence': ['Data Scientist', 'ML Engineer', 'AI Researcher'],
        'Web Development': ['Web Developer', 'Frontend Developer', 'Software Engineer'],
        'Mobile Apps': ['Mobile Developer', 'Software Engineer'],
        'Data Science': ['Data Scientist', 'Data Analyst', 'ML Engineer'],
        'Cybersecurity': ['Cybersecurity Analyst'],
        'Entrepreneurship': ['Business Analyst', 'Software Engineer'],
        'Research': ['AI Researcher', 'Research Scientist', 'Data Scientist'],
        'Business': ['Business Analyst', 'Data Analyst']
    }
    
    # Check if any user interest aligns with the career
    for interest in user_interests:
        if interest in interest_to_career_mapping:
            if career_name in interest_to_career_mapping[interest]:
                return 90  # High match
    return 60  # Medium match

def normalize_cgpa(cgpa):
    """Normalize CGPA to 0-1 scale"""
    try:
        cgpa_val = float(cgpa)
        if cgpa_val <= 10:
            return cgpa_val / 10
        else:
            return min(cgpa_val / 100, 1)
    except:
        return 0.5  # Default normalized score

def get_career_recommendations(user_data):
    """Get comprehensive career recommendations based on user profile"""
    user_skills = user_data.get('skills', [])
    user_interests = user_data.get('interests', [])
    cgpa = user_data.get('cgpa', 0)
    
    # Normalize CGPA
    cgpa_normalized = normalize_cgpa(cgpa)
    
    # Calculate scores for all careers
    career_scores = []
    
    for career_name, career_info in career_profiles.items():
        # Calculate skill match (40% weight)
        skill_score = calculate_skill_match_score(user_skills, career_info['skills'])
        
        # Calculate interest match (30% weight)
        interest_score = calculate_interest_match_score(user_interests, career_name)
        
        # Calculate CGPA score (30% weight)
        cgpa_score = cgpa_normalized * 100
        
        # Weighted total score
        total_score = (skill_score * 0.4) + (interest_score * 0.3) + (cgpa_score * 0.3)
        
        career_scores.append({
            'career': career_name,
            'score': round(total_score),
            'skill_match': round(skill_score),
            'interest_match': round(interest_score),
            'cgpa_score': round(cgpa_score),
            'info': career_info
        })
    
    # Sort by score (highest first)
    career_scores.sort(key=lambda x: x['score'], reverse=True)
    
    return career_scores

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction endpoint that handles multiple skills and interests
    
    Expects JSON payload:
    {
      "name": "Alice",
      "cgpa": 8.3,
      "skills": ["Python", "Machine Learning"],
      "interests": ["Artificial Intelligence", "Data Science"]
    }

    Returns:
    {
      "career": "Data Scientist",
      "info": { "companies": [...], "salary": "...", "skills": [...], "courses": [...], "roadmap": [...] },
      "alternatives": [...],
      "match_percentage": 85
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload'}), 400

    try:
        # Extract user data
        name = data.get('name', 'Anonymous')
        cgpa = float(data.get('cgpa', 0))
        skills = data.get('skills', [])
        interests = data.get('interests', [])
        
        # If only single interest provided (backward compatibility)
        if not interests and 'interest' in data:
            interests = [data['interest']]
        
        # Validate input
        if not skills and not interests:
            return jsonify({'error': 'Please provide at least skills or interests'}), 400
        
        # Get career recommendations
        recommendations = get_career_recommendations({
            'skills': skills,
            'interests': interests,
            'cgpa': cgpa
        })
        
        if not recommendations:
            return jsonify({'error': 'No career recommendations found'}), 400
        
        # Primary recommendation
        primary = recommendations[0]
        
        # Alternative recommendations (top 2-3 excluding primary)
        alternatives = recommendations[1:4]  # Get next 3 alternatives
        
        # Prepare response
        response = {
            'career': primary['career'],
            'match_percentage': primary['score'],
            'info': {
                'companies': primary['info']['companies'],
                'salary': primary['info']['salary'],
                'skills': primary['info']['skills'],
                'courses': primary['info']['courses'],
                'roadmap': primary['info']['roadmap']
            },
            'alternatives': [
                {
                    'career': alt['career'],
                    'match_percentage': alt['score']
                } for alt in alternatives
            ],
            'analysis': {
                'skill_match': primary['skill_match'],
                'interest_match': primary['interest_match'],
                'cgpa_score': primary['cgpa_score']
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Use environment variables or defaults
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
    app.run(host='0.0.0.0', port=port, debug=debug)
