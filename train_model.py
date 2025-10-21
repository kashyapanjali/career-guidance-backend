# backend/train_model.py
"""
Enhanced training script for career guidance model.
Creates a more sophisticated model that can handle multiple skills and interests.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Enhanced dataset with more realistic career mappings
# This creates a more comprehensive dataset for better predictions
def create_enhanced_dataset():
    """Create a more comprehensive dataset for career guidance"""
    
    # Define skill sets for different career paths
    career_skill_mapping = {
        'Data Scientist': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'TensorFlow', 'Pandas', 'NumPy'],
        'Software Engineer': ['Java', 'C++', 'Python', 'JavaScript', 'Data Structures', 'Algorithms', 'OOPs'],
        'Web Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js', 'MongoDB', 'Express'],
        'ML Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'MLOps', 'Docker', 'Kubernetes'],
        'Frontend Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue.js', 'TypeScript', 'SASS'],
        'Data Analyst': ['Python', 'SQL', 'Excel', 'Tableau', 'Power BI', 'Statistics', 'R'],
        'AI Researcher': ['Python', 'TensorFlow', 'PyTorch', 'Research', 'Mathematics', 'Statistics', 'Papers'],
        'Cybersecurity Analyst': ['Network Security', 'Ethical Hacking', 'SIEM', 'Risk Assessment', 'Python', 'Linux'],
        'Business Analyst': ['Business Analysis', 'SQL', 'Excel', 'Power BI', 'Communication', 'Statistics', 'Process']
    }
    
    # Define interest to career mappings
    interest_career_mapping = {
        'Artificial Intelligence': ['Data Scientist', 'ML Engineer', 'AI Researcher'],
        'Web Development': ['Web Developer', 'Frontend Developer', 'Software Engineer'],
        'Mobile Apps': ['Mobile Developer', 'Software Engineer'],
        'Data Science': ['Data Scientist', 'Data Analyst', 'ML Engineer'],
        'Cybersecurity': ['Cybersecurity Analyst'],
        'Entrepreneurship': ['Business Analyst', 'Software Engineer'],
        'Research': ['AI Researcher', 'Research Scientist', 'Data Scientist'],
        'Business': ['Business Analyst', 'Data Analyst']
    }
    
    # CGPA ranges for different career types
    cgpa_ranges = {
        'Data Scientist': (8.0, 9.5),
        'Software Engineer': (7.0, 9.0),
        'Web Developer': (6.5, 8.5),
        'ML Engineer': (8.2, 9.5),
        'Frontend Developer': (6.5, 8.5),
        'Data Analyst': (7.0, 9.0),
        'AI Researcher': (8.5, 9.8),
        'Cybersecurity Analyst': (7.0, 8.8),
        'Business Analyst': (6.5, 8.5)
    }
    
    # Generate synthetic data
    data = []
    
    for career, skills in career_skill_mapping.items():
        # Generate 50 samples per career
        for _ in range(50):
            # Randomly select 3-5 skills from career skill set
            num_skills = np.random.randint(3, min(6, len(skills) + 1))
            selected_skills = np.random.choice(skills, num_skills, replace=False)
            
            # Find matching interests
            matching_interests = []
            for interest, careers in interest_career_mapping.items():
                if career in careers:
                    matching_interests.append(interest)
            
            # Select 1-2 interests
            if matching_interests:
                num_interests = np.random.randint(1, min(3, len(matching_interests) + 1))
                selected_interests = np.random.choice(matching_interests, num_interests, replace=False)
            else:
                selected_interests = ['Web Development']  # Default
            
            # Generate CGPA based on career type
            cgpa_min, cgpa_max = cgpa_ranges[career]
            cgpa = np.random.uniform(cgpa_min, cgpa_max)
            
            data.append({
                'skills': selected_skills,
                'interests': selected_interests,
                'cgpa': round(cgpa, 2),
                'career_path': career
            })
    
    return pd.DataFrame(data)

# Create the enhanced dataset
df = create_enhanced_dataset()

print(f"Created dataset with {len(df)} samples")
print(f"Career distribution:")
print(df['career_path'].value_counts())

# Feature engineering for skills and interests
def create_skill_features(df):
    """Create binary features for each skill"""
    all_skills = set()
    for skills_list in df['skills']:
        all_skills.update(skills_list)
    
    skill_features = {}
    for skill in all_skills:
        skill_features[f'skill_{skill.lower().replace(" ", "_")}'] = df['skills'].apply(
            lambda x: 1 if skill in x else 0
        )
    
    return pd.DataFrame(skill_features)

def create_interest_features(df):
    """Create binary features for each interest"""
    all_interests = set()
    for interests_list in df['interests']:
        all_interests.update(interests_list)
    
    interest_features = {}
    for interest in all_interests:
        interest_features[f'interest_{interest.lower().replace(" ", "_")}'] = df['interests'].apply(
            lambda x: 1 if interest in x else 0
        )
    
    return pd.DataFrame(interest_features)

# Create feature matrices
skill_features = create_skill_features(df)
interest_features = create_interest_features(df)

# Combine all features
X_features = pd.concat([
    df[['cgpa']],  # CGPA as continuous feature
    skill_features,  # Binary skill features
    interest_features  # Binary interest features
], axis=1)

y = df['career_path']

print(f"Feature matrix shape: {X_features.shape}")
print(f"Features: {list(X_features.columns)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.25, random_state=42)

# Train multiple models and select the best one
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = cv_scores.mean()
    
    print(f"{name} - CV Score: {mean_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with score: {best_score:.3f}")

# Train the best model
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (for RandomForest)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

# Create a simple encoder for backward compatibility (though we don't use it in the new system)
le_interest = LabelEncoder()
le_interest.fit(['Artificial Intelligence', 'Web Development', 'Mobile Apps', 'Data Science', 
                 'Cybersecurity', 'Entrepreneurship', 'Research', 'Business'])

# Save model and encoder
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, os.path.join(MODEL_DIR, 'career_guidance_pipeline.joblib'))
joblib.dump(le_interest, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

# Save feature names for later use
feature_names = list(X_features.columns)
joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.joblib'))

print("\n Enhanced model trained and saved to /models")
print(f"Model type: {best_name}")
print(f"Test accuracy: {test_accuracy:.3f}")
print("Label classes mapping:", dict(zip(le_interest.classes_, le_interest.transform(le_interest.classes_))))
