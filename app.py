from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for PDF extraction
import spacy
from ai.database.career_database import CareerDatabase  # Import the CareerDatabase class
from ai.ai_model import AdvancedCareerRecommendationAI  # Import your AI model

# Initialize Flask app
app = Flask(__name__, static_folder='templates/static')

# Ensure the "uploads" directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load NLP model (spaCy)
nlp = spacy.load("en_core_web_sm")

# Initialize the AI model and the database
ai_model = AdvancedCareerRecommendationAI()
db = CareerDatabase()  # Initialize the CareerDatabase

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to extract skills, education, and work experience
def extract_resume_info(text):
    doc = nlp(text)
    skills, education, experience = [], [], []

    skill_keywords = ["Python", "Java", "SQL", "Machine Learning", "Data Science", "Deep Learning"]
    education_keywords = ["Bachelor", "Master", "PhD", "BSc", "MSc", "MBA"]
    experience_keywords = ["Developer", "Engineer", "Manager", "Intern"]

    for ent in doc.ents:
        if any(edu in ent.text for edu in education_keywords) and ent.text not in education:
            education.append(ent.text)
        elif any(skill in ent.text for skill in skill_keywords) and ent.text not in skills:
            skills.append(ent.text)
        elif any(exp in ent.text for exp in experience_keywords) and ent.text not in experience:
            experience.append(ent.text)

    return {
        "skills": ", ".join(set(skills)),
        "education": ", ".join(set(education)),
        "experience": ", ".join(set(experience)),
    }

# Route for predicting career based on user input (skills, degree)
@app.route("/predict", methods=["POST"])
def predict_career():
    # Extract form data
    data = request.get_json()
    skills = data.get("skills")
    degree = data.get("degree")

    # Create an instance of the database class
    db = CareerDatabase()

    # Create the necessary table if it doesn't exist
    db.create_table()

    # Predict career path using AI model
    suggested_career = ai_model.predict({"skills": skills, "degree": degree})

    # Optionally save the data to the database
    db.insert_user_data(skills, degree, suggested_career)

    # Fetch all user data (just as an example)
    user_data = db.fetch_user_data()

    # Close the database connection once done
    db.close()

    # Respond with the career suggestion
    return jsonify({
        "suggested_career": suggested_career,
        "user_data": user_data,
    })

# Route for uploading resume and extracting information
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    try:
        file.save(file_path)
        extracted_text = extract_text_from_pdf(file_path)
        resume_info = extract_resume_info(extracted_text)
        return jsonify(resume_info)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Route for the home page (handling both resume upload and form submission)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Check for uploaded resume
            if "resume" in request.files:
                file = request.files["resume"]
                if file.filename != "":
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(file_path)

                    extracted_text = extract_text_from_pdf(file_path)
                    resume_info = extract_resume_info(extracted_text)

                    # Extract additional fields
                    interests = request.form.get("interests", "")
                    skills = resume_info["skills"] if resume_info["skills"] else request.form.get("skills", "")
                    degree = resume_info["education"] if resume_info["education"] else request.form.get("degree", "")
                    certifications = resume_info["experience"] if resume_info["experience"] else request.form.get("certifications", "")
                else:
                    raise ValueError("No file uploaded or file name is empty.")
            else:
                # Manual input handling
                interests = request.form.get("interests", "")
                skills = request.form.get("skills", "")
                degree = request.form.get("degree", "")
                certifications = request.form.get("certifications", "")

            working = request.form.get("working", "")
            specialization = request.form.get("specialization", "")
            percentage = request.form.get("percentage", "")

            user_data = {
                "skills": skills,
                "degree": degree,
            }

            # Predict career path using AI model
            suggested_career = ai_model.predict(user_data)
            
            # Optionally save the data to the database
            db.insert_data(skills, degree, suggested_career)

            # Learn from the feedback if provided
            user_feedback = request.form.get("feedback", "")
            if user_feedback:
                ai_model.learn_from_feedback(user_data, user_feedback)

            return render_template("home.html", user_query=user_data, generated_output=suggested_career)

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return render_template("home.html", user_query=None, generated_output=None)


if __name__ == "__main__":
    app.run(debug=True)
