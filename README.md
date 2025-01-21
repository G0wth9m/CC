# Career Suggestion AI

This repository contains an advanced AI-driven system designed to provide personalized career suggestions based on a user's skills and educational background. It leverages machine learning and natural language processing to analyze user data and recommend optimal career paths.

## Features

- **Resume Analysis**: Extracts skills, education, and work experience from uploaded resumes.
- **AI-Powered Predictions**: Uses a trained neural network model to suggest career paths.
- **Dynamic Learning**: Incorporates user feedback to improve recommendations over time.
- **Database Integration**: Stores user data and feedback for model retraining and analysis.
- **Web Interface**: Interactive and user-friendly interface built with Flask for ease of use.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/career-suggestion-ai.git
   cd career-suggestion-ai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the following files:
   - `career_recommender.csv`: Initial dataset for model training.
   - `advanced_career_model.h5`: Pre-trained model (if available).

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000` to access the web interface.

## How It Works

1. **Data Preparation**:
   - The system uses a CSV file containing skills, degrees, and career mappings to train the model.
   - The data is preprocessed and split into training and testing sets.

2. **Model Training**:
   - A neural network is built using Keras with input features (skills and degrees) and output classes (career suggestions).
   - The model is trained and evaluated, achieving high accuracy for predictions.

3. **Prediction**:
   - Users input their skills and educational background via the web interface.
   - The system encodes the input and predicts the most suitable career using the trained model.

4. **Feedback Loop**:
   - Users can provide feedback on the suggestions.
   - The feedback is stored and used to retrain the model, improving future predictions.

## File Structure

```
career-suggestion-ai/
├── app.py                    # Flask application
├── advanced_career_model.h5  # Pre-trained model file
├── career_recommender.csv    # Initial dataset
├── templates/                # HTML templates for the web interface
├── static/                   # Static files (CSS, JS, images)
├── ai/                       # AI-related scripts and classes
│   ├── model.py              # Neural network implementation
│   ├── database.py           # Database management
├── uploads/                  # Directory for uploaded resumes
└── requirements.txt          # Python dependencies
```

## API Endpoints

- `POST /predict`:
  - Input: JSON containing `skills` and `degree`.
  - Output: Suggested career and user data.

- `POST /upload_resume`:
  - Input: File upload (PDF format).
  - Output: Extracted skills, education, and experience.

- `GET /`:
  - Renders the home page.

## Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Database**: SQLite
- **NLP**: SpaCy for text extraction and processing

## How to Contribute

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

- **GitHub**: [G0wth9m](https://github.com/G0wth9m)
- **LinkedIn**: [Gowtham Sudhakaran](https://www.linkedin.com/in/gowtham-sudhakaran-2a7213253)

Feel free to reach out for collaboration or any questions related to the project!

