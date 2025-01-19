import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras
import joblib
from .career_database import CareerDatabase


class AdvancedCareerRecommendationAI:
    def __init__(self, model_file="advanced_career_model.h5", db_name="career_recommender.db"):
        self.model_file = model_file
        self.db_name = db_name
        self.model = None
        self.label_encoder = LabelEncoder()

        # Create or connect to the database
        self.db = CareerDatabase(db_name)

        # Load existing model if it exists
        if os.path.exists(self.model_file):
            self.model = keras.models.load_model(self.model_file)
        else:
            self.build_model()

    def prepare_data(self):
        """Fetches data from the database and prepares it"""
        data = self.db.fetch_all_data()

        if not data:
            raise ValueError("No data available in the database.")

        # Extract features and labels from the data
        skills = [item[1] for item in data]
        degrees = [item[2] for item in data]
        careers = [item[3] for item in data]

        # Fit the label encoder on skills and degrees
        self.label_encoder.fit(skills + degrees + careers)

        # Encode features and labels
        skills_encoded = self.label_encoder.transform(skills)
        degrees_encoded = self.label_encoder.transform(degrees)
        careers_encoded = self.label_encoder.transform(careers)

        # Features and labels
        X = np.column_stack((skills_encoded, degrees_encoded))
        y = careers_encoded

        return X, y

    def build_model(self):
        """Builds the deep learning model using Keras"""
        X, y = self.prepare_data()

        # Build the model
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=2, activation='relu'))  # 2 input features: skills and degree
        self.model.add(Dropout(0.2))  # Dropout for regularization
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(set(y)), activation='softmax'))  # Output layer (number of unique careers)

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train the model
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)

        # Save the model after training
        self.model.save(self.model_file)

    def train_model(self):
        """Retrains the model on available data"""
        X, y = self.prepare_data()

        # Train the model
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)

        # Save the model after training
        self.model.save(self.model_file)

    def predict(self, user_data):
        """Predicts the career based on user input"""
        skills, degree = user_data['skills'], user_data['degree']

        # Encode the user data
        skills_encoded = self.label_encoder.transform([skills])[0]
        degree_encoded = self.label_encoder.transform([degree])[0]

        # Predict the career path
        prediction = self.model.predict(np.array([[skills_encoded, degree_encoded]]))
        career_prediction = np.argmax(prediction)  # Get the index with the highest probability

        # Decode the prediction back into the original label
        career_path = self.label_encoder.inverse_transform([career_prediction])[0]

        return career_path

    def update_data(self, user_data, career_feedback):
        """Updates the database with new user data and retrains the model"""
        skills = user_data['skills']
        degree = user_data['degree']
        career = career_feedback

        # Insert new data into the database
        self.db.insert_data(skills, degree, career)

        # Retrain the model with updated data
        self.train_model()

    def learn_from_feedback(self, user_data, career_feedback):
        """Allows the AI to learn from user feedback"""
        # Update the database with user feedback
        self.update_data(user_data, career_feedback)

        # Retrain the model with updated data
        self.train_model()

    def close_connection(self):
        """Closes the database connection"""
        self.db.close_connection()


# Example usage:
if __name__ == "__main__":
    ai = AdvancedCareerRecommendationAI()

    # Train the model (initial training)
    ai.train_model()

    # Make a career suggestion based on user input
    user_data = {"skills": "Python", "degree": "Bachelors"}
    suggested_career = ai.predict(user_data)
    print(f"Suggested Career Path: {suggested_career}")

    # Simulate user feedback
    user_feedback = "Data Scientist"  # Assume the user agrees with the suggestion
    ai.learn_from_feedback(user_data, user_feedback)

    # Close the database connection
    ai.close_connection()
