import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras

class AdvancedCareerRecommendationAI:
    def __init__(self, model_file="advanced_career_model.h5", data_file="/workspaces/CC/data/career_recommender.csv"):
        self.model_file = model_file
        self.data_file = data_file
        self.model = None
        self.label_encoder = LabelEncoder()

        # Load the dataset once
        self.df = pd.read_csv(self.data_file)

        # Load existing model if it exists
        if os.path.exists(self.model_file):
            self.model = keras.models.load_model(self.model_file)
        else:
            self.build_model()

        # Check if data file exists, otherwise create it
        if not os.path.exists(self.data_file):
            self.create_initial_data()

    def create_initial_data(self):
        """Creates initial dataset if it does not exist"""
        initial_data = {
            "skills": ["Python", "Machine Learning", "Data Science", "Java", "SQL", "Cloud Computing",
                       "R", "AI", "Web Development", "C++", "DevOps", "Big Data", "Cybersecurity",
                       "Blockchain", "Full Stack Development", "Artificial Intelligence", "JavaScript",
                       "Data Engineering", "IoT", "Statistics", "Software Development", "Business Intelligence",
                       "Computer Vision"],
            "degree": ["Bachelors", "Masters", "PhD", "Bachelors", "Masters", "Bachelors", "Bachelors",
                       "Masters", "Bachelors", "Bachelors", "Masters", "Masters", "Bachelors", "PhD",
                       "Bachelors", "PhD", "Bachelors", "Bachelors", "Masters", "Bachelors", "PhD", "Masters"],
            "career": ["Data Scientist", "ML Engineer", "Data Scientist", "Software Developer", "Data Analyst",
                       "Cloud Engineer", "Data Scientist", "AI Engineer", "Web Developer", "Software Engineer",
                       "DevOps Engineer", "Data Engineer", "Cybersecurity Analyst", "Blockchain Developer",
                       "Full Stack Developer", "AI Researcher", "Front End Developer", "Data Engineer", "IoT Developer",
                       "Data Analyst", "Software Architect", "BI Analyst", "AI Researcher"]
        }

        df = pd.DataFrame(initial_data)
        df.to_csv(self.data_file, index=False)

    def build_model(self):
        """Builds the deep learning model using Keras"""
        # Check for expected columns in self.df (already loaded)
        required_columns = ['Skills', 'Degree', 'Career']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"The following columns are missing: {', '.join(missing_columns)}")
        else:
            print("Columns are correctly present:", self.df.columns)

        # Fit the label encoder on the 'career' column
        self.label_encoder.fit(self.df['career'])

        # Build the model
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=2, activation='relu'))  # 2 input features: skills and degree
        self.model.add(Dropout(0.2))  # Dropout for regularization
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))  # Output layer

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def prepare_data(self):
        """Prepares the data by encoding skills and degree"""
        # Encode the skills and degree as features
        self.label_encoder.fit(self.df['skills'].append(self.df['degree']))  # Fit on both columns

        self.df['skills'] = self.label_encoder.transform(self.df['skills'])
        self.df['degree'] = self.label_encoder.transform(self.df['degree'])

        # Features and labels
        X = self.df[['skills', 'degree']]
        y = self.df['career']

        return X, y

    def train_model(self):
        """Trains the model on available data"""
        X, y = self.prepare_data()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # Evaluate the model
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Model Accuracy: {score[1] * 100:.2f}%")

        # Save the model after training
        self.model.save(self.model_file)

    def predict(self, user_data):
        """Predicts the career based on user input"""
        skills, degree = user_data['skills'], user_data['degree']

        # Convert user data into the format the model understands
        skills_encoded = self.label_encoder.transform([skills])[0]
        degree_encoded = self.label_encoder.transform([degree])[0]

        # Predict the career path
        prediction = self.model.predict(np.array([[skills_encoded, degree_encoded]]))
        career_prediction = np.argmax(prediction)  # Get the index with the highest probability

        # Decode the prediction back into the original label
        career_path = self.label_encoder.inverse_transform([career_prediction])[0]

        return career_path

    def update_data(self, user_data, career_feedback):
        """Updates the dataset with new user data and retrains the model"""
        new_data = {
            "skills": user_data['skills'],
            "degree": user_data['degree'],
            "career": career_feedback
        }

        df = pd.read_csv(self.data_file)
        new_row = pd.DataFrame(new_data, index=[len(df)])
        df = df.append(new_row, ignore_index=True)

        # Save updated data
        df.to_csv(self.data_file, index=False)

        # Retrain the model with updated data
        self.train_model()

    def learn_from_feedback(self, user_data, career_feedback):
        """Allows the AI to learn from user feedback"""
        # Update the data with user feedback
        self.update_data(user_data, career_feedback)

        # Re-train the model with updated data
        self.train_model()


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
