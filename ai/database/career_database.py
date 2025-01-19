import sqlite3  # Or any other DB connector you are using

class CareerDatabase:
    def __init__(self, db_name="career_recommender.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.db_name)  # Using SQLite as an example
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
    
    def create_table(self):
        """Create the necessary table(s) if they do not exist."""
        self.connect()
        try:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skills TEXT,
                degree TEXT,
                suggested_career TEXT
            )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
    
    def insert_user_data(self, skills, degree, suggested_career):
        """Insert user data into the database."""
        self.connect()
        try:
            self.cursor.execute('''
            INSERT INTO users (skills, degree, suggested_career) 
            VALUES (?, ?, ?)
            ''', (skills, degree, suggested_career))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
    
    def fetch_user_data(self):
        """Fetch all users' data from the database."""
        self.connect()
        try:
            self.cursor.execute('SELECT * FROM users')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
