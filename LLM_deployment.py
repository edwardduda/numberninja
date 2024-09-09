from flask import Flask, request, jsonify
from groq import Groq
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from flask_cors import CORS
from sqlalchemy import create_engine, text, MetaData, Table
from datasets import load_dataset, concatenate_datasets
import warnings

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "true"
app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
class EmbeddingVectorDatabase:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5', trust_remote_code=True)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.llm_client = None
        self.db_engine = None
        self.user_id = 2
        self.user_firstname = 'Edward'
        self.user_age = 17
        self.user_gradelvl = 12
        self.llm_id = 1
        self.session_id = None
        self.route = "/sign_up"
    
    def find_dataset(self):
        print("Starting find_dataset\n")
        competition_math = load_dataset("hendrycks/competition_math")
        competition_math = concatenate_datasets([competition_math['train'], competition_math['test']])

        dataframe = pd.DataFrame(competition_math)
        dataframe = dataframe.rename(columns={'problem': 'problem_text', 'type' : 'subject_name'})
        
        dataframe = dataframe.dropna()
        dataframe["problem_embedding"] = dataframe["problem_text"].apply(self.str_embedding)
        
        
        return dataframe

    def get_user_input_embedding(self, user_input):
        embedding = self.embedding_model.encode(user_input)
        embedding_as_str = [f"{x:.16f}" for x in embedding]
        return "[" + ", ".join(embedding_as_str) + "]"

    def str_embedding(self, message):
        embedding = self.embedding_model.encode(message)
        embedding_as_str = [f"{x:.16f}" for x in embedding]
        embedding_as_str  = "[" + ", ".join(embedding_as_str) + "]"
        return embedding_as_str
    
    def add_subjects_VDB(self, df):
        print("dropping duplicates")
        
        df = df.drop_duplicates('subject_name')
        print(df)
        
        with self.db_engine.connect() as connection:
            trans = connection.begin()
            print("Beginning transaction")
            try:
                # Prepare the SQL statement for inserting into the catalog table
                query = text("""
                INSERT IGNORE INTO subject(subject_name)
                VALUES (:subject_name)
                """)
            
                for _, row in df.iterrows():
                    connection.execute(query, {'subject_name': row['subject_name']})
                    trans.commit()
                    print("Transaction committed successfully")

            except Exception as e:
                if trans.is_active:
                    trans.rollback()
                print(f"An error occurred: {str(e)}")
        
    def dataset_to_VDB(self, df, chunk_size=100):
        print("Dropping duplicates")
        df = df.drop_duplicates(subset=['subject_name', 'problem_text'])  # Drop duplicates in one go
    
        with self.db_engine.connect() as connection:
            trans = connection.begin()
            print("Beginning transaction")
            try:
                # Prepare the SQL statement for inserting into the catalog table
                catalog_insert_query = text("""
                INSERT INTO catalog(subject_name, problem_text, problem_embedding, solution)
                VALUES(:subject_name, :problem_text, :problem_embedding, :solution)
                ON DUPLICATE KEY UPDATE
                problem_embedding = VALUES(problem_embedding),
                solution = VALUES(solution)
                """)
            
                # Insert into the catalog table in chunks
                for start in range(0, len(df), chunk_size):
                    chunk = df.iloc[start:start + chunk_size].to_dict(orient='records')
                    connection.execute(catalog_insert_query, chunk)
            
                trans.commit()
                print("Transaction committed successfully")

            except Exception as e:
                if trans.is_active:
                    trans.rollback()
                print(f"An error occurred: {str(e)}")


    def initialize_llm_client(self):
        try:
            self.llm_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        except Exception as e:
            print(f'Unable to connect to GROQ {e}')
            
    def initialize_db_connection(self):
        try:
            self.db_engine = create_engine(os.getenv('TIDB_DATABASE_URL'))
            
        except ConnectionError as e:
            print(f'Unable to connect to Database {e}')
            
    def get_session_id(self, user_id):
        with self.db_engine.connect() as connection:
            trans = connection.begin()
            query = text("""
                    INSERT INTO session(user_id)
                    VALUES (:user_id)
            ;""")
            connection.execute(query, {'user_id': user_id})
            trans.commit()
            result = connection.execute(text("SELECT LAST_INSERT_ID()"))
            self.session_id = result.scalar()
    
    def add_user(self):
        
        with self.db_engine.connect() as connection:
            trans = connection.begin()
            query = text("""
            INSERT INTO user(first_name, birth_year, grade_lvl)
            VALUES(:first_name, :age, :grade_lvl);
            """)
            
            connection.execute(query, {
                'first_name' : self.session_id,
                'age' : self.user_id,
                'grade_lvl' : self.message,
            })
            
            trans.commit()
    
    def insert_query(self, user_id, message):
        if(self.session_id == None):
            self.get_session_id(user_id)


        with self.db_engine.connect() as connection:
            trans = connection.begin()
            query = text("""
            INSERT INTO conversation(session_id, respondent_id, message)
            VALUES (:session_id, :respondent_id, :message)
            ;""")
            
            connection.execute(query, {
                'session_id' : self.session_id,
                'respondent_id' : user_id,
                'message' : message,
            })
            
            trans.commit()
    
    def input_message(self, user_input, context_df):
        response_params = f"user=u,u.first_name={self.user_firstname},u.age={self.user_age},u.grade_level={self.user_gradelvl},math_equation_format=Tex,confirm-each-step,response=math_tutor"
        context = ""
        for i in range(context_df.shape[0]):
            respondent = str(context_df.at[i, 'respondent_id']) + ".out="
            context += respondent + str(context_df.at[i, 'message']) + ","

        # Include the user's input and format the prompt properly
        prompt = f"{response_params}\n\nContext: {context}\n\nUser Input: {user_input}\n\nResponse:"
        return prompt

    def search_similar_problems(self, user_input_embedding, k=3, min_distance=0.2):
        with self.db_engine.connect() as connection:
            query = text(f"""
                SELECT problem_text, solution, Vec_Cosine_distance(problem_embedding, :user_input_embedding) AS distance
                FROM catalog
                WHERE Vec_Cosine_distance(problem_embedding, :user_input_embedding) < :min_distance
                ORDER BY distance ASC
                LIMIT :k;
            """)
    
            results = connection.execute(query, {'user_input_embedding': user_input_embedding, 'min_distance': min_distance, 'k': k})
            similar_problems = results.fetchall()

        # Print the distances and examples for debugging
        for i, row in enumerate(similar_problems):
            problem_text, solution, distance = row  # Unpack all three values here
            print(f"Example {i+1}:\nProblem: {problem_text}\nSolution: {solution}\nCosine Distance: {distance}\n")

        # Return only the problem_text and solution for each similar problem
        return [(row[0], row[1]) for row in similar_problems[:2]]

    def context_rag(self):
        with self.db_engine.connect() as connection:
            query = text("""
            SELECT e.respondent_id, e.message, e.created_at
            FROM user u
            JOIN session s ON u.user_id = s.user_id
            JOIN conversation e ON s.session_id = e.session_id
            WHERE s.session_id = :session_id AND u.user_id = :user_id
            ORDER BY e.created_at ASC
            """)
        
            params = {
                'user_id': self.user_id,
                'session_id': self.session_id
            }
        
            context_window = pd.read_sql(query, connection, params=params)
            print(context_window)
        
        return context_window
        
    def chat(self):
        try:
            data = request.get_json()
    
            if not data or 'message' not in data:
                return jsonify({'error': 'Invalid request. Message is required.'}), 400
    
            user_input = data['message']
    
            self.insert_query(self.user_id, user_input)
    
            # Get user input embedding
            user_input_embedding = self.get_user_input_embedding(user_input)
    
            # Search for similar problems
            similar_problems = self.search_similar_problems(user_input_embedding)
    
            # Build context with similar problems
            context_window = self.context_rag()
        
            examples = "\n\n".join([f"Example Problem: {problem}\nSolution: {solution}" for problem, solution in similar_problems])

            if context_window.empty:
                print("Warning: Context window is empty")
                input_prompt = user_input + "\n\n" + examples
            else:
                input_prompt = self.input_message(user_input, context_window) + "\n\n" + examples
    
            # Get response from LLM
            chat_completion = self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_prompt,
                    }
                ],
                model="llama-3.1-70b-versatile",
            )
            response = chat_completion.choices[0].message.content + " "
            self.insert_query(self.llm_id, response)
            self.context_rag()
            return jsonify({'response': response})

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An error occurred while processing your request.'}), 500
    
    def end_session(self):
        self.session_id = None
        
    def reflect(self):
        metadata = MetaData()
        
        table = Table('user', metadata, autoload_with=self.db_engine)
        print(table.columns.keys())


        
# Instantiate the class
evb = EmbeddingVectorDatabase()
#dataframe = evb.find_dataset()

evb.initialize_db_connection()

#evb.add_subjects_VDB(dataframe)
#evb.dataset_to_VDB(dataframe)
# Initialize clients

evb.initialize_llm_client()


#print(dataframe)
# Register Flask routes with the instance methods
@app.route('/chat', methods=['POST'])
def chat_route():
    return evb.chat()

if __name__ == '__main__':
    app.run(debug=True)