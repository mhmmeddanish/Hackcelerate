import streamlit as st
import numpy as np
import pandas as pd
import faiss
import random
import html
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_chat import message

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import networkx as nx
import plotly.graph_objects as go

# Title of the project
st.title("Student-Centric AI Tutor")

# Description
st.markdown("""
    Welcome to the Student-Centric AI Tutor. This application has four modes to assist students in different ways:
    - **Edhook**: Interactive educational content.
    - **EdPath**: Personalized learning paths.
    - **EdDocs**: Document-based question answering.
    - **Edvision**: Visual learning and recognition.
    - **EdMockText**: Online Mock Test Application.
""")

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state['mode'] = None

# Load data for EdPath
data = pd.read_csv(r'/Users/danish/Desktop/HackPrix/StudentCentricAITutor/coursera_courses.csv')

# TF-IDF vectorization
course_corpus = data['course_title']
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=5)
X = vectorizer.fit_transform(course_corpus)

# Convert sparse matrix to numpy array
X_array = np.float32(X.toarray())

# Create Faiss index
index = faiss.IndexFlatL2(X_array.shape[1])
index.add(X_array)

# for LLM gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyCvXu33gltO3ZEL5WRjqSyrl4ANgDeO84o"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
llm = ChatGoogleGenerativeAI(model="gemini-pro")
prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Imagine you are a course recommender, ask the candidates questions one by one to get his personal interests, end goals and current skill level and provide him with a curated list of courses alongside mentioning its difficulty level as beginner, intermediate and advanced. do not ask multiple questions at once. wait for the user to answer each question one by one and after 4-5 question provide him with a list of courses. suggest atleat one course from knowvationlearnings.in"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

# Functions for EdPath
def create_and_plot_graph(recommendations):
    G = nx.DiGraph()
    beginner_courses = [
        ("Data Science from Johns Hopkins University", "Fractal Data Science from Fractal Analytics"),
        ("Data Science from Johns Hopkins University", "What is Data Science? from IBM"),
        ("IBM Data Science from IBM", "SQL for Data Science from University of California, Davis"),
        ("IBM Data Science from IBM", "Data Science Math Skills from Duke University"),
        ("Tools for Data Science from IBM", "Practical Data Science with MATLAB from MathWorks")
    ]
    intermediate_courses = [
        ("Data Science with Databricks for Data Analysts from Databricks", "Genomic Data Science from Johns Hopkins University"),
        ("IBM Data Science from IBM", "Introduction to Data Science from IBM"),
        ("IBM Data Science from IBM", "Tools for Data Science from IBM"),
        ("IBM Data Science from IBM", "Applied Data Science from IBM")
    ]
    advanced_courses = [
        ("Genomic Data Science from Johns Hopkins University", "Foundations of Data Science from Google"),
        ("IBM Data Science from IBM", "Executive Data Science from Johns Hopkins University"),
        ("IBM Data Science from IBM", "Data Science Methodology from IBM")
    ]
    G.add_edges_from(beginner_courses)
    G.add_edges_from(intermediate_courses)
    G.add_edges_from(advanced_courses)
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    for node in G.nodes:
        x, y = pos[node]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=16, color="skyblue"),
            text=node,
            hoverinfo="text",
            name=node,
        ))
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color="gray", width=0.5),
            hoverinfo="none",
        ))
    fig.update_layout(
        title_text="Data Science Courses Flow Diagram",
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

def recommend_courses_by_difficulty(title):
    search_text = [title]
    search_text_vector = vectorizer.transform(search_text)
    search_text_vector_array = np.float32(search_text_vector.toarray())
    distances, indices = index.search(search_text_vector_array, 15)
    recommendations = {'Beginner': [], 'Intermediate': [], 'Advanced': []}
    for i in range(15):
        course_title = data['course_title'][indices[0][i]]
        organization = data['course_organization'][indices[0][i]]
        difficulty = data['course_difficulty'][indices[0][i]]
        link = data['course_url'][indices[0][i]]
        recommendation = f"[{course_title}]({link}) from {organization}\nDifficulty: {difficulty}"
        recommendations[difficulty].append(recommendation)
        if all(len(recommendations[level]) >= 3 for level in ['Beginner', 'Intermediate', 'Advanced']):
            break
    return recommendations

def langchain_conversation(user_input):
    try:
        response = conversation({"question": user_input})
        return response
    except Exception as e:
        print("An error occurred:", e)

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

# Function definitions
def edhook():
    import os

    def has_profanity(text):
        from better_profanity import Profanity
        return Profanity().contains_profanity(text)

    def filter_text(text):
        while has_profanity(text):
            text = input("Please provide an alternative prompt: ")
        return text

    def docquery1(uploaded_file):
        from docquery import document, pipeline
        p = pipeline('document-question-answering')
        st.write(os.getcwd(), uploaded_file)
        doc = document.load_document(uploaded_file)
        for q in ["What are the components of Cloud Computing Architecture?", "What is cloud reference model?"]:
            st.write(q, p(question=q, **doc.context))

    def hfgptinput(style, topic):
        import streamlit as st
        from hugchat import hugchat
        from hugchat.login import Login
        

        st.write("// Logging into HugChat")

        # ⚠️ WARNING: Never hardcode credentials in real-world apps
        email = 'mohmmeddanishh@gmail.com'
        passwd = 'Danishstu@1221'

        # Step 1: Login and get cookies
        sign = Login(email, passwd)
        cookies = sign.login()
        st.write("// Logged into HugChat")

        # Step 2: Create ChatBot with cookies
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

        # Step 3: Ask question with web search
        st.write("// Using web search to fetch real-time data")
        st.write("// Generating hook...")

        prompt = style + topic
        query_result = chatbot.chat(prompt)

        st.write("// Generated Hook")
        st.write(query_result.text)

        return query_result


    def video(prompt):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import export_to_video
        import torch
        import subprocess
        import datetime
        os.makedirs('/videos', exist_ok=True)

        st.write("// Image gen model begin.. model = damo-vilab/text-to-video-ms-1.7b")
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
        st.write("// Generating video for:", prompt)
        negative_prompt = "low quality"
        num_frames = 30
        video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_frames=num_frames).frames
        output_video_path = export_to_video(video_frames)

        new_video_path = f'/videos/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", output_video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "-2",
            new_video_path
        ]

        try:
            subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            st.write(f"Video conversion successful. Output saved to: {new_video_path}")
        except subprocess.CalledProcessError as e:
            st.write(f"Error converting video: {e}")

        st.write("Video generated:")
        video_file = open(new_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
    import torch
    import streamlit as st

    def images(prompt):
        model = "dreamlike-art/dreamlike-photoreal-2.0"
        st.write("// Image gen model begin.. model =", model)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model, scheduler=scheduler)
        pipe = pipe.to(device)

        st.write("// Generating images for:", prompt)

        num_images = 3
        filtered_input = filter_text(prompt)
        output = pipe(filtered_input, height=512, width=512, num_inference_steps=30, guidance_scale=9, num_images_per_prompt=num_images)

        st.image(output.images)


    def concept_explanation():
        st.subheader("Concept Explanation Using Hooks")
        topic = st.text_input("Enter the topic:")

        with st.sidebar:
            st.markdown("## Style")
            style_selected = st.radio("", ["Story", "Question", "Image", "Video", "Real Event", "Surprising Fact"])

            st.markdown("## Grade Level")
            grade_level_selected = st.radio("", ["Primary", "Secondary"])

            st.markdown("## Tone")
            tone_selected = st.radio("", ["Humorous", "Serious"])

        if style_selected == "Story":
            if st.button("Generate Story Hook") and topic:
                append = 'generate a story hook for the topic '
                generated_hook_hf = hfgptinput(append, topic)
                st.subheader("Generated Hook:")
                st.write(generated_hook_hf.text)

        if style_selected == "Surprising Fact":
            if st.button("Generate a Hook with surprising fact") and topic:
                append = 'generate a hook with surprising fact for the topic'
                generated_hook_hf = hfgptinput(append, topic)
                st.subheader("Generated Hook:")
                st.write(generated_hook_hf.text)

        if style_selected == "Real Event":
            if st.button("Generate a Hook with a real-world event") and topic:
                append = 'generate a hook referring to a current real world event for the topic'
                generated_hook_hf = hfgptinput(append, topic)
                st.subheader("Generated Hook:")
                st.write(generated_hook_hf.text)

        if style_selected == "Image":
            if st.button("Generate images") and topic:
                prompt = topic
                images(prompt)

        if style_selected == "Video":
            if st.button("Generate video") and topic:
                prompt = topic
                video(prompt)

        if style_selected == "Question":
            if st.button("Generate Question Hook") and topic:
                append = 'generate a question hook for the topic '
                generated_hook_hf = hfgptinput(append, topic)
                st.subheader("Generated Hook:")
                st.write(generated_hook_hf.text)

    concept_explanation()

# Define EdPath mode
def edpath():
    st.subheader("EdPath: Personalized Learning Path")
    st.markdown("""
        Welcome to EdPath. This mode allows you to create a personalized learning path based on your interests and progress.
    """)
    image = open('/Users/danish/Desktop/HackPrix/StudentCentricAITutor/diagram.jpg', 'rb').read()
    st.image(image, caption='The backend Architecture')
    st.subheader("Course Recommendation from data:")
    user_input = st.text_input("Search for a course:", "data science")
    if st.button("Search using FAISS vector Search"):
        recommended_courses = recommend_courses_by_difficulty(user_input)
        for difficulty, recommendations in recommended_courses.items():
            st.subheader(f"{difficulty} Courses:")
            for recommendation in recommendations:
                st.markdown(recommendation, unsafe_allow_html=True)
        st.title("Data Science Courses Flow Diagram")
        st.plotly_chart(create_and_plot_graph(recommended_courses))
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    user_input2 = st.text_input("ask gemini:", "i wanna be a data scientist")
    if user_input2 and st.button("Ask Gemini AI!"):
        output = langchain_conversation(user_input2)
        st.session_state.past.append(user_input2)
        st.session_state.generated.append(output["chat_history"][1].content)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def eddocs():
    import streamlit as st
    import os
    import numpy as np
    import re
    from collections import Counter
    import io

    try:
        import PyPDF2
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "PyPDF2"])
        import PyPDF2

    class SimpleVectorizer:
        def __init__(self):
            self.vocabulary = set()
            self.vocab_to_idx = {}
            self.idf = None

        def _tokenize(self, text):
            return re.findall(r'\w+', text.lower())

        def fit(self, text_documents):
            all_tokens = []
            for doc in text_documents:
                tokens = self._tokenize(doc)
                all_tokens.extend(tokens)
                self.vocabulary.update(tokens)

            self.vocabulary = sorted(self.vocabulary)
            self.vocab_to_idx = {word: i for i, word in enumerate(self.vocabulary)}

            doc_freq = Counter()
            for doc in text_documents:
                tokens = set(self._tokenize(doc))
                for token in tokens:
                    doc_freq[token] += 1

            N = len(text_documents)
            self.idf = np.zeros(len(self.vocabulary))
            for word, idx in self.vocab_to_idx.items():
                df = doc_freq.get(word, 0) + 1
                self.idf[idx] = np.log(N / df)

        def transform(self, documents):
            doc_vectors = []
            for doc in documents:
                tokens = self._tokenize(doc)
                term_freq = Counter(tokens)
                vector = np.zeros(len(self.vocabulary))
                for token, count in term_freq.items():
                    if token in self.vocab_to_idx:
                        idx = self.vocab_to_idx[token]
                        vector[idx] = count * self.idf[idx]
                doc_vectors.append(vector)
            return np.array(doc_vectors)

        def fit_transform(self, documents):
            self.fit(documents)
            return self.transform(documents)

    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    if "documents" not in st.session_state:
        st.session_state.documents = []
        st.session_state.document_sources = []
        st.session_state.vectorizer = None
        st.session_state.doc_vectors = None

    def process_pdf_buffer(file_buffer, filename="uploaded_document.pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file_buffer)
            page_texts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            chunks = []
            for text in page_texts:
                for para in text.split('\n\n'):
                    if len(para.strip()) > 50:
                        chunks.append(para.strip())
            for chunk in chunks:
                st.session_state.documents.append(chunk)
                st.session_state.document_sources.append(filename)
            return f"✅ Processed {filename} with {len(chunks)} chunks."
        except Exception as e:
            return f"❌ Error: {e}"

    def update_vectors():
        if not st.session_state.documents:
            return False
        st.session_state.vectorizer = SimpleVectorizer()
        st.session_state.doc_vectors = st.session_state.vectorizer.fit_transform(st.session_state.documents)
        return True

    def query_question(user_question, top_k=3):
        if not st.session_state.documents:
            return "No documents uploaded yet."
        if st.session_state.vectorizer is None or st.session_state.doc_vectors is None:
            if not update_vectors():
                return "Error updating vectors."

        query_vector = st.session_state.vectorizer.transform([user_question])[0]
        similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in st.session_state.doc_vectors]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_docs = [(st.session_state.documents[i], similarities[i], st.session_state.document_sources[i]) for i in top_indices]

        result = "### 🔍 Top Matching Chunks:\n"
        for i, (doc, score, source) in enumerate(top_docs):
            if score < 0.01:
                continue
            snippet = doc[:300] + "..." if len(doc) > 300 else doc
            result += f"**[{i+1}]** *From {source}* (score: `{score:.2f}`)\n\n```\n{snippet}\n```\n\n"
        return result.strip() or "No relevant information found."

    # Streamlit UI
    # st.set_page_config(page_title="PDF RAG Query", layout="centered")
    st.title("📄 RAG PDF Query System")

    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            msg = process_pdf_buffer(uploaded_file, uploaded_file.name)
            update_vectors()
            st.success(msg)

    st.markdown(f"**Total Text Chunks Stored:** `{len(st.session_state.documents)}`")

    question = st.text_input("💬 Ask a question based on uploaded PDFs")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = query_question(question)
            st.markdown(answer)

# Define EdVision mode
def edvision():
    st.write("Edvision mode is not implemented yet.")

# Define EdMockText mode
def edmocktext():
    st.title("Online Mock Test Application")

    # Initialize session state variables
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = []
    if 'selected_answers' not in st.session_state:
        st.session_state.selected_answers = []
    if 'time_left' not in st.session_state:
        st.session_state.time_left = 0
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0

    def fetch_questions(topic, num_questions):
        topic_mapping = {
            "jee": 17,  # Science & Nature
            "neet": 17,  # Science & Nature
            "civil services": 23  # History
        }

        if topic not in topic_mapping:
            st.warning("Invalid topic.")
            return

        category = topic_mapping[topic]

        url = f"https://opentdb.com/api.php?amount={num_questions}&category={category}&type=multiple"
        response = requests.get(url)
        if response.status_code != 200:
            st.error("Failed to fetch questions from the internet.")
            return

        data = response.json()
        if data['response_code'] != 0:
            st.error("No questions available for the selected topic.")
            return

        st.session_state.questions = data['results']
        st.session_state.total_questions = len(st.session_state.questions)
        for question in st.session_state.questions:
            question['year'] = random.randint(2000, 2023)
        st.session_state.correct_answers = [html.unescape(q['correct_answer']) for q in st.session_state.questions]
        display_question()

    def display_question():
        if st.session_state.question_index < len(st.session_state.questions):
            question_data = st.session_state.questions[st.session_state.question_index]
            question_number = st.session_state.question_index + 1
            st.write(f"Q{question_number}: ({question_data['year']}) {html.unescape(question_data['question'])}")

            correct_answer = html.unescape(question_data['correct_answer'])
            options = [html.unescape(opt) for opt in question_data['incorrect_answers']] + [correct_answer]
            random.shuffle(options)

            selected_option = st.radio("Select an option", options, key=f"option_{st.session_state.question_index}")

            if selected_option:
                st.session_state.selected_answers.append(selected_option)

            st.button("Previous", on_click=previous_question)
            if st.session_state.question_index < len(st.session_state.questions) - 1:
                st.button("Next", on_click=next_question)
            else:
                st.button("Submit", on_click=submit_test)

        else:
            show_result_window()

    def previous_question():
        if st.session_state.question_index > 0:
            st.session_state.question_index -= 1
            display_question()

    def next_question():
        st.session_state.question_index += 1
        display_question
    def submit_test():
        show_result_window()

    def show_result_window():
        st.write(f"Your Score: {st.session_state.score}/{len(st.session_state.questions)}\n")

        for idx, question in enumerate(st.session_state.questions):
            st.write(f"Q{idx + 1}: ({question['year']}) {html.unescape(question['question'])}")
            st.write(f"A: {html.unescape(st.session_state.correct_answers[idx])}")

            if idx < len(st.session_state.selected_answers):
                if st.session_state.correct_answers[idx] == st.session_state.selected_answers[idx]:
                    st.success("Correct")
                    st.session_state.score += 1
                else:
                    st.error("Incorrect")
            else:
                st.warning("No answer selected")

        suggestion_text = get_suggestion()
        st.write(f"Suggestion: {suggestion_text}")

        reset_mock_test()

    def get_suggestion():
        if st.session_state.score > 7:
            return "Excellent!"
        elif st.session_state.score > 4:
            return "Good job! Keep practicing."
        else:
            return "You need more practice."

    def reset_mock_test():
        st.session_state.score = 0
        st.session_state.question_index = 0
        st.session_state.questions = []
        st.session_state.correct_answers = []
        st.session_state.selected_answers = []
        st.session_state.time_left = 0
        st.session_state.total_questions = 0

    topic = st.selectbox("Select Subject:", ["SELECT", "JEE", "NEET", "CIVIL SERVICES"])
    num_questions = st.number_input("Number of Questions:", min_value=1, max_value=50, value=5, step=1)

    if st.button("Start Mock Test"):
        if topic == "SELECT":
            st.warning("Please select a topic.")
        else:
            fetch_questions(topic.lower(), num_questions)

# Function to load the respective mode
def load_mode():
    mode_name = st.session_state['mode']
    if mode_name == "Edhook":
        st.success("Loading Edhook...")
        edhook()
    elif mode_name == "EdPath":
        st.success("Loading EdPath...")
        edpath()
    elif mode_name == "EdDocs":
        st.success("Loading EdDocs...")
        eddocs()
    elif mode_name == "Edvision":
        st.success("Loading Edvision...")
        edvision()
    elif mode_name == "EdMockText":
        st.success("Loading EdMockText...")
        edmocktext()

# Buttons to select mode
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Edhook"):
        st.session_state['mode'] = "Edhook"
        st.experimental_rerun()
with col2:
    if st.button("EdPath"):
        st.session_state['mode'] = "EdPath"
        st.experimental_rerun()
with col3:
    if st.button("EdDocs"):
        st.session_state['mode'] = "EdDocs"
        st.experimental_rerun()
with col4:
    if st.button("Edvision"):
        st.session_state['mode'] = "Edvision"
        st.experimental_rerun()
with col5:
    if st.button("EdMockText"):
        st.session_state['mode'] = "EdMockText"
        st.experimental_rerun()

# Load the selected mode
if st.session_state['mode']:
    load_mode()    