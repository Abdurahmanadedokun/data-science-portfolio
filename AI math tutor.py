import streamlit as st
import openai
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart AI Math Tutor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📘 Smart AI Math Tutor</h1>", unsafe_allow_html=True)

# ---------------- API KEY ----------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Set it in Streamlit Secrets.")
    st.stop()

# ---------------- PROGRESS UTILITIES ----------------
PROGRESS_FILE = "progress.json"

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE) as f:
        return json.load(f)

def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_progress(user, topic, correct):
    data = load_progress()
    data.setdefault(user, {})
    data[user].setdefault(topic, {"attempts": 0, "correct": 0})
    data[user][topic]["attempts"] += 1
    if correct:
        data[user][topic]["correct"] += 1
    save_progress(data)

def ask_llm(messages, temp=0.3):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temp
    )
    return response.choices[0].message.content

# ---------------- SIDEBAR ----------------
with st.sidebar:
    user = st.text_input("Your Name", "student")
    mode = st.radio("Mode", ["Teach", "Quiz"])
    pdf = st.file_uploader("Upload PDF Notes (optional)", type="pdf")

# ---------------- PDF CONTEXT ----------------
context = ""
if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    context = "\n".join([d.page_content[:600] for d in docs[:3]])
    st.success("PDF loaded! Will be used as reference.")

# ---------------- TABS ----------------
tabs = st.tabs(["📖 Teach Mode", "📝 Quiz Mode", "📊 Progress"])

# ---------------- TEACH MODE ----------------
with tabs[0]:
    question = st.text_area("Ask a math question:", placeholder="e.g., Solve 2x2 matrix inverse")
    if st.button("Explain"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            messages = [
                {"role": "system", "content": f"""
You are a smart AI math tutor.
- Automatically identify the topic of the math question.
- Explain the solution step by step, numbering each step.
- Use PDF context if provided.
- Provide hints if the student struggles.
"""}, 
                {"role": "user", "content": f"Question: {question}\nContext: {context}"}
            ]
            answer = ask_llm(messages)
            st.markdown("<div style='background-color:#E6E6FA; padding:15px; border-radius:10px;'>"
                        f"<h3 style='color:#4B0082;'>Step-by-Step Explanation:</h3>{answer}</div>", unsafe_allow_html=True)

# ---------------- QUIZ MODE ----------------
with tabs[1]:
    num_questions = st.number_input("Number of quiz questions", min_value=1, max_value=10, value=3, step=1)

    if st.button("Generate Quizzes"):
        # Adaptive difficulty
        progress = load_progress()
        user_data = progress.get(user, {})
        attempted = sum([v["attempts"] for v in user_data.values()])
        if attempted < 3:
            level_desc = "Beginner"
        elif attempted < 6:
            level_desc = "Intermediate"
        else:
            level_desc = "Advanced"

        quiz_prompt = f"""
Generate {num_questions} math problems suitable for {level_desc} level.
- Include step-by-step solution and final answer.
- Automatically recognize topic.
- Provide hints for difficult steps.
- Return as a numbered list of separate problems.
"""
        quizzes = ask_llm([{"role": "system", "content": quiz_prompt}])
        st.session_state.quizzes = quizzes
        st.session_state.selected_quiz_index = 0  # default to first quiz

    if "quizzes" in st.session_state:
        # Split quizzes
        quiz_list = st.session_state.quizzes.split("\n\n")  # double line break separation
        st.markdown("<h3 style='color:#4B0082;'>Generated Quizzes:</h3>", unsafe_allow_html=True)
        for i, q in enumerate(quiz_list):
            st.markdown(f"**Question {i+1}:**")
            st.markdown(f"{q[:500]}...")  # preview first 500 chars
            if st.button(f"Select Question {i+1}", key=f"select_{i}"):
                st.session_state.selected_quiz_index = i

        # Show selected quiz
        selected_quiz = quiz_list[st.session_state.selected_quiz_index]
        st.markdown("<div style='background-color:#F0F8FF; padding:15px; border-radius:10px;'>"
                    f"<h4 style='color:#4B0082;'>Selected Quiz:</h4>{selected_quiz}</div>", unsafe_allow_html=True)

        # Student solution input
        student_steps = st.text_area("Enter your solution step by step:")

        if st.button("Submit Answer"):
            eval_prompt = f"""
You are a strict math examiner.
- Compare student's solution with the official solution.
- Check method, steps, final answer.
- Provide hints for mistakes without giving full solution.
- Give final verdict: Correct or Incorrect.

Problem & Solution:
{selected_quiz}

Student Solution:
{student_steps}
"""
            feedback = ask_llm([{"role": "system", "content": eval_prompt}], temp=0.1)
            st.markdown("<div style='background-color:#FFF0F5; padding:15px; border-radius:10px;'>"
                        "<h4 style='color:#4B0082;'>Feedback:</h4>"
                        f"{feedback}</div>", unsafe_allow_html=True)
            correct = "final verdict: correct" in feedback.lower()
            update_progress(user, "auto-topic", correct)

# ---------------- PROGRESS ----------------
with tabs[2]:
    st.markdown("<h3 style='color:#4B0082;'>Your Learning Progress:</h3>", unsafe_allow_html=True)
    progress = load_progress()
    if user in progress:
        st.json(progress[user])
    else:
        st.write("No progress recorded yet.")
