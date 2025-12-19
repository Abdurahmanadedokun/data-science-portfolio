import streamlit as st
import openai
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Math Tutor", layout="centered")
st.title("📘 AI Math Tutor (Advanced & Accurate)")

openai.api_key = os.getenv("OPENAI_API_KEY")
PROGRESS_FILE = "progress.json"

# ---------------- UTILITIES ----------------
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
user = st.sidebar.text_input("Student name", "student")
level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
mode = st.sidebar.radio("Mode", ["Teach", "Quiz"])
topic = st.sidebar.text_input("Topic", "Matrices")

# ---------------- PDF (RAG) ----------------
context = ""
pdf = st.sidebar.file_uploader("Upload Math PDF (optional)", type="pdf")

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    context = "\n".join([d.page_content[:600] for d in docs[:3]])
    st.sidebar.success("PDF loaded and will be used for answers")

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = f"""
You are an expert mathematics tutor.

STRICT RULES:
- Accuracy is more important than speed
- Always reason step by step
- Verify results before concluding
- Adapt explanation to {level} level
- Use correct formulas and notation
- If PDF context is provided, rely on it
"""

# ---------------- TEACH MODE ----------------
if mode == "Teach":
    question = st.text_area(
        "Ask a math question",
        placeholder="e.g. Explain how to find the inverse of a 2x2 matrix"
    )

    if st.button("Explain"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""
Topic: {topic}
Context (if any): {context}
Question: {question}

Explain step by step.
End with ONE short question to check understanding.
"""}
            ]
            answer = ask_llm(messages, temp=0.3)

            st.markdown("### 📖 Step-by-Step Explanation")
            st.write(answer)

# ---------------- QUIZ MODE (STRICT STEP CHECKING) ----------------
if mode == "Quiz":

    if st.button("Generate Quiz"):
        quiz_prompt = f"""
Generate ONE mathematics problem.

Topic: {topic}
Level: {level}

Include clearly:
1. The question
2. A FULL step-by-step solution
3. The final answer (clearly labeled)
"""
        quiz = ask_llm(
            [{"role": "system", "content": quiz_prompt}],
            temp=0.2
        )
        st.session_state.quiz = quiz

    if "quiz" in st.session_state:
        st.markdown("### 📝 Quiz Problem")
        st.write(st.session_state.quiz)

        student_steps = st.text_area(
            "Write your solution step by step (methods + calculations)"
        )

        if st.button("Submit for Evaluation"):
            evaluation_prompt = f"""
You are a strict mathematics examiner.

TASK:
Compare the student's solution with the correct solution.

EVALUATE:
- Correctness of method
- Logical flow of steps
- Mathematical accuracy
- Final answer correctness

MARKING RULES:
- Wrong method = Incorrect (even if final answer matches)
- Identify the FIRST incorrect step
- Explain why it is incorrect
- Show the corrected step
- Give a FINAL VERDICT: Correct or Incorrect

PROBLEM & OFFICIAL SOLUTION:
{st.session_state.quiz}

STUDENT SOLUTION:
{student_steps}
"""
            feedback = ask_llm(
                [{"role": "system", "content": evaluation_prompt}],
                temp=0.1
            )

            st.markdown("### 🧠 Step-by-Step Evaluation")
            st.write(feedback)

            is_correct = "final verdict: correct" in feedback.lower()
            update_progress(user, topic, is_correct)

# ---------------- PROGRESS TRACKING ----------------
st.markdown("---")
st.markdown("### 📊 Learning Progress")

progress = load_progress()
if user in progress:
    st.json(progress[user])
else:
    st.write("No progress recorded yet.")
