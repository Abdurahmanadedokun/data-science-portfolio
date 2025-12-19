import streamlit as st
import openai
import os
import json
from langchain_community.document_loaders import PyPDFLoader

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart AI Math Tutor", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:#4B0082;'>📘 Smart AI Math Tutor</h1>",
    unsafe_allow_html=True
)

# ---------------- API KEY ----------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Add it to Streamlit Secrets.")
    st.stop()

# ---------------- PROGRESS ----------------
PROGRESS_FILE = "progress.json"

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE, "r") as f:
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

# ---------------- LLM ----------------
def ask_llm(messages, temp=0.3):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temp
    )
    return response.choices[0].message.content

# ---------------- RENDER MATH ----------------
def render_math_paper_style(text):
    """
    Renders text with LaTeX blocks properly.
    Anything wrapped in $$ ... $$ is rendered as math.
    """
    blocks = text.split("$$")
    for i, block in enumerate(blocks):
        if i % 2 == 1:
            st.latex(block)
        else:
            if block.strip():
                st.markdown(block)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    user = st.text_input("Student Name", "student")
    pdf = st.file_uploader("Upload Math PDF (optional)", type="pdf")

# ---------------- PDF CONTEXT ----------------
context = ""
if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    texts = [d.page_content for d in docs if d.page_content.strip()]
    
    if texts:
        context = "\n".join(texts[:3])
        st.sidebar.success("PDF loaded successfully! (Used as context only)")
    else:
        st.sidebar.warning("PDF has no readable text. Skipping context.")

# ---------------- TABS ----------------
tabs = st.tabs(["📖 Teach Mode", "📝 Quiz Mode", "📊 Progress"])

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are a professional mathematics tutor.
Rules:
- Automatically identify the topic.
- Explain step by step like writing on paper.
- ALL mathematics must be written in LaTeX.
- Use × for multiplication, fractions, powers, aligned equations.
- Wrap ALL math expressions inside $$ ... $$ for block math.
- Do NOT put parentheses around variables like (x) or (y^2). Use proper LaTeX: x, y^2, etc.
- For multi-step derivations, use aligned equations with \\begin{align*} ... \\end{align*}.
- Number each step clearly.
- Use uploaded PDF content if available to improve the answer.
"""

# ================== TEACH MODE ==================
with tabs[0]:
    question = st.text_area(
        "Ask a math question:",
        placeholder="e.g. Solve x^2 + 3x - 4 = 0"
    )

    if st.button("Explain Step-by-Step"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nContext: {context}"}
            ]
            answer = ask_llm(messages)
            st.markdown("### ✏️ Solution (Paper Style)")
            render_math_paper_style(answer)

# ================== QUIZ MODE ==================
with tabs[1]:
    num_q = st.number_input("Number of quiz questions", 1, 10, 3)

    if st.button("Generate Quiz"):
        progress = load_progress()
        user_data = progress.get(user, {})
        attempts = sum(v["attempts"] for v in user_data.values())
        level = "Beginner" if attempts < 3 else "Intermediate" if attempts < 6 else "Advanced"

        quiz_prompt = f"""
Generate {num_q} {level} math questions.
For EACH question:
- Provide step-by-step solution
- Use LaTeX math only
- Paper-style formatting
- Include final answer clearly
Wrap all math expressions in $$ $$.
"""
        quizzes = ask_llm([{"role": "system", "content": quiz_prompt}])
        st.session_state.quizzes = quizzes.split("\n\n")
        st.session_state.selected = 0
        st.session_state.hint_index = 0
        st.session_state.hints = []

    if "quizzes" in st.session_state:
        st.markdown("### 📘 Generated Questions")
        for i, q in enumerate(st.session_state.quizzes):
            if st.button(f"Question {i+1}", key=f"quiz_{i}"):
                st.session_state.selected = i
                st.session_state.hint_index = 0
                # Request step-by-step hints
                hint_prompt = f"""
Split the solution of the following math problem into 3 hints, step by step.
Do NOT give the final answer immediately.
Format as numbered hints in LaTeX math mode.

Problem:
{q}
"""
                hints_text = ask_llm([{"role":"system","content":SYSTEM_PROMPT + "\n" + hint_prompt}])
                st.session_state.hints = [h for h in hints_text.split("\n") if h.strip()]

        selected_quiz = st.session_state.quizzes[st.session_state.selected]
        st.markdown("### 📝 Selected Question")
        render_math_paper_style(selected_quiz)

        if st.session_state.hints:
            if st.button("Reveal Next Hint"):
                idx = st.session_state.hint_index
                if idx < len(st.session_state.hints):
                    st.markdown(f"**Hint {idx+1}:** {st.session_state.hints[idx]}")
                    st.session_state.hint_index += 1
                else:
                    st.info("No more hints available for this question.")

        student_answer = st.text_area("Write your solution (step-by-step):", key="student_answer")

        if st.button("Submit Answer"):
            eval_prompt = f"""
You are a strict math examiner.
- Check each step logically.
- Point out first wrong step (if any).
- Provide correction hints.
- Use LaTeX paper-style math.
- End with: Final Verdict: Correct or Incorrect

Question & Solution:
{selected_quiz}

Student Answer:
{student_answer}
"""
            feedback = ask_llm([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": eval_prompt}], temp=0.1)
            st.markdown("### 🧠 Feedback")
            render_math_paper_style(feedback)

            correct = "final verdict: correct" in feedback.lower()
            update_progress(user, "math", correct)

# ================== PROGRESS ==================
with tabs[2]:
    st.markdown("### 📊 Student Progress")
    progress = load_progress()
    if user in progress:
        st.json(progress[user])
    else:
        st.info("No progress yet. Start practicing!")
