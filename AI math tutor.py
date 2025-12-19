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

# ---------------- SESSION MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
def ask_llm_with_memory(user_input, temp=0.3):
    # Append user input to session history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Combine SYSTEM_PROMPT with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.chat_history
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temp
    )
    
    reply = response.choices[0].message.content
    
    # Save assistant reply to history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    
    return reply

# ---------------- RENDER MATH ----------------
def render_math_paper_style(text):
    """
    Renders text with mixed LaTeX and plain-text "If ... then ..." style.
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
    
    if st.button("Reset Memory"):
        st.session_state.chat_history = []
        st.success("Memory reset. You can start a new question.")

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

# ---------------- SYSTEM PROMPT (Mixed Style) ----------------
SYSTEM_PROMPT = """
You are a professional mathematics tutor.

Rules:
1. For derivations, fractions, powers, and equations, use LaTeX math inside $$ ... $$.
2. For modulo/case checks or “If ... then ...” explanations, write in plain text style, like:
If y ≡ 0 mod 4, then y^2 ≡ 0 mod 4
If y ≡ 1 mod 4, then y^2 ≡ 1 mod 4
If y ≡ 2 mod 4, then y^2 ≡ 0 mod 4
If y ≡ 3 mod 4, then y^2 ≡ 1 mod 4
3. DO NOT put parentheses around variables in any explanations.
4. Use \text{...} for textual explanations inside LaTeX.
5. Step numbers should be included in both LaTeX and plain-text steps.
6. Apply this style to all Teach Mode, Quiz Mode, and Hint outputs.
7. Remember previous messages in the session. If student asks "Explain more on Question 1" or "Go deeper", continue from previous context.
"""

# ================== TEACH MODE ==================
with tabs[0]:
    question = st.text_area(
        "Ask a math question or request further explanation:",
        placeholder="e.g. Solve x^2 + 3x - 4 = 0 or Explain more on Question 1"
    )

    if st.button("Explain Step-by-Step"):
        if not question.strip():
            st.warning("Please type a question or follow-up request.")
        else:
            answer = ask_llm_with_memory(f"Question: {question}\nContext: {context}")
            st.markdown("### ✏️ Solution (Textbook + If-Then Style)")
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
- Provide step-by-step solution.
- Use LaTeX math for equations, fractions, and powers.
- Use plain-text "If ... then ..." style for modulo/case checks.
- Include final answer clearly.
"""
        quizzes = ask_llm_with_memory(quiz_prompt)
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
Use LaTeX math for equations/fractions/powers and plain-text "If ... then ..." for modulo/case checks.
Do NOT give the final answer immediately.

Problem:
{q}
"""
                hints_text = ask_llm_with_memory(hint_prompt)
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
- Use LaTeX for equations/fractions/powers and "If ... then ..." style for modulo/case checks.
- End with: Final Verdict: Correct or Incorrect

Question & Solution:
{selected_quiz}

Student Answer:
{student_answer}
"""
            feedback = ask_llm_with_memory(eval_prompt, temp=0.1)
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
