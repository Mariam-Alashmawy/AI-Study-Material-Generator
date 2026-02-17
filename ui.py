import streamlit as st
import requests

st.set_page_config(page_title="AI Study Buddy", page_icon="🎓")
st.title("🎓 AI Study Material Generator")

API_URL = "http://localhost:8000/generate_study_material"
API_KEY = "secret123"

if "study_data" not in st.session_state:
    st.session_state.study_data = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if st.button("Generate Study Materials"):
    if uploaded_file:
        with st.spinner("Generating..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            headers = {"Authorization": f"Bearer {API_KEY}"}
            resp = requests.post(API_URL, files=files, headers=headers)
            if resp.status_code == 200:
                st.session_state.study_data = resp.json()

if st.session_state.study_data:
    data = st.session_state.study_data
    tab1, tab2 = st.tabs(["📝 Quiz", "🗂️ Flashcards"])
    
    with tab1:
        with st.form("quiz_form"):
            user_answers = []
            for i, q in enumerate(data.get("questions", [])):
                st.write(f"**Q{i+1}:** {q['question_text']}")
                # index=None ensures no default radio button is selected
                ans = st.radio(f"Select answer", q['options'], index=None, key=f"q_{i}")
                user_answers.append((ans, q['correct_answer']))
            
            if st.form_submit_button("Submit Quiz"):
                score = 0
                for i, (ua, ca) in enumerate(user_answers):
                    if ua and ua.strip().lower() == str(ca).strip().lower():
                        st.success(f"Q{i+1}: Correct!")
                        score += 1
                    else:
                        st.error(f"Q{i+1}: Incorrect. Correct answer: {ca}")
                st.metric("Score", f"{score}/{len(user_answers)}")

    with tab2:
        for card in data.get("flashcards", []):
            with st.container(border=True):
                st.write(f"**Front:** {card['front']}")
                with st.expander("Show Back"):
                    st.write(card['back'])
