
import streamlit as st
from together import Together
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import numpy as np
import re
import os
from huggingface_hub import InferenceClient
import json
# Load environment variables

# LLaMA 3.3 API configuration
api_key = os.getenv("HF_API_KEY")

# -------------------------
# Model Loaders
# -------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# RAG Utilities
# -------------------------
def extract_paragraphs_from_pdfs(uploaded_files):
    paragraphs = []
    for pdf in uploaded_files:
        reader = PyPDF2.PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text() or ''
            blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
            paragraphs.extend(blocks)
    return paragraphs

def build_faiss_index(paragraphs):
    model = load_sentence_transformer()
    embeddings = model.encode(paragraphs, show_progress_bar=False)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed)
    return index, model

def retrieve_top_k(query, index, model, paragraphs, k=3):
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = q_emb / np.linalg.norm(q_emb)
    _, ids = index.search(q_emb, k)
    return [paragraphs[i] for i in ids[0]]

def generate_answer_with_together(prompt, temperature=0.3):
    client = InferenceClient(
    api_key=api_key
    )
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# Adaptive Testing Utilities
# -------------------------
def generate_mcq(difficulty, topic, history):
    prompt = (
        f"\n You have to generate a quiz question (MCQ type) on the topic {topic} based on the following instructions given below :"
        f"\n INSTRUCTIONS:\n"
        f"\n There are 4 difficulty levels present : Easy , Moderate , Intermediate and Advanced. Each difficulty Level has its specific characterstics. \n"
        f"Easy : Questions focus solely on fundamental definitions, straightforward facts, and key terminology.They are designed to test basic recall and introductory knowledge of the topic.No in-depth understanding or contextual application is required.\n"
        f"Moderate : Questions emphasize understanding core concepts and their immediate context.They assess the ability to explain, interpret, or rephrase ideas.The focus is on comprehension rather than trickiness or subtle distinctions.\n"
        f"Intermediate : These questions should be tricky to answer (like the options are confusing to choose from , etc)They may involve multi-step reasoning, slight misdirection in answer choices, or interpretation of deeper contextual meaning.These questions aim to evaluate decision-making and precision in understanding.\n"
        f"Advanced : Questions require advanced analytical thinking, problem-solving, and integration of multiple, interconnected concepts.They are application-based and demand reasoning, critical analysis, or evaluation to arrive at the correct answer.Often, these questions simulate real-world scenarios or advanced theoretical applications."
        f"Generate a {difficulty} level quiz question (MCQ type) on the topic {topic}"
        f"\n NOTE: \n Do not give answer in the response. Your response should strictly consist only the question and the options, nothing else should be included in the response."
        f"\n The given question should strictly consist a single answer. That is for a question there shouldn't be multiple correct options"
        f"\n A list of questions are given below , Do not generate the same question again from the below list of questions"
        f"\n questions : {history}"
        f"\n Note : Make sure your question should relatable to a computer science undergraduate student. It should not be way out of scope for the students"
        f"Try to be a little bit creatived with the questions , do not try to give the most obvious question"
    )
    return generate_answer_with_together(prompt, temperature=0.7)

def evaluate_mcq(question, answer):
    prompt = (
        f"You have to evaluate the given question and answer"
        f"Question : {question}\n"
        f"Answer : {answer}\n"
        f"If the given answer is correct then give response as True or else give the response as False."
        f"Note : Do not include any other words or staements or punctuation in your response. Your response should consist only 1 word."
    )
    return generate_answer_with_together(prompt)

def explain_mcq(question):
    prompt = (
        f"Answer the given question and give a short explanation for why the answer is correct"
        f"for the given question. Do not include extra commentary.\n"
        f"Question: {question}\n"
    )
    return generate_answer_with_together(prompt)

# -------------------------
# Difficulty progression
# -------------------------
def update_difficulty(correct):
    d = st.session_state.diff
    c = st.session_state.corr
    w = st.session_state.wrong
    if correct:
        c += 1
    else:
        w += 1
    if d == 'Easy' and c >= 2:
        d, c, w = 'Moderate', 0, 0
    elif d == 'Moderate':
        if c >= 3:
            d, c, w = 'Intermediate', 0, 0
        elif w >= 2:
            d, c, w = 'Easy', 0, 0
    elif d == 'Intermediate':
        if c >= 3:
            d, c, w = 'Advanced', 0, 0
        elif w >= 3:
            d, c, w = 'Moderate', 0, 0
    elif d == 'Advanced' and w >= 2:
        d, c, w = 'Intermediate', 0, 0
    st.session_state.diff = d
    st.session_state.corr = c
    st.session_state.wrong = w

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title='TeachMate', layout='wide')

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stTextInput > div > input,
    .stNumberInput > div > input {
        margin-top: -8px !important;
    } 


    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(82,197,247,1) 50%, rgba(83,237,234,1) 100%);
        background-size: cover;
        background-attachment: fixed;
        color:black;
    }

    .home-container {
        text-align: center;
        margin-top: 15vh;
    }

    .main-title {
        font-size: 3.5em;
        font-weight: 800;
        margin-bottom: 0.3em;
        color: #333;
    }

    .sub-title {
        font-size: 1.3em;
        font-weight: 400;
        margin-bottom: 2em;
        color: #333;
    }

    .button-row {
        display: flex;
        justify-content: center;
        gap: 2em;
        margin-top: 1em;
    }
    header { visibility: hidden; }
    .block-container {
        padding-top: 2rem;
    }
    .top-left-title {
    position: fixed;
    top: 20px;
    left: 30px;
    font-size: 1.8rem;
    font-weight: 700;
    color: black;
    
    padding: 8px 16px;
   
    z-index: 1000;
    font-family: 'Inter', sans-serif;
    }
    .stRadio label {
    color: black;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    }
    /* Style radio button labels */
    /* Style the radio option text (A, B, C, D) */
    div[role="radiogroup"] label span {
        color: black !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    .stRadio div[role="radiogroup"] label > div:nth-child(2) {
    color: black !important;
    }

    /* Uniform button style for the entire app */
    div.stButton > button {
        padding: 0.75em 2em;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 30px;
        background-color: #7a5af8;
        color: white;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: all 0.2s ease-in-out;
        font-family: 'Inter', sans-serif;
    }

    /* Hover effect */
    div.stButton > button:hover {
        transform: scale(1.05);
    }

    * {
    color: black !important;
    font-family: 'Inter', sans-serif !important;
    }

    input[type="text"],
    input[type="number"] {
        color: white !important;
        background-color: rgba(0, 0, 0, 0.25) !important; /* Optional for better contrast */
        border: 1px solid #ffffff55 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Optional: placeholder text color */
    input::placeholder {
        color: #cccccc !important;
    }


    </style>
""", unsafe_allow_html=True)







ss = st.session_state
# Initialize defaults
for key, val in {
    'mode': None,
    'diff': 'Easy',
    'corr': 0,
    'wrong': 0,
    'num_q': 10,
    'topic': 'Computer Science',
    'idx': 0,
    'history': [],
    'results': {},
    'paragraphs': None,
    'faiss_idx': None,
    'rag_model': None,
    'rag_active': False
}.items():
    ss.setdefault(key, val)

# -------------------------
# Home Button on Top Right
# -------------------------
home_col, btn_col = st.columns([9,1])
with home_col:
    pass
if btn_col.button('Home'):
    ss.mode = None
    ss.idx = 0
    ss.history = []
    ss.results = {}
    ss.paragraphs = None
    ss.faiss_idx = None
    ss.rag_model = None
    ss.rag_active = False
# # No explicit rerun here; Streamlit will auto rerun on state change

# -------------------------
# Home Screen
# -------------------------


if ss.mode is None:
    # Top-right logo (TeachMate)
    # st.markdown("""
    #     <div style="text-align: left; padding: 1.5rem 2rem 0 0; color: black;">
    #         <span style="font-size: 1.5rem; font-weight: 700;">TeachMate</span>
    #     </div>
    # """, unsafe_allow_html=True)

    # Centered Hero Section
    st.markdown("""
        <style>
        .button-center-row {
            display: flex;
            justify-content: center;
            gap: 2rem;
        }

        div.stButton > button {
            padding: 0.75em 2em;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 30px;
            background-color: #7a5af8;
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            transition: all 0.2s ease-in-out;
        }

        div.stButton > button:hover {
            transform: scale(1.05);
        }
        </style>

        <div style="display: flex; flex-direction: column; align-items: center;  text-align: center;margin-top: 10vh;">
            <h1 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; color: black;">
                AI That Reads Your Book, Answers Your Questions,<br> and Tests You Like a Pro.
            </h1>
            <p style="font-size: 1.2rem; max-width: 600px; margin-bottom: 2rem; color: black;">
                Upload your textbook, ask anything, and practice with adaptive quizzes — all in one place, powered by TeachMate.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="top-left-title">
        TeachMate
    </div>
    """, unsafe_allow_html=True)

    # Custom styles that actually apply to Streamlit buttons
    # Button styling (if not already added)# --- Style buttons & layout --
    # --- Layout using container ---
    with st.container():
        st.markdown('<div class="button-center-row">', unsafe_allow_html=True)
        col2, col3, col4, col5 = st.columns([ 6, 2, 2, 6])
        with col3:
            if st.button("Quick Quiz"):
                ss.mode = "adaptive"
        with col4:
            if st.button("Chatbot"):
                ss.mode = "rag"
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# -------------------------
# Adaptive Testing Flow
# -------------------------
if ss.mode == 'adaptive':
    st.title('Adaptive Testing')

    # st.markdown("""
    # <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 1rem; color: black; font-family: 'Inter', sans-serif;">
    #     Adaptive Testing
    # </h1>
    # """, unsafe_allow_html=True)
    # Settings
    if ss.idx == 0 and not ss.history:
        # ss.num_q = st.number_input('Number of Questions', 10, 30, ss.num_q)
        # ss.topic = st.text_input('Topic', ss.topic)
   
        ss.num_q = st.number_input('', 10, 30, ss.num_q)
        st.markdown('<label style="color: black; font-weight: 600; font-family: Inter, sans-serif; margin-bottom: 4px; display: block;">Number of Questions</label>', unsafe_allow_html=True)


        ss.topic = st.text_input('', ss.topic)
        st.markdown('<label style="color: black; font-weight: 600; font-family: Inter, sans-serif; margin-bottom: 4px; display: block;">Topic</label>', unsafe_allow_html=True)
        if st.button('Start Test'):
            ss.idx = 1
        st.stop()
    # Question Loop
    if ss.idx <= ss.num_q:
        if ss.idx not in ss.results:
            question = generate_mcq(ss.diff, ss.topic, '\n'.join(ss.history))
            ss.history.append(question)
            ss.results[ss.idx] = {'question': question}
        # st.markdown(f"**Question {ss.idx}/{ss.num_q} ({ss.diff})**")
        # st.write(ss.results[ss.idx]['question'])

        st.markdown(f"""
            <div style="color: black; font-weight: 700; font-size: 1.2rem; font-family: 'Inter', sans-serif;">
                Question {ss.idx}/{ss.num_q} ({ss.diff})
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="color: black; font-size: 1.05rem; font-family: 'Inter', sans-serif;">
                {ss.results[ss.idx]['question']}
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            /* Target the text inside each radio option */
            div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
                color: black !important;
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        answer = st.radio('Answer:', ['A','B','C','D'], key=f'ans{ss.idx}')

        def submit_callback():
            correct = evaluate_mcq(ss.results[ss.idx]['question'], answer) == 'True'
            ss.results[ss.idx].update({'answer': answer, 'correct': correct})
            update_difficulty(correct)
            ss.idx += 1

        st.button('Submit', key=f'sub{ss.idx}', on_click=submit_callback)
        st.stop()


    # Report
    st.title('Test Report')
    score = sum(1 for r in ss.results.values() if r.get('correct'))
    st.write(f'**Score: {score}/{ss.num_q}**')
    for i, r in ss.results.items():
        st.markdown(f"**{i}.** {r['question']}")
        st.write(f"Your Answer: {r.get('answer')} - {'✅' if r.get('correct') else '❌'}")
        if st.button(f'Explain {i}', key=f'exp{i}'):
            st.write(explain_mcq(r['question']))
    st.stop()

# -------------------------
# RAG Chatbot Flow
# -------------------------
if ss.mode == 'rag':
    st.title('RAG Chatbot')
    # Upload PDFs
    if ss.paragraphs is None:
        st.markdown("""
            <style>
           
            /* Optional: make "Browse files" button text white too */
            .stFileUploader button {
                color: white !important;
                border-color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
        files = st.file_uploader('Upload PDF(s) to start', type='pdf', accept_multiple_files=True)
        if files:
            paragraphs = extract_paragraphs_from_pdfs(files)
            ss.paragraphs = paragraphs
            idx, model = build_faiss_index(paragraphs)
            ss.faiss_idx, ss.rag_model = idx, model
            st.success(f'Extracted {len(paragraphs)} paragraphs')
            if st.button('Chat with PDF'):
                ss.rag_active = True
        st.stop()
    # Activate Chat Interface
    if not ss.rag_active:
        if st.button('Chat with PDF'):
            ss.rag_active = True
        else:
            st.stop()
    # Chat UI
    query = st.text_input('Enter your query:')
    k = st.slider('Top K passages', 1, 5, 3)
    if st.button('Generate Insight'):
        # Show insight first
        top_paras = retrieve_top_k(query, ss.faiss_idx, ss.rag_model, ss.paragraphs, k)
        context = ' '.join(top_paras)
        prompt = (
            f"INSTRUCTIONS:\nAnswer the QUESTION using the DOCUMENT text above. If DOCUMENT lacks info, say NONE.\n"
            f"DOCUMENT:\n{context}\nQUESTION:\n{query}\nANSWER:"
        )
        insight = generate_answer_with_together(prompt)
        st.write('### Insight')
        st.write(insight)
        # Then passages
        st.write('### Retrieved Passages')
        for p in top_paras:
            st.write(p)
