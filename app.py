import streamlit as st
import pickle
import re

# Load your trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s*', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s*', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', r' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = cleanText.lower()
    return cleanText

def ordinal(n):
    # Returns 1st, 2nd, 3rd, 4th, etc.
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

st.title("AI Resume Screening for Job Applications")

# User enters requirements/keywords
user_requirements = st.text_input("Enter job requirements or keywords (comma separated):", "")

# Upload multiple resumes (TXT or PDF)
uploaded_files = st.file_uploader("Upload multiple resumes (TXT or PDF)", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files and user_requirements.strip():
    resume_texts = []
    resume_names = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        else:
            text = uploaded_file.read().decode("utf-8")
        resume_texts.append(text)
        resume_names.append(uploaded_file.name)

    # Clean and vectorize resumes
    cleaned_resumes = [cleanResume(txt) for txt in resume_texts]
    features = tfidf.transform(cleaned_resumes)

    # Clean and vectorize user requirements
    cleaned_req = cleanResume(user_requirements)
    req_features = tfidf.transform([cleaned_req])

    # Compute cosine similarity between requirements and each resume
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(req_features, features)[0]

    # Rank resumes by similarity
    ranked = sorted(zip(resume_names, resume_texts, similarities), key=lambda x: x[2], reverse=True)
    st.subheader("Resumes ranked by match to your requirements:")
    for i, (name, text, score) in enumerate(ranked, 1):
        rank_str = f"{ordinal(i)} Rank"
        st.markdown(f"{rank_str}: {name}** (Match Score: {score:.2f})")
        with st.expander("Show Resume Preview"):
            st.write(text[:1000])
elif uploaded_files:
    st.info("Please enter job requirements or keywords above to rank resumes.")
