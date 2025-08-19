import streamlit as st
import pickle
import re

# Optional: For PDF and DOCX support
import PyPDF2
from docx import Document

# Load your trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s*', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s*', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = cleanText.lower()
    return cleanText

def ordinal(n):
    # Returns 1st, 2nd, 3rd, 4th, etc.
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

st.title("AI Resume Screening for Job Applications")

# User enters requirements/keywords
user_requirements = st.text_input("Enter job requirements or keywords (comma separated):", "")

# Option to paste resume text
st.markdown("**Or paste resume text below (one per box, click 'Add Another' for more):**")
resume_text_inputs = []
resume_text = st.text_area("Paste Resume Text 1 (optional)", "")
if resume_text.strip():
    resume_text_inputs.append(("Pasted Resume 1", resume_text))

add_more = st.checkbox("Add another pasted resume")
if add_more:
    resume_text2 = st.text_area("Paste Resume Text 2 (optional)", "")
    if resume_text2.strip():
        resume_text_inputs.append(("Pasted Resume 2", resume_text2))

# Upload multiple resumes (TXT, PDF, DOCX)
uploaded_files = st.file_uploader(
    "Upload multiple resumes (TXT, PDF, DOCX)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

resume_texts = []
resume_names = []

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = ""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = uploaded_file.read().decode("utf-8")
        resume_texts.append(text)
        resume_names.append(uploaded_file.name)

# Add pasted resumes to the lists
for name, text in resume_text_inputs:
    resume_texts.append(text)
    resume_names.append(name)

if (resume_texts and user_requirements.strip()):
    # Clean and vectorize resumes
    cleaned_resumes = [cleanResume(txt) for txt in resume_texts]
    features = tfidf.transform(cleaned_resumes)

    # Clean and vectorize user requirements
    cleaned_req = cleanResume(user_requirements)
    req_features = tfidf.transform([cleaned_req])

    # Compute cosine similarity between requirements and each resume
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(req_features, features)[0]

    # Predict categories for each resume
    # If you have category_mapping, you can show predicted category as well
    try:
        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
            24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
            1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
            17: "Network Security Engineer", 21: "Sap Developer", 5: "Civil Engineer", 0: "Advocate"
        }
        predicted_ids = clf.predict(features)
        predicted_categories = [category_mapping.get(pid, "Unknown") for pid in predicted_ids]
    except Exception:
        predicted_categories = ["N/A"] * len(resume_names)

    # Rank resumes by similarity (no threshold)
    ranked = sorted(
        zip(resume_names, resume_texts, similarities, predicted_categories),
        key=lambda x: x[2], reverse=True
    )

    if ranked:
        st.subheader("All Resumes (ranked by match to your requirements):")
        for i, (name, text, score, pred_cat) in enumerate(ranked, 1):
            rank_str = f"{ordinal(i)} Rank"
            st.markdown(f"**{rank_str}: {name}** (Match Score: {score:.2f}, Predicted Category: {pred_cat})")
        # Show full previews of all resumes in one expander
        with st.expander(" Show All Resume Previews"):
            for i, (name, text, score, pred_cat) in enumerate(ranked, 1):
                st.markdown(f"**{ordinal(i)} Resume - {name}** (Predicted Category: {pred_cat}, Match Score: {score:.2f})")
                st.write(text)
                st.markdown("---")
    else:
        st.warning("No resumes uploaded or pasted.")
elif resume_texts:
    st.info("Please enter job requirements or keywords above to rank resumes.")
