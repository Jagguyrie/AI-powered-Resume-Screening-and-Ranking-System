import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screening & Ranking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to count words in text
def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

# Keyword banks for different fields
KEYWORD_BANKS = {
    "Data Analytics": [
        "python", "r", "sql", "tableau", "power bi", "excel", "statistics",
        "data visualization", "data cleaning", "etl", "machine learning",
        "regression", "clustering", "pandas", "numpy", "hypothesis testing",
        "a/b testing", "dashboard", "kpi", "metrics", "data mining", "big data",
        "hadoop", "spark", "data modeling", "data warehouse", "business intelligence"
    ],
    "Cybersecurity": [
        "network security", "threat analysis", "penetration testing", "vulnerability assessment",
        "firewall", "encryption", "security audit", "incident response", "security operations",
        "malware analysis", "siem", "ethical hacking", "security compliance", "risk assessment",
        "cryptography", "security architecture", "intrusion detection", "forensics", "iam",
        "osint", "zero trust", "soc", "iso 27001", "nist", "cis", "cissp", "ceh"
    ],
    "Software Development": [
        "java", "c++", "c#", "python", "javascript", "go", "ruby", "php", "scala", "kotlin",
        "object oriented", "algorithms", "data structures", "design patterns", "api",
        "microservices", "unit testing", "debugging", "version control", "git", "agile",
        "scrum", "ci/cd", "devops", "tdd", "database", "distributed systems", "scalability"
    ],
    "Web Development": [
        "html", "css", "javascript", "typescript", "react", "angular", "vue", "node.js",
        "express", "django", "flask", "ruby on rails", "php", "wordpress", "responsive design",
        "web apis", "rest", "graphql", "oauth", "jwt", "webpack", "babel", "sass", "less",
        "bootstrap", "tailwind", "seo", "web accessibility", "dom", "pwa"
    ],
    "AI/ML": [
        "machine learning", "deep learning", "neural networks", "natural language processing",
        "computer vision", "reinforcement learning", "tensorflow", "pytorch", "keras", "scikit-learn",
        "feature engineering", "model validation", "hyperparameter tuning", "data preprocessing",
        "classification", "regression", "clustering", "dimensionality reduction", "ensemble methods",
        "transfer learning", "transformer models", "gpt", "bert", "attention mechanism", "cnn", "rnn", "lstm"
    ]
}

# Blacklisted words/phrases that may lead to rejection
BLACKLISTED_WORDS = [
    "currently pursuing", "soon to be", "aiming to", "looking for", "seeking position",
    "no experience", "little experience", "familiar with", "basic knowledge", "beginner level",
    "new graduate", "entry level", "internship", "freshman", "sophomore", "junior", "senior",
    "coursework", "project experience only", "academic experience", "school project"
]

# Function to check for blacklisted words
def check_blacklisted_words(text):
    text_lower = text.lower()
    found_words = []
    for word in BLACKLISTED_WORDS:
        if word in text_lower:
            found_words.append(word)
    return found_words

# Function to validate resume length
def validate_resume_length(text, min, max):
    word_count = count_words(text)
    if word_count < min:
        return False, f"Resume is too short ({word_count} words). Minimum: {min} words."
    elif word_count > max:
        return False, f"Resume is too long ({word_count} words). Maximum: {max} words."
    return True, f"Resume length is acceptable ({word_count} words)."

# Function to rank resumes based on job description with improved scoring
def rank_resumes(job_description, resumes, job_field=None):
    # Basic TF-IDF scoring
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    # Improved scoring with field-specific keywords if a field is selected
    if job_field and job_field in KEYWORD_BANKS:
        keyword_scores = []
        field_keywords = KEYWORD_BANKS[job_field]

        for resume in resumes:
            resume_lower = resume.lower()
            # Count matching keywords
            matches = sum(1 for keyword in field_keywords if keyword in resume_lower)
            # Calculate keyword score (normalized by total keywords)
            keyword_score = matches / len(field_keywords)
            keyword_scores.append(keyword_score)

        # Combine cosine similarity (70%) with keyword matching (30%)
        combined_scores = (0.7 * cosine_similarities) + (0.3 * np.array(keyword_scores))
        return combined_scores

    return cosine_similarities

# Function to create a downloadable link
def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download results as CSV</a>'
    return href

# Function to generate word cloud data
def generate_wordcloud_data(text):
    from collections import Counter
    import re

    # List of common stopwords
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'when', 'where', 'how', 'i', 'he', 'she', 'it', 'we', 'they', 'to', 'of',
                'for', 'with', 'by', 'at', 'on', 'in']

    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords and short words
    words = [word for word in words if word not in stopwords and len(word) > 2]

    # Count words
    word_counts = Counter(words)

    # Get top words
    top_words = word_counts.most_common(25)

    return top_words

# App title
st.title("📄 AI Resume Screening & Candidate Ranking System")
st.markdown("<p>Upload resumes and enter a job description to find the best matches using TF-IDF and cosine similarity.</p>", unsafe_allow_html=True)

# Create two columns for the main layout
col1, col2 = st.columns([1, 2])

with col1:
    # Job description input
    st.header("🔍 Job Description")
    job_description = st.text_area("Enter the job description", height=200, placeholder="Paste the job description here...")

    # Job field selection for keyword matching
    job_field = st.selectbox("Select Job Field", ["Select a field..."] + list(KEYWORD_BANKS.keys()))

    if job_field == "Select a field...":
        job_field = None

    st.subheader("Word Count Limits")
    word_count_min = st.slider("Minimum Word Count", 200, 600, 450)
    word_count_max = st.slider("Maximum Word Count", 450, 1000, 600)

    # Show key skills section with expandable text input
    with st.expander("📌 Highlight Key Skills (Optional)"):
        key_skills = st.text_area("Enter key skills or requirements separated by commas",
                                 placeholder="Python, machine learning, data analysis, etc.")

    # File uploader
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    # Display number of uploaded files
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} resumes uploaded")

        # Process button
        process_button = st.button("Analyze Resumes", use_container_width=True)

with col2:
    if uploaded_files and job_description and 'process_button' in locals() and process_button:
        st.header("🔎 Analyzing Resumes")

        # Progress bar for visual feedback
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process resumes
        resumes = []
        resume_texts = []
        rejected_resumes = []
        blacklist_violations = {}
        validation_messages = {}

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            text = extract_text_from_pdf(file)

            # Validate resume length
            valid, message = validate_resume_length(text, word_count_min, word_count_max)
            validation_messages[file.name] = message

            # Check for blacklisted words
            blacklisted = check_blacklisted_words(text)
            if blacklisted:
                blacklist_violations[file.name] = blacklisted

            # Reject resume if it fails validation or contains blacklisted words
            if not valid or blacklisted:
                rejected_resumes.append(file.name)
            else:
                processed_text = preprocess_text(text)
                resumes.append(processed_text)
                resume_texts.append(text)

            progress_bar.progress((i + 1) / len(uploaded_files))

        # Process job description
        status_text.text("Processing job description...")
        job_description_processed = preprocess_text(job_description)

        # Display rejected resumes if any
        if rejected_resumes:
            st.warning(f"{len(rejected_resumes)} resumes rejected")
            with st.expander("View Rejected Resumes"):
                for resume in rejected_resumes:
                    st.markdown(f"**{resume}**")
                    if resume in validation_messages:
                        st.markdown(f"- {validation_messages[resume]}")
                    if resume in blacklist_violations:
                        st.markdown(f"- Contains blacklisted terms: {', '.join(blacklist_violations[resume])}")
                    st.markdown("---")

        # Check if any valid resumes remain
        if not resumes:
            st.error("No valid resumes to analyze. All uploads were rejected.")
            progress_bar.empty()
            status_text.empty()
        else:
            # Rank resumes
            status_text.text("Ranking candidates...")
            scores = rank_resumes(job_description_processed, resumes, job_field)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Get valid resume names
            valid_resume_names = [file.name for file in uploaded_files if file.name not in rejected_resumes]

            # Display scores
            results = pd.DataFrame({
                "Resume": valid_resume_names,
                "Match Score": np.round(scores * 100, 2)
            })
            results = results.sort_values(by="Match Score", ascending=False)

            # Highlight the best resume
            best_resume = results.iloc[0]
            st.success(f"🏆 Best Match: **{best_resume['Resume']}** with a score of **{best_resume['Match Score']}%**")

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Ranking", "📋 Detailed View", "📈 Analysis"])

            with tab1:
                # Show results in a table
                st.subheader("Candidate Ranking")
                st.dataframe(results, use_container_width=True)
                st.markdown(get_download_link(results, "resume_rankings.csv"), unsafe_allow_html=True)

                # Display bar chart
                fig = px.bar(
                    results,
                    x="Resume",
                    y="Match Score",
                    color="Match Score",
                    color_continuous_scale=["#4361ee", "#3a0ca3"],
                    text="Match Score",
                    labels={"Match Score": "Match Score (%)"},
                    height=400
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Show detailed view of selected resume
                st.subheader("Resume Content")
                selected_resume = st.selectbox("Select a resume to view", results["Resume"].tolist())

                # Get index of selected resume
                selected_index = results[results["Resume"] == selected_resume].index[0]

                # Display resume content and score
                selected_score = results.loc[selected_index, "Match Score"]
                st.metric("Match Score", f"{selected_score}%")

                # Show word count
                word_count = count_words(resume_texts[selected_index])
                st.metric("Word Count", word_count)

                # Show resume text with expander
                with st.expander("View Resume Content", expanded=True):
                    st.markdown(f"<div style='height: 300px; overflow-y: scroll; padding: 10px; background-color: white; border-radius: 5px;'>{resume_texts[selected_index]}</div>", unsafe_allow_html=True)

            with tab3:
                # Analysis tab
                st.subheader("Keyword Analysis")

                # Get keywords from job description
                job_keywords = generate_wordcloud_data(job_description)
                job_df = pd.DataFrame(job_keywords, columns=["Keyword", "Count"])

                # Display field-specific keywords if a field is selected
                if job_field:
                    st.subheader(f"Field-Specific Keywords: {job_field}")
                    field_keywords = KEYWORD_BANKS[job_field]

                    # Show which field keywords appear in the selected resume
                    selected_resume = st.selectbox("Select resume for keyword analysis",
                                                 results["Resume"].tolist(),
                                                 key="field_keyword_select")
                    selected_idx = valid_resume_names.index(selected_resume)
                    resume_text_lower = resume_texts[selected_idx].lower()

                    matched_keywords = []
                    missing_keywords = []

                    for keyword in field_keywords:
                        if keyword in resume_text_lower:
                            matched_keywords.append(keyword)
                        else:
                            missing_keywords.append(keyword)

                    col_match, col_miss = st.columns(2)

                    with col_match:
                        st.markdown(f"##### Matched Keywords ({len(matched_keywords)})")
                        if matched_keywords:
                            st.markdown("<div style='height: 200px; overflow-y: scroll;'>", unsafe_allow_html=True)
                            for keyword in matched_keywords:
                                st.markdown(f"✅ {keyword}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("No field-specific keywords found.")

                    with col_miss:
                        st.markdown(f"##### Missing Keywords ({len(missing_keywords)})")
                        if missing_keywords:
                            st.markdown("<div style='height: 200px; overflow-y: scroll;'>", unsafe_allow_html=True)
                            for keyword in missing_keywords:
                                st.markdown(f"❌ {keyword}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("All field-specific keywords found!")

                # Get keywords from selected resume
                st.subheader("Word Frequency Analysis")
                selected_resume = st.selectbox("Select resume for word analysis",
                                             results["Resume"].tolist(),
                                             key="analysis_select")
                selected_idx = valid_resume_names.index(selected_resume)
                resume_keywords = generate_wordcloud_data(resume_texts[selected_idx])
                resume_df = pd.DataFrame(resume_keywords, columns=["Keyword", "Count"])

                # Display side by side
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("##### Job Description Keywords")
                    fig_job = px.bar(
                        job_df.head(10),
                        x="Count",
                        y="Keyword",
                        orientation='h',
                        color="Count",
                        color_continuous_scale=["#4361ee", "#3a0ca3"],
                        labels={"Count": "Frequency"},
                        height=300
                    )
                    fig_job.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=30, b=20),
                        coloraxis_showscale=False,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_job, use_container_width=True)

                with col_b:
                    st.markdown(f"##### Resume Keywords: {selected_resume}")
                    fig_resume = px.bar(
                        resume_df.head(10),
                        x="Count",
                        y="Keyword",
                        orientation='h',
                        color="Count",
                        color_continuous_scale=["#4361ee", "#3a0ca3"],
                        labels={"Count": "Frequency"},
                        height=300
                    )
                    fig_resume.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=30, b=20),
                        coloraxis_showscale=False,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_resume, use_container_width=True)

# Display instructions when no files are uploaded
if not uploaded_files:
    with col2:
        st.header("👋 Getting Started")
        st.markdown("""
        ### How to use this tool:

        1. **Enter a job description** on the left sidebar
        2. **Select the job field** for specialized keyword matching
        3. **Upload resumes** in PDF format (multiple files supported)
        4. **Click "Analyze Resumes"** to process and rank candidates
        5. **Review the results** in the ranking and analysis tabs

        This tool uses TF-IDF vectorization and cosine similarity to match resumes
        with job descriptions, enhanced with field-specific keyword matching.

        **Note:** Resumes that fall outside the 450-600 word range or contain
        blacklisted phrases will be automatically rejected.
        """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px;">
    <p>AI Resume Screening & Ranking System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
