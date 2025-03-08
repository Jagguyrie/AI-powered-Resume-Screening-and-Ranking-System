# AI-powered-Resume-Screening-and-Ranking-System

This application is designed to streamline the resume screening process using artificial intelligence. It utilizes natural language processing (NLP) techniques to analyze resumes and rank candidates based on their match with a provided job description.

## Table of Contents

-   [Features](#features)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [Running the Application](#running-the-application)
-   [Usage](#usage)
-   [Functionality Breakdown](#functionality-breakdown)
-   [Contributing](#contributing)
-   [License](#license)
-   [Contact](#contact)

## Features

-   **PDF Resume Processing:** Extracts text from uploaded PDF resumes.
-   **Job Description Analysis:** Analyzes and processes the provided job description.
-   **Resume Validation:** Validates resume length and filters out unsuitable candidates based on word count.
-   **Blacklist Filtering:** Automatically rejects resumes containing specified blacklisted words or phrases.
-   **Keyword Matching:** Matches keywords from resumes with the job description and field-specific keyword banks.
-   **TF-IDF and Cosine Similarity Ranking:** Ranks resumes based on TF-IDF vectorization and cosine similarity, enhanced with field-specific keyword matching scores.
-   **User-Friendly Interface:** Provides an intuitive web interface using Streamlit.
-   **Detailed Analysis:** Offers detailed views of resume content, keyword analysis, and word frequency analysis.
-   **Interactive Visualizations:** Generates interactive bar charts and other visualizations using Plotly.
-   **Downloadable Reports:** Allows users to download ranking results as a CSV file.
-   **Field Specific Keyword Matching:** Improves accuracy by comparing resumes to keywords relevant to specific industry fields.

## Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

-   Python 3.7+
-   pip (Python package installer)

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/Jagguyrie/AI-powered-Resume-Screening-and-Ranking-System ]
    cd [AI-powered-Resume-Screening-and-Ranking-System]
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    ```

    -   On Windows:

        ```bash
        venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the application, execute the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Usage

1.  **Enter Job Description:** Input the job description in the left sidebar.
2.  **Select Job Field:** Choose the appropriate job field for keyword matching.
3.  **Upload Resumes:** Upload PDF resumes.
4.  **Analyze Resumes:** Click the "Analyze Resumes" button.
5.  **Review Results:** View ranking results, detailed resume content, and keyword analysis.

## Functionality Breakdown

  - **`extract_text_from_pdf(file)`:** Extracts text from PDF files using PyPDF2.
  - **`preprocess_text(text)`:** Preprocesses text by converting to lowercase, removing special characters, and normalizing whitespace.
  - **`count_words(text)`:** Counts the number of words in a given text.
  - **`check_blacklisted_words(text)`:** Checks for blacklisted words in the resume text.
  - **`validate_resume_length(text, min, max)`:** Validates the resume length against specified minimum and maximum word counts.
  - **`rank_resumes(job_description, resumes, job_field=None)`:** Ranks resumes using TF-IDF and cosine similarity, enhanced with field-specific keyword matching.
  - **`get_download_link(df, filename)`:** Creates a downloadable link for the ranking results in CSV format.
  - **`generate_wordcloud_data(text)`:** Generates word frequency data for word cloud visualization.
  - The Streamlit application provides the user interface, manages file uploads, processes input, and displays results.

## Contributing

Contributions are welcome\! If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Commit your changes and push to your fork.
5.  Submit a pull request.

## License

This project is licensed under the [MIT License](https://www.google.com/url?sa=E&source=gmail&q=LICENSE).

## Contact

For any questions or inquiries, please contact:

[Jagrut Lade]
[jagguyrie@gmail.com]

```
**Note:** Replace `[repository_url]`, `[repository_directory]`, `[Your Name/Organization]`, and `[Your Email]` with your actual information. Also, create a `requirements.txt` file by running `pip freeze > requirements.txt` in your virtual environment and add a `LICENSE` file if you intend to use a specific license.
```
