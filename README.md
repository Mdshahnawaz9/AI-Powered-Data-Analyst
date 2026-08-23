# AI-Powered Data Analyst

An AI-powered data analysis web application built using **Python, Streamlit, Pandas, Matplotlib, LangChain, and Google Gemini AI**.

The application allows users to upload CSV files, clean and analyze data, create visualizations, and ask AI-powered questions about their dataset.

## Features

- Upload CSV datasets
- Dataset preview and basic information
- Missing value detection and cleaning
- Duplicate row detection and removal
- Statistical summary of raw and cleaned data
- Outlier detection and visualization
- Interactive data visualizations
  - Bar Chart
  - Line Chart
  - Histogram
  - Scatter Plot
  - Correlation Heatmap
- Download cleaned dataset as CSV
- Ask questions about the dataset using Gemini AI
- AI-generated dataset insights
- AI data cleaning suggestions
- AI chart recommendations
- Error handling for empty or invalid datasets
- Handles datasets with no numeric columns

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- LangChain
- Google Gemini AI
- python-dotenv

## Project Structure

```text
AI-Powered-Data-Analyst/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
└── data/
    ├── normal_dataset.csv
    ├── missing_values.csv
    ├── duplicate_rows.csv
    └── no_numeric_columns.csv
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd AI-Powered-Data-Analyst
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## API Key Setup

Create a `.env` file in the project root directory:

```text
GOOGLE_API_KEY=your_google_gemini_api_key
```

⚠️ Never upload your `.env` file or API key to GitHub.

## Run the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

## Testing Datasets

The `data` folder contains test CSV files for checking different scenarios:

- `normal_dataset.csv` — Normal dataset testing
- `missing_values.csv` — Missing value handling
- `duplicate_rows.csv` — Duplicate detection and removal
- `no_numeric_columns.csv` — No numeric columns handling

## Author

**Mohammed Shahnawaz**

Built using ❤️ with **Python, Streamlit, and Google Gemini AI**.