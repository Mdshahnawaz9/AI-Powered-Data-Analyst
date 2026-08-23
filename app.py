import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="AI-Powered Data Analyst",
    page_icon="📊",
    layout="wide"
)

with st.sidebar:
    st.title("AI Data Analyst")
    st.write(
        "Upload a CSV file, clean your data, "
        "visualize it, and get AI-powered insights."
    )
    st.divider()

    st.subheader("✨ Features")

    st.write("📁 CSV Upload")
    st.write("🧹 Data Cleaning")
    st.write("📊 Data Visualization")
    st.write("🤖 AI Data Analysis")
    st.write("💡 AI Insights")
    st.write("📈 Chart Recommendations")

    st.divider()

    st.caption(st.caption("Built by Mohammed Shahnawaz | Streamlit + Gemini AI"))

#Load API key from .env file
load_dotenv()

#create Gemini LLm
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

st.title("AI-Powered Data Analyst")
st.write(
    "Upload your CSV file, automatically clean the data, "
    "visualize important patterns, and get AI-powered insights."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your csv file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV file is empty!")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    with col2:
        st.metric("📊 Total Rows", df.shape[0])
    with col3:
        st.metric("📋 Total Columns", df.shape[1])

    st.divider()

    st.subheader("📋 Dataset Preview")
    st.dataframe(df)

    st.subheader("📊 Dataset Statistics")
    col1,col2 = st.columns(2)
    with col1:
        st.metric("Total Rows",df.shape[0])
    with col2:
        st.metric("Total Columns",df.shape[1])

    st.divider()
    st.write("")
#--------------------------------------------------------
    st.subheader("ℹ️ Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Column Names")
        for column in df.columns:
            st.write("•", column)

        st.write("#### Dataset Shape")
        st.write(f"{df.shape[0]} Rows x {df.shape[1]} Columns")

    with col2:
        st.write("#### Data Types")
        st.dataframe(df.dtypes)

        st.metric("Total Missing Values",df.isnull().sum().sum())

    st.subheader("📈 Statistical Summary of Raw Dataset")
    st.dataframe(df.describe())

    st.divider()

# cleaning part starts here
    st.subheader("🧹 Missing Values")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)

    st.subheader("✨ Cleaned Missing Values")
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        st.warning("This dataset has no numeric columns for analysis or visualization.")
        st.stop()
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean()
    )
    st.success("Missing values cleaned successfully!")

    st.subheader("🔁 Duplicates Values")
    duplicates_count = df.duplicated().sum()
    st.write("Number of duplicates rows:",duplicates_count)

    df = df.drop_duplicates()
    st.success("Duplicates rows removed successfully!")

    st.subheader("🧼 Cleaned Dataset")
    st.dataframe(df)

    st.subheader("📊 Dataset Statistics of Cleaned Dataset")
    st.write(df.describe())

    st.divider()
    st.subheader("🔍 Outlier Detection")

    selected_outliers_column = st.selectbox(
        "Select a column to check for outliers",
        numeric_columns,
        key="outliers"
    )
    Q1 = df[selected_outliers_column].quantile(0.25)
    Q3 = df[selected_outliers_column].quantile(0.75)

    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 - 1.5 * IQR

    outliers = df[
        (df[selected_outliers_column]<lower_limit) |
        (df[selected_outliers_column]>upper_limit)
    ]

    st.metric("Total Outliers",len(outliers))
    if len(outliers)>0:
        st.dataframe(outliers)
    else:
        st.success("No outliers found!")

    st.subheader("📦 Outlier Visualization")
    fig , ax = plt.subplots()
    ax.boxplot(df[selected_outliers_column])
    ax.set_title(
        f"Box Plot of {selected_outliers_column}"
    )
    ax.set_ylabel(selected_outliers_column)
    st.pyplot(fig)
    st.info("""
    How to read this Box Plot:

    • The box shows where most of the data values are concentrated.

    • The line inside the box represents the median value.

    • The circles outside the main range represent potential outliers.

    • Outliers are unusual values that are significantly higher or lower than most other values.
    """)

    st.divider()
#------------------------------------------------------------
    st.subheader("🤖 AI Data Cleaning Suggestions")
    cleaning_button = st.button("Get AI Cleaning Suggestion")

    if cleaning_button:
        missing_info = df.isnull().sum().to_string()

        data_context = f"""
        Dataset Columns:
        {df.columns.tolist()}

        Dataset Shape:
        {df.shape}

        Missing Values:
        {missing_info}

        Duplicate Row:
        {df.duplicated().sum()}

        Numeric Columns:
        {numeric_columns.tolist()}
        """

        cleaning_prompt = f"""
        You are an expert data analyst.
        
        Analyze the dataset information below and provide useful
        data cleaning suggestions.

        Dataset Information:
        {data_context}

        Instruction:
        -check if there are missing values
        -Check for duplicate data.
        -Suggest whether outliers should be investigated.
        -Tell whether the dataset looks ready for analysis.
        -Give exactly 4 to 5 short suggestions
        -Use simple and beginner-friendly language.
        -Do not make up information
        """

        with st.spinner("Anlayzing data quality..."):
            response = llm.invoke(cleaning_prompt)

        st.success("Suggestion generated!")

        st.write(response.content)

    st.divider()
#--------------------------------------------------------------
    st.subheader("📈 AI Chart Recommendations")
    chart_button = st.button("Get AI Chart Recommations")
    if chart_button:
        data_context = f"""
        Dataset Columns:
        {df.columns.tolist()}

        Dataset Shape:
        {df.shape}

        Data Types:
        {df.dtypes.to_string()}

        Dataset Summary:
        {df.describe().to_string()}
        """

        chart_prompt = f"""
        You are an expert data analyst.

        Analyst the dataset information and recommend the most
        suitable charts for visualizing this dataset.

        Dataset Information:
        {data_context}

        Instructions:
        
        - Recommend exactly 5 useful charts.
        - mention which column or columns should be used.
        - Explain briefly why that chart is useful.
        - Use simple and beginner-friendly language.
        - Only recommend charts that make sense for the available dataset.
        - Do not make up columns that do not exist.

        Format:

        1. Chart Name
            Columns: ...
            Reason: ...
        """
        with st.spinner("Finding the best charts..."):
            response = llm.invoke(chart_prompt)

        st.success("Chart recommendations generated!")
        st.write(response.content)

    st.divider()
#------------------------------------------------------------
    st.subheader("📊 Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bar Chart")

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns,
            key="bar"
        )

        st.bar_chart(df[selected_column])
    with col2:
        st.subheader("Line Chart")
        selected_line_column = st.selectbox(
            "Select a numeric column for line chart",
            numeric_columns,
            key= "line"
        )
        st.line_chart(df[selected_line_column])

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Histogram")
        selected_hist_column = st.selectbox(
            "Select a numeric column for histogram",
            numeric_columns,
            key="histogram"
        )
        fig , ax = plt.subplots()
        ax.hist(df[selected_hist_column])
        st.pyplot(fig)

    with col4:
        st.subheader("Scatter Plot")
        x_column = st.selectbox(
            "Select X-axis column",
            numeric_columns,
            key="scatter_x"
        )

        y_column = st.selectbox(
            "Select Y-axis column",
            numeric_columns,
            key="scatter_y"
        )
        st.scatter_chart(
            df,
            x =x_column,
            y = y_column
        )

    st.subheader("Correlation Heatmap")
    correlation = df[numeric_columns].corr()
    fig,ax = plt.subplots()
    image = ax.imshow(correlation)
    fig.colorbar(image)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(correlation.columns, rotation=45)
    ax.set_yticklabels(correlation.columns)
    #values show inside the boxes
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            ax.text(
                j,
                i,
                round(correlation.iloc[i,j],2),
                ha='center',
                va='center'
            )

    st.pyplot(fig)
    st.info("""
        Correlation Values:

        +1 → Strong Positive Relationship
        0 → No Strong Relationship
        -1 → Strong Negative Relationship

        Example:
        0.90 means both values usually increase together.
        -0.80 means when one increases, the other usually decreases.
    """)

    st.divider()
#-------------------------------------------------------------
#Download Cleaned Dataset code
    st.subheader("⬇️ Download Cleaned Dataset")
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    st.divider()

#------------------------------------------------------------
    st.subheader("💬 Ask Questions About Your Data")
    st.write("Try these questions: ")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is the average of each column?"):
            st.session_state.question = "What is the average of each numeric column?"

    with col2:
        if st.button("Which column has the highest value?"):
            st.session_state.question = "Which column has the highest maximum value?"

    with col3:
        if st.button("Give me a dataset summary"):
            st.session_state.question = "Give me a simple summary of this dataset."

    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input(
        "Ask a question about your dataset",
        key="question"
    )
    analyze_button = st.button("Analyze")


    if analyze_button:
        if question:
            data_context = f"""
            Dataset Columns:
            {df.columns.tolist()}

            Dataset Shape:
            {df.shape}

            Dataset summary:
            {df.describe().to_string()}

            Sample Data:
            {df.head().to_string()}
            """

            prompt = f"""
            You are an expert data analyst.

            Analyze the dataset carefully and answer the user's question.

            Dataset Information:
            {data_context}

            User Question:
            {question}

            Instructions:
            - Answer only based on the provided dataset.
            - Explain the answer in simple and beginner-friendly language.
            - If possible, include relevant numbers or statistics.
            - Do not make up information that is not available in the dataset.
            - Keep the answer clear and concise.
            """

            with st.spinner("Analyzing your data...."):
                response = llm.invoke(prompt)

            st.success("Analysis completed")

            st.write(response.content)
        else:
            st.warning("Please enter a question first!")

    st.divider()
    st.subheader("💡 AI-Generated Insights")
    insights_button = st.button("Generate Insights")

    if insights_button:
        data_context= f"""
        Dataset Columns:
        {df.columns.tolist()}

        Dataset Shape:
        {df.shape}

        Dataset Summary:
        {df.describe().to_string()}

        Sample Data:
        {df.head().to_string()}
        """

        insight_prompt = f"""
        You are an expert data analyst.

        Analyze the following dataset and provide 5 useful insights.

        Dataset:
        {data_context}

        Instructions:
        - Give exactly 5 insights.
        - Use simple and beginner-friendly language.
        - Include numbers or statistics when relevant.
        - Do not make up information.
        - Format the response using numbered points.
        """

        with st.spinner("Generating insights..."):
            response = llm.invoke(insight_prompt)

        st.success("Insights generated!")

        st.write(response.content)