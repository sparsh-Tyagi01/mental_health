# AI Mental Health Early Warning and Smart Intervention

A Streamlit-based machine learning dashboard for mental health risk analysis using tabular survey data (`survey.csv`).

The app includes a complete ML workflow:

1. Input Data
2. Exploratory Data Analysis (EDA)
3. Data Engineering and Cleaning
4. Feature Selection
5. Data Split
6. Model Selection
7. Model Training
8. K-Fold Validation
9. Performance Metrics

It also includes an optional Gemini-powered Smart Intervention tab for personalized guidance generation after risk prediction.

## Project Structure

```text
mental_health/
├── app.py
├── requirements.txt
├── survey.csv
└── README.md
```

## Tech Stack

- Python 3.14 (venv)
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- LangChain + Gemini (`langchain-google-genai`)

## Setup

1. Create and activate virtual environment (if not already active):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## How to Use

1. Upload your CSV from the sidebar (you can use `survey.csv`).
2. Select the target column (default is auto-chosen if available).
3. Explore data in the `Data & EDA` tab.
4. Apply imputation/outlier handling in `Cleaning & Engineering`.
5. Choose modeling features in `Feature Selection`.
6. Train in `Model Training`:
	- Choose model (Logistic Regression or Random Forest)
	- Set test split
	- Set K-Fold value
7. Review metrics in `Performance`:
	- Holdout metrics (accuracy, precision, recall, F1)
	- K-Fold mean/std and per-fold scores
	- Classification report and confusion matrix
8. Use `Smart Intervention`:
	- Enter patient profile
	- Add Gemini API key
	- Generate intervention plan

## Notes

- Model is session-only (no disk persistence).
- If you refresh or restart Streamlit, retrain the model.
- Gemini API key is only needed for intervention text generation; core ML features work without it.

## Requirements

Installed from `requirements.txt`:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- langchain
- langchain-google-genai
- google-generativeai

## License

For educational and project demonstration use.
