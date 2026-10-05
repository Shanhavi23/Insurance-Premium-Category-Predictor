# Insurance Premium Category Predictor

An end-to-end machine learning project: a scikit-learn model that predicts a person's **insurance premium category** (Low / Medium / High), served through a **FastAPI** backend and used from a **Streamlit** frontend.

## Live demo

Try the app here: [insurance-premium-category-predictor-1.streamlit.app](https://insurance-premium-category-predictor-1.streamlit.app/)

## How it works

```
Streamlit UI (frontend.py)  --JSON-->  FastAPI (app.py)  -->  model.pkl (scikit-learn pipeline)
```

1. The user enters their details in the Streamlit form.
2. The frontend sends them to the FastAPI `/predict` endpoint.
3. FastAPI validates the input with Pydantic and derives features from it.
4. The trained pipeline predicts the premium category, which is shown in the UI.

## Features and model

**Inputs:** age, weight (kg), height (m), annual income (LPA), smoker (yes/no), city, occupation.

**Engineered features** (computed in the API with Pydantic `computed_field`, the same logic used in the notebook):

| Feature | Logic |
|---|---|
| `bmi` | weight / height² |
| `age_group` | young (<25), adult (<45), middle-aged (<60), senior |
| `lifestyle_risk` | high (smoker and BMI > 30), medium (smoker or BMI > 27), else low |
| `city_tier` | 1 (metro cities), 2 (larger cities), 3 (all others) |

**Model:** `Pipeline` with a `ColumnTransformer` (one-hot encoding for categorical features, passthrough for numeric features) followed by a `RandomForestClassifier`.

**Training data:** `insurance.csv` (100 rows). The model scored 0.90 accuracy on a 20% test split. With a dataset this small, treat that figure as a rough indication rather than a reliable estimate.

## Project structure

```
.
├── app.py             # FastAPI backend with the /predict endpoint
├── frontend.py        # Streamlit frontend
├── ml-model.ipynb     # Feature engineering, training and evaluation
├── model.pkl          # Trained pipeline
├── insurance.csv      # Training data
└── requirements.txt
```

## Getting started

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API (terminal 1)
uvicorn app:app --reload

# 5. Start the frontend (terminal 2)
streamlit run frontend.py
```

- API docs: http://127.0.0.1:8000/docs
- Streamlit app: http://localhost:8501

> `model.pkl` was saved with scikit-learn 1.9.1, which is pinned in `requirements.txt`. Using another version may cause load errors or warnings. To avoid this, re-run `ml-model.ipynb` to regenerate the file.

## API

### `POST /predict`

Request body:

```json
{
  "age": 35,
  "weight": 70.5,
  "height": 1.72,
  "income_lpa": 12.0,
  "smoker": false,
  "occupation": "private_job",
  "city": "Mumbai"
}
```

`occupation` must be one of: `business_owner`, `freelancer`, `government_job`, `private_job`, `retired`, `student`, `unemployed`.

Response:

```json
{ "predicted_category": "Low" }
```

Invalid input returns a `422` error with details.

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"weight":70.5,"height":1.72,"income_lpa":12,"smoker":false,"occupation":"private_job","city":"Mumbai"}'
```

## Limitations

- The dataset is small, so predictions are illustrative and not suitable for real insurance decisions.
- City names are matched exactly (case-sensitive) against the tier lists. Unknown cities default to tier 3.
- Loading a pickle file can run arbitrary code, so only load `model.pkl` files from sources you trust.

## Tech stack

Python, FastAPI, Pydantic, scikit-learn, pandas, Streamlit

## Possible improvements

- Larger dataset, cross-validation and hyperparameter tuning
- Dockerfile with separate API and frontend containers
- Tests for the API with `pytest`
- Configurable API URL in the frontend through an environment variable
