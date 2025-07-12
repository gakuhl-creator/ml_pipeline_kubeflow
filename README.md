# Churn Prediction ML Pipeline

A production-grade machine learning pipeline for customer churn prediction using:

- **Kubeflow** for ML pipeline
- **MLflow** for experiment tracking
- **Scikit-learn** for modeling
- **Docker** containers for each stage

---

## 📦 Features

- Cleanly structured pipeline: `load → preprocess → train → evaluate → predict`
- Tracked and reproducible ML runs via MLflow
- Modular and maintainable Python package layout
- Configurable through a centralized `.env` file
- Ready for local (Kind) or cloud (Kubernetes) deployment

---

## 🚀 Quickstart

### 1. Clone the Repository

### 2. Create a .env file

values should look something like
`RAW_DATA_PATH="path/to/your/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"`

### 3. Create docker images from dockerfiles

I have included a Makefile that creates the docker images from each Dockerfile_* in /dockerfiles. 

To create the Docker images, execute: `make` from project root.

### 4. Consider updating image tags in ml_pipeline() function of pipeline_definition.py

For example, consider this component: `load = load_op(data_path=data_path).set_image('load:latest')`
`set_image()` must feature the correct tag of the docker image you will be using for that pipeline step. You may wish to
tweak these settings depending on your workflow.

---

## 🧠 Pipeline Tasks found in /scripts

| Task            | Description                                      |
|-----------------|--------------------------------------------------|
| `load.py`       | Load and persist raw input data as CSV           |
| `preprocess.py` | Create preprocessing pipeline                    |
| `train.py`      | Train LogisticRegression model                   |
| `evaluate.py`   | Evaluate test data + log metrics to Mlfow + plot |
| `deploy.py`     | Serve prediction endpoint via kserve             |
| `predict.py`    | Predict Churn based on cleaned data              |

Each task is independently runnable and integrated into a Kubeflow stage

---

## 📁 Folder Structure

```
ml_pipeline_kubeflow/
├── data/                     # Input/output data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── dockerfiles/
│   └── Dockerfile_deploy
│   └── Dockerfile_evaluate
│   └── Dockerfile_load
│   └── Dockerfile_predict
│   └── Dockerfile_train
├── requirements/
│   └── requirements-deploy.txt
│   └── requirements-evaluate.txt
│   └── requirements-load.txt
│   └── requirements-predict.txt
│   └── requirements-preprocess.txt
│   └── requirements-train.txt
├── scripts/
│   └── deploy_utils.py
│   └── deploy.py
│   └── evaluate.py
│   └── load.py
│   └── predict.py
│   └── preprocess.py
│   └── train.py
├── .env                      # Create your own
├── .gitignore
├── LICENSE
├── Makefile
├── pipeline_definition.py    # Where all the things connect
├── README.md
└── trigger_pipeline.py       # 
```

---

## 📊 MLflow Tracking

MLflow logs:

- Parameters: `max_iter`
- Metrics: `accuracy`, `precision`, `recall`, `f1`
- Artifacts: model pickle, confusion matrix image

Track at: [http://localhost:5000](http://localhost:5000)

---

## 🧬 Configuration (`.env`)

```python
RAW_DATA_PATH="/path/to/your/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PREDICTION_PATH="/path/to/your/data/predictions.csv"
CLEAN_DATA_PATH="/path/to/your/data/clean_input.csv"
PROCESSSOR_PATH="/path/to/your/data/model_pipeline.pkl"
MODEL_OUTPUT_PATH= "/path/to/your/data/fitted_model_pipeline.pkl"
X_TEST_PATH="/path/to/your/data/X_test.csv"
Y_TEST_PATH="/path/to/your/data/y_test.csv"
CONF_MATRIX_OUTPUT_PATH="/path/to/your/data/conf_matrix.png"
MODEL_NAME="Churn Prediction"
```

---

## Run the API Server

Send a JSON payload such as the following toward  ```/predict```

```JSON
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.35,
  "TotalCharges": 850.5
}
```

... and you will receive a response such the following:

```JSON
{
  "prediction": 1,
  "prediction_english": "Churn",
  "churn_probability": 0.6690679352888962
}
```


## ✅ Next Steps

- TODO prove this works on local Kubeflow Pipeline (using Kind)
- Track data versions and schema drift
- Include Infrasture as Code (IaC) tool
- Address scalability concerns

---

## 🧾 License

MIT License
