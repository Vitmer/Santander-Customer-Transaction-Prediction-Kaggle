
# Santander Customer Transaction Prediction (Kaggle)

![Project Banner](https://via.placeholder.com/800x200?text=Santander+Customer+Transaction+Prediction)

Welcome to the **Santander Customer Transaction Prediction** project, a powerful solution to predict customer transaction behavior using advanced machine learning techniques. This project leverages the LightGBM model with feature engineering to extract valuable insights from the dataset, helping Santander Bank better understand their customers.

## 🚀 Project Overview

Santander Customer Transaction Prediction is a Kaggle competition where the goal is to predict whether a customer will make a specific type of transaction. This project includes:

- Data preprocessing and feature engineering.
- Building a robust LightGBM model for prediction.
- Evaluation using metrics like ROC AUC.
- Modularized code with logging and unit tests for better maintainability.

## 🛠️ Features

- **Feature Engineering**: Uses custom feature extraction methods to add "magic" features that enhance the predictive power of the model.
- **LightGBM Classifier**: A gradient boosting model optimized for speed and performance.
- **Cross-Validation**: Implements StratifiedKFold for better model generalization.
- **Logging**: Comprehensive logging to track the model training process.
- **Unit Testing**: Includes tests for data loading, feature extraction, and model training to ensure code reliability.

## 📈 Performance

- **Best ROC AUC Score**: `91.17`

  The model achieved a ROC AUC score of 91.17, indicating high accuracy in predicting customer transaction behavior.

## 📁 Project Structure

```
Santander Customer Transaction Prediction/
│
├── src/
│   ├── main.py               # Main script for training and prediction
│   └── ...                   # Other source files
│
├── tests/
│   ├── test_main.py          # Unit tests for main.py
│   └── ...                   # Other test files
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Files and directories to ignore in Git
├── README.md                 # Project description (You're reading it!)
└── ...
```

## 🛠️ Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/USERNAME/REPOSITORY_NAME.git
   cd Santander Customer Transaction Prediction
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the training script**:
   ```bash
   python src/main.py
   ```

   This will load the dataset, perform feature engineering, train the model, and generate a submission file.

2. **Run tests**:
   To ensure everything is working as expected:
   ```bash
   python -m unittest discover -s tests
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [Your Name](https://github.com/USERNAME)
