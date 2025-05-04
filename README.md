# A Comparative Study of Sentiment Analysis Techniques in Artificial Intelligence

This project compares the performance of three sentiment analysis techniques: **Multilayer Perceptron (MLP)**, **Long Short-Term Memory (LSTM)**, and **Logistic Regression**. The models are trained and evaluated on a Twitter dataset, and their performance is optimized using a Genetic Algorithm.

## Project Overview

The project involves the following steps:

1. **Data Loading and Exploration:** The Twitter dataset is loaded and explored to understand its structure and content.
2. **Data Cleaning and Preprocessing:** The data is cleaned by removing irrelevant characters, handling missing values, and converting categorical features into numerical representations.
3. **Model Implementation:** Three sentiment analysis models (MLP, LSTM, and Logistic Regression) are implemented using Python libraries like scikit-learn and TensorFlow.
4. **Model Optimization:** The performance of each model is optimized using a Genetic Algorithm to find the best hyperparameter settings.
5. **Model Evaluation and Comparison:** The models are evaluated using metrics like accuracy, precision, recall, F1-score, and ROC AUC. The results are compared to determine the best-performing model.

## Dataset

The project uses the "[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)" dataset available on Kaggle. This dataset contains tweets labeled with their corresponding sentiment (positive, negative, or neutral).

## Requirements

To run this project, you will need the following libraries:

- Python 3.7 or higher
- pandas
- scikit-learn
- TensorFlow
- DEAP (for Genetic Algorithm)
- matplotlib
- numpy
- re

You can install these libraries using pip:
 
```bash
pip install pandas scikit-learn tensorflow deap matplotlib numpy
```

## Usage

1. Clone this repository to your local machine.
2. Download the dataset from Kaggle and place it in the project directory.
3. Open the Jupyter Notebook and run the cells to execute the code.
4. The results of the model evaluation and comparison will be displayed in the notebook.

## Results

The results of the project show that the optimized LSTM model achieved the highest accuracy, followed by the optimized Logistic Regression model and the optimized MLP model.

## Conclusion

This project demonstrates the effectiveness of different sentiment analysis techniques and the benefits of using a Genetic Algorithm for model optimization.

## Authors

- Amna Bubshait (A00723)
- Athbi Salmeen (A01195)
- Zahra Alshehabi (A00755)

## Acknowledgments

- Dr. Bara Alalawsh (Professor)
- American University of Bahrain (AUBH)
