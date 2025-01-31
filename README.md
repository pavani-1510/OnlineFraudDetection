# Credit Card Fraud Detection Using Machine Learning and Genetic Algorithm for Feature Selection

## Overview
Credit card fraud poses a significant threat to financial security, affecting millions of transactions globally. This project aims to develop an effective fraud detection system by leveraging machine learning techniques combined with Genetic Algorithm (GA) for optimal feature selection. The objective is to improve detection accuracy, reduce false positives, and provide a robust fraud prevention mechanism.

## Team Members
- **M Sahitya** (2200031651)
- **M Jhansi** (2200031441)
- **R Pavani** (2200033283)

## Problem Statement
Traditional fraud detection methods often struggle with accuracy due to the highly imbalanced nature of fraud datasets, where fraudulent transactions are rare compared to legitimate ones. This project addresses these challenges using advanced machine learning models and feature selection techniques to enhance fraud detection performance.

## Technologies Used
- **Amazon SageMaker** for model training and deployment
- **Python** for data processing and analysis
- **Scikit-learn** and **TensorFlow** for machine learning model implementation
- **Genetic Algorithm (GA)** for feature selection
- **AWS** for cloud computing and scalability

## Dataset
The dataset used consists of anonymized credit card transactions with a significant class imbalance. Advanced resampling and feature engineering techniques are applied to ensure accurate fraud identification.
[Onlin Fraud Detection Dataset](https://www.kaggle.com/datasets/rizwanash/onlinefraud/data)

## Machine Learning Models Implemented
- **Decision Tree**
- **Random Forest**
- **Logistic Regression**
- **Artificial Neural Networks (ANN)**
- **Naive Bayes**

## Approach
1. **Data Preprocessing**
   - Handling missing values and outliers
   - Feature scaling and transformation
2. **Feature Selection using Genetic Algorithm (GA)**
   - Optimizing input variables to enhance classifier performance
3. **Model Training & Evaluation**
   - Comparing different machine learning models
   - Measuring accuracy, precision, recall, and F1-score
4. **Deployment with Amazon SageMaker**
   - Deploying the best-performing model to detect fraudulent transactions in real-time

## Results
The proposed approach using GA-based feature selection significantly enhances model efficiency and detection accuracy, leading to a more reliable fraud detection system for real-world financial applications.

## References
- [IEEE Paper on Fraud Detection](https://ieeexplore.ieee.org/document/10493954)
- [ScienceDirect Research Article](https://www.sciencedirect.com/science/article/pii/S2666827024000793)
- [Fraud Detection with AWS SageMaker](https://aws.amazon.com/blogs/machine-learning/detect-fraudulent-transactions-using-machine-learning-with-amazon-sagemaker/)
- [IJCRT Research Paper](https://www.ijcrt.org/papers/IJCRT2408004.pdf)

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Deploy the model using SageMaker:
   ```bash
   python deploy.py
   ```

