# Predictive Modeling for Student Math Score Prediction  

## Overview  

This project focuses on building a **predictive model** to estimate the math scores of students using the **Student Performance** dataset from Kaggle. The model leverages supervised learning algorithms such as **decision trees**, **logistic regression**, and **random forests** to predict student performance based on demographic and educational factors.  

---

## Dataset  

**Source**: [Student Performance Dataset on Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)  

### Key Features:  
- **Gender**: Gender of the student.  
- **Race/Ethnicity**: Group to which the student belongs.  
- **Parental Level of Education**: Education level of the parents.  
- **Lunch**: Type of lunch the student receives (standard/reduced).  
- **Test Preparation Course**: Whether the student completed the course.  
- **Math Score**: Target variable to predict.  

---

## Objectives  

1. Build a predictive model to estimate **math scores** based on other features.  
2. Apply supervised learning algorithms:  
   - **Decision Trees**  
   - **Logistic Regression**  
   - **Random Forests**  
3. Perform **feature selection** to identify the most important predictors.  
4. Evaluate models using metrics:  
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  

---

## Tools & Libraries  

- **Programming Language**: Python  
- **Libraries**:  
  - `pandas` for data manipulation  
  - `numpy` for numerical operations  
  - `matplotlib` and `seaborn` for visualization  
  - `sklearn` for machine learning algorithms and evaluation  

---

## Methodology  

1. **Data Exploration & Preprocessing**:  
   - Cleaned and visualized the dataset to understand relationships between variables.  
   - Encoded categorical features for model compatibility.  

2. **Predictive Modeling**:  
   - Built models using supervised learning algorithms.  
   - Fine-tuned hyperparameters to improve performance.  

3. **Feature Selection**:  
   - Identified key features impacting math scores using feature importance and correlation analysis.  

4. **Model Evaluation**:  
   - Assessed model performance using metrics such as accuracy, precision, recall, and F1 score.  
   - Compared the performance of all three algorithms.  

---

## Results  

- **Feature Importance**:  
  - Key predictors include **test preparation course** and **parental level of education**.  
- **Best Performing Model**: Random Forest achieved the highest accuracy and F1 score.  
- **Insights**:  
  - Students who completed the test preparation course tend to have higher math scores.  
  - Parental education level positively correlates with student performance.  

---

## Installation

1. Install dependencies:  

   ```bash  
   pip install pandas numpy matplotlib seaborn scikit-learn  
   ```  

2. Download the dataset from [Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams).  


## Deliverables  

1. **Jupyter Notebook**: Complete implementation of data preprocessing, modeling, and evaluation.  
2. **Report**: Detailed analysis of results, including feature selection insights and recommendations.  

## References  

- [Student Performance Dataset on Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)  
