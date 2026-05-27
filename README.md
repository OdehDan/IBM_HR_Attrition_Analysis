👥 IBM HR Employee Attrition Analysis

 A comprehensive Data Science project analysing employee attrition patterns using SQL, Machine Learning, and Clustering techniques to help organisations proactively identify and retain at-risk employees.

📌 Project Overview
Employee attrition is one of the most costly challenges facing organisations today — replacing a single employee can cost between 50% to 200% of their annual salary. This project leverages the IBM HR Employee Attrition dataset to uncover the key drivers of voluntary employee turnover and build predictive models that help HR departments take proactive retention action.

The dataset contains 1,470 employee records across 35 features covering demographics, compensation, job satisfaction, tenure, and work environment — making it a rich foundation for both exploratory analysis and machine learning.

🎯 Objectives
- Perform structured data exploration using SQL queries
- Clean and preprocess the dataset through rigorous data quality checks
- Engineer meaningful features that capture deeper patterns in employee behaviour
- Segment employees into meaningful personas using KMeans Clustering
- Build and evaluate classification models to predict attrition
- Identify the most influential features driving employee turnover
- Improve model performance through threshold tuning to maximise Recall

🗂️ Dataset
- Source: IBM HR Analytics Employee Attrition & Performance (publicly available)
- Records: 1,470 employees
- Features: 35 original columns covering demographics, compensation, satisfaction scores, and tenure
- Target Variable: `Attrition` (Yes = employee left, No = employee stayed)
- Class Distribution: 84% stayed, 16% left — imbalanced dataset

🛠️ Tools & Libraries
| Tool | Purpose |

| Python | Primary programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| SQLite + Pandas SQL | Structured data querying |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib & Seaborn | Data visualisation |
| Jupyter Notebook | Development environment |
| KaggleHub | Direct dataset import from Kaggle |

---
🔍 Project Workflow
1. SQL Exploration
Before any Python analysis, the dataset was pushed to a SQLite database and explored using structured queries covering:
- Average monthly income by Education Field
- Attrition based on Average Monthly Income, Average years at the company and average age
- Average years since last promotion based on Job Role

2. Data Cleaning & Quality Checks
A three-layer missing value detection approach was applied:

- Layer 1: `isnull().sum()` — detected actual NaN values
- Layer 2: Unique value inspection — identified disguised missing values in categorical columns
- Layer 3: Zero count analysis — identified suspicious zeros in numerical columns

Key findings:
- Three constant columns dropped: `Over18`, `EmployeeCount`, `StandardHours`
- `NumCompaniesWorked` zeros replaced with median — 197 employees with 9+ working years showing 0 companies was not realistic
- `TotalWorkingYears` zeros retained — confirmed as valid fresh graduates aged 18–19 at JobLevel 1

3. Feature Engineering
Three new features were created to capture deeper patterns:

| Feature | Formula | Insight |

| `WorkingExperience` | Binned TotalWorkingYears | Entry Level, Mid Level, Experienced, Veteran career stages |
| `IncomePerExperienceYear` | MonthlyIncome / (TotalWorkingYears + 1) | Measures compensation fairness relative to experience |
| `MonthlyIncome_per_JobLevel` | MonthlyIncome / JobLevel | Measures compensation fairness relative to seniority |
| `TenureRatio` | YearsAtCompany / (TotalWorkingYears + 1) | Measures year spent relative to total working years |


4. Encoding & Scaling

| Feature Type | Encoding Method | Justification |

| WorkingExperience | LabelEncoder | Has a meaningful order — Entry to Veteran |
| Department, JobRole, MaritalStatus, EducationField, BusinessTravel | OneHotEncoder | No natural order between categories |
| Gender, OverTime, Attrition | OneHotEncoder | Binary categories |

Features were scaled using StandardScaler — chosen over MinMaxScaler due to the presence of outliers in compensation and tenure columns.

5. KMeans Clustering — Employee Persona Segmentation

KMeans was run at three different K values to demonstrate how employee personas evolve with increasing granularity:

Choosing Optimal K
- Elbow Method → suggested K=4 (point of maximum curvature)
- Silhouette Score → highest at K=2 (0.153)
- Final K selected: K=4 — provides more actionable HR personas while being supported by the elbow method

PCA Visualisation
Principal Component Analysis (PCA) was applied to reduce the feature space to 2 dimensions for visual cluster inspection.

K=4 Cluster Profiles & Personas

| Cluster | Persona Name | Attrition Rate |

| 0 | Tenured High Level Professionals | Low |
| 1 | Balanced Career Progressors | Moderate |
| 2 | Low Income Early Career Drifters | Highest — 33.08% |
| 3 | Experienced but Undervalued Staff | Moderate-High |

6. Classification Models

Three classification models were trained and evaluated to predict employee attrition:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation Strategy
Due to the 84/16 class imbalance, accuracy was deprioritised in favour of:
- Recall — primary metric; catching actual leavers is the most critical business need
- F1 Score — secondary metric; balances precision and recall

Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

| Baseline LR | 0.867 | 0.654 | 0.362 | 0.466 | 0.817 |
| LR + Threshold 0.3 | 0.850 | 0.532 | 0.532 | 0.532 | 0.817 |
| LR + SMOTE | 0.779 | 0.385 | 0.638 | 0.480 | 0.797|
| Baseline RF | 0.827 | 0.300 | 0.064 | 0.105 | 0.748 |
| RF + Threshold 0.2 | 0.776 | 0.373 | 0.596 | 0.459 | 0.748 |
| RF + SMOTE | 0.820 | 0.421 | 0.340 | 0.376 | 0.786 |
| Baseline DT | 0.844 | 0.526 | 0.213 | 0.303 | 0.713 |
| DT + Threshold 0.2 | 0.776 | 0.373 | 0.596 | 0.459 | 0.713 |
| DT + SMOTE | 0.799 | 0.342 | 0.277 | 0.306 | 0.677

Final Deployed Model
Logistic Regression with Threshold 0.3 — highest Recall of 0.532 and ROC-AUC of 0.817 across all nine configurations.

Threshold Tuning
Default classification threshold of 0.5 was tuned to improve Recall — reducing costly False Negatives (missed leavers).

Feature Importance
Post-training feature importance analysis identified the most influential drivers of attrition, providing actionable insights for HR intervention strategies.

## 📊 Key Findings

1.Overtime is the strongest attrition signal — employees working overtime leave at nearly 3x the rate of those who do not (30.53% vs 10.44%)
2.Sales department has the highest attrition rate at 20.63% despite not having the most employees
3.Low job satisfaction employees leave at 22.84% — nearly double the rate of very high satisfaction employees (11.33%)
4.Low Income Early Career Drifters carry the highest attrition rate of 33.08% — more than double the dataset average
5.Compensation inequity — Experienced but Undervalued Staff are older but earn less than peers, making them a hidden flight risk
6.K=4 clustering revealed four distinct employee personas with meaningfully different attrition risk profiles

💡 HR Recommendations

- Target retention budgets at Cluster 2 — Low Income Early Career Drifters are leaving at the highest rate
- Address compensation inequity in Cluster 3 — experienced employees being paid below their experience level will quietly exit
- Reduce overtime dependency — the strongest single predictor of attrition
- Accelerate promotion timelines — recent promotions alone are insufficient
- Invest in manager quality — short tenure with current manager consistently appears in high attrition profiles


📚 Skills Demonstrated

- Structured data querying with SQL
- Data cleaning and quality assurance
- Feature engineering and domain-driven analysis
- Unsupervised learning — KMeans clustering and persona building
- Supervised learning — classification and model evaluation
- Imbalanced dataset handling
- Threshold tuning for business-specific optimisation
- Data visualisation and insight communication


👥 Team Members

1)DANIEL OFUKOWOICHO ODEH

2)ESSANGENYI BRIGHT VICTOR

3)UWEH BLESSING

4)SUNDAY GLORIA AUDU

5)RACHEL OLUWATOBILOBA ODEJOBI

6)ADEYEMO OREOLUWA VICTORIA

7)KEHINDE KANYINSOLA TAIWO

8)UBONG KINGSLEY UFOT

9)OGUNSOLA IYABO NGOZI

10)AMUSA QUDUS KEHINDE



🏫 Programme

TS Academy — Data Science Training Programme
Instructor: Chukwujekwu Anah



📄 License

This project is open source and available under the [MIT License](LICENSE).

