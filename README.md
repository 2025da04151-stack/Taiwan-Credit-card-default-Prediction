# Taiwan-Credit-card-default-Prediction

#  Problem Statement
Financial institutions face significant financial losses when credit card clients default on their monthly payments. The objective of this project is to predict whether a credit card client in Taiwan will default on their payment in the upcoming month based on their demographic details, credit limit, and 6-month historical repayment trends.

This is a binary classification problem where the primary goal is to accurately identify potential defaulters early, enabling proactive risk mitigation. The complete solution—spanning data ingestion, preprocessing, predictive modeling, and an interactive web dashboard—is built and deployed as an end-to-end Streamlit application.

Input features
* Credit Limit
* Demographics(Sex,Education,Marriage,Age)
* Past Repayment Status
* Bill Statement Amounts
* Previous Payments Made

Output/Target feature
* Default payment in next month(0:Client will not default, 1: Client will default)

# Dataset description

Source - <b>Default of Credit Card Clients</b> from UCI (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

|Property|Description|
|-------|------------|
|Dataset Characteristics|Multivariate|
|Subject Area|Business|
|Associated Tasks|Binary Classification|
|Number of Instances|300000|
|Number of Features|23(Type -Integer,Real)|
|Target Feature|default payment next month[1=Default(Instance%-22.12%),0=Not Default(Instance%-77.88)]|

### Features

|Feature|Description|Demographic|Missing/Null values|Acceptable values|
|-------|--------|--------|--------|------------|
|ID|Unique Identifier||No|Integer|
|LIMIT_BAL|Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.||No|Integer|
|SEX|Gender of  Credit card holder|Yes|No|1 = male; 2 = female|
|EDUCATION|Educational Qualification of Card holder|Yes|No|1 = graduate school; 2 = university; 3 = high school; 4 = others|
|MARRIAGE|marital Status of Card holder|Yes|No|1 = married; 2 = single; 3 = others|
|AGE|Age of Card holder in years|Yes|No|Integer|
|PAY_0 to PAY_6|History of past payment-Repayment status(from April to September, 2005) in reverse order||No|-1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.|
|BILL_AMT1 to BILL_AMT6|Amount of bill statement (NT dollar) in reverse order of month((from April to September, 2005)||No|Integer|
|PAY_AMT1 to PAY_AMT6|Amount of previous payment (NT dollar) reverse order of month(from April to September, 2005)||No|Integer|
|default payment next month|Target variable||No|1=Default,0=Not Default|


# Github Repository Link
https://github.com/2025da04151-stack/Taiwan-Credit-card-default-Prediction.git

# Models used:
Below 5 classification models are implemented to predict Credit Default. For the Training Testing split, 80:20 stratified split has been used 
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian
5. Ensemble Model - Random Forest

Models are evaluated based on the below evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)

### Model Evaluation Metrics

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|-------------|--------|---|---------|------|--|---|
Logistic Regression            |0.8068     |0.7063     |0.6826     |0.2366     |0.3514     |0.3204 |   
Decision Tree Classifier       |0.8087     |0.7247     |0.6201     |0.3482     |0.4459     |0.3619  |  
KNN Classifier                 |0.8075     |0.7248     |0.6352     |0.3044     |0.4116     |0.3435   | 
Naive Bayes Classifier         |0.4133     |0.6593     |0.2487     |0.8176     |0.3814     |0.1082    |
Random Forest (Ensemble)                 |0.8173     |0.7691     |0.6624     |0.3549     |0.4622     |0.3898|

### Observation on Model Performance

|ML Model Name|Observation about model performance|
|-------------|--------|
|Logistic Regression|Logistic Regression achieved an accuracy of 80.68% with the highest precision (68.26%), meaning its positive predictions were usually correct. However, its AUC (0.7063) was the lowest among the better-performing models, and the low recall (23.66%) shows that many actual positive cases were missed. This is reflected in its F1-score (0.3514) and MCC (0.3204), which indicate only moderate overall performance.|
|Decision Tree|The Decision Tree slightly improved the accuracy (80.87%) and AUC (0.7247) compared to Logistic Regression. Its precision (62.01%) and recall (34.82%) were more balanced, resulting in a better F1-score (0.4459). The MCC (0.3619) also suggests that the model classified both classes more consistently.|
|kNN|The kNN model achieved an accuracy of 80.75% with an AUC of 0.7248, which was almost identical to the Decision Tree. Although its precision (63.52%) remained good, the recall (30.44%) indicates that some positive cases were still missed. This led to an F1-score of 0.4116 and an MCC of 0.3435, showing acceptable but average overall performance.|
|Naive Bayes|Naive Bayes produced the highest recall (81.76%), so it identified most positive cases. However, its accuracy (41.33%), AUC (0.6593), and precision (24.87%) were much lower than the other models, indicating a high number of false positives. As a result, its F1-score (0.3814) and MCC (0.1082) remained low, making it the weakest overall model despite its high recall.|
|Random Forest (Ensemble)|Random Forest achieved the highest accuracy (81.73%), AUC (0.7691), F1-score (0.4622), and MCC (0.3898), showing the strongest overall performance. Although its precision (66.24%) was slightly lower than Logistic Regression, the better recall (35.49%) gave it a more balanced performance and improved its ability to identify positive cases without sacrificing overall reliability.|
|Overall Winner for your dataset?|Random Forest (Ensemble) performed the best overall because it achieved the highest accuracy, AUC, F1-score, and MCC, while maintaining a good balance between precision and recall. This makes it the most reliable model for this dataset.|

# Streamlit App Link
