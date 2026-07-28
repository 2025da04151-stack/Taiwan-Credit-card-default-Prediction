# Taiwan-Credit-card-default-Prediction

##  Problem Statement
Financial institutions face significant financial losses when credit card clients default on their monthly payments. The objective of this project is to predict whether a credit card client in Taiwan will default on their payment in the upcoming month based on their demographic details, credit limit, and 6-month historical repayment trends.

This is a binary classification problem where the primary goal is to accurately identify potential defaulters early, enabling proactive risk mitigation. The complete solution—spanning data ingestion, preprocessing, predictive modeling, and an interactive web dashboard—is built and deployed as an end-to-end Streamlit application.

### Input features
* Credit Limit
* Demographics(Sex,Education,Marriage,Age)
* Past Repayment Status
* Bill Statement Amounts
* Previous Payments Made

### Output/Target feature
* Default payment in next month(0:Client will not default, 1: Client will default)

## Dataset description

Source - <b>Default of Credit Card Clients</b> from UCI (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

|Property|Description|
|-------|------------|
|Dataset Characteristics|Multivariate|
|Subject Area|Business|
|Associated Tasks|Binary Classification|
|Number of Instances|300000|
|Number of Features|23(Type -Integer,Real)+ID(primary key)+Target feature|
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

#### Some observation from Data Analysis
1. As per the accepatble values in source dataset in UCI, categorical fields has very few erroneous which are outside acceptable range - Marriage(0 with 0.18%), Education(0,5,6 -1.15%). These incorrect data are dropped before processing.
2. LIMIT_BAL shows a slight negative correlation with default which signifies Clients with higher credit limits are slightly less likely to default
3. There is an extremely strong positive correlation among all bill amount features (BILL_AMT1 to BILL_AMT6).Because these features carry redundant information, linear models (like Logistic Regression) may suffer from multicollinearity, so need to drop BILL_AMT2-6 for better results.
4. Repayment Status(PAY_0 to PAY_6) features show the strongest positive correlation with the target variable (default payment next month), with PAY_0 (recent payment status) having the highest positive correlation.



## Github Repository Link
https://github.com/2025da04151-stack/Taiwan-Credit-card-default-Prediction.git

## Models used:
Below 5 classification models are implemented to predict Credit Default. For the Training Testing split, 80:20 stratified split has been used 
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian
5. Ensemble Model - Random Forest

### Models are evaluated based on the below evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)

### Model Evaluation Metrics

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|-------------|--------|---|---------|------|--|---|
|Logistic Regression            |0.8086     |0.7232     |0.7176     |0.2347     |0.3537     |0.3332  |  
|Decision Tree Classifier       |0.8080     |0.7169     |0.6090     |0.3891     |0.4748     |0.3779  |  
|KNN Classifier                 |0.8026     |0.7240     |0.6184     |0.3005     |0.4045     |0.3311  |  
|Naive Bayes Classifier         |0.3682     |0.6836     |0.2481     |0.9023     |0.3892     |0.1246  |  
|Random Forest                  |0.8184     |0.7709     |0.6737     |0.3611     |0.4702     |0.3989  | 

### Observation on Model Performance

|ML Model Name|Observation about model performance|
|-------------|--------|
|Logistic Regression|LLogistic Regression achieved an accuracy of 80.86% with the highest precision (71.76%), meaning its positive predictions were usually correct. However, its AUC (0.7232) sat near the lower end among the better-performing models, and its low recall (23.47%) shows that many actual positive cases were missed. This is reflected in its F1-score (0.3537) and MCC (0.3332), which indicate only moderate overall performance.|
|Decision Tree|Decision Tree Classifier achieved an accuracy of 80.80% and an AUC of 0.7169. By trading off some precision (60.90%), it captured significantly more positive cases, reaching a higher recall (38.91%) than even Random Forest. This improvement in capturing positive cases boosted its F1-score (0.4748) and MCC (0.3779), reflecting a strong overall balance.|
|kNN|KNN Classifier yielded an accuracy of 80.26% and an AUC of 0.7240, performing closely to Logistic Regression and Decision Tree. It maintained decent precision (61.84%) but had a lower recall (30.05%), missing a noticeable portion of positive instances. Consequently, its F1-score (0.4045) and MCC (0.3311) sit firmly in the middle range among the evaluated models.|
|Naive Bayes|Naive Bayes Classifier was an outlier, posting the lowest accuracy (36.82%) and precision (24.81%) due to its strong tendency to over-predict the positive class. While this resulted in an exceptionally high recall (90.23%), the flood of false positives limited its AUC to 0.6836 and severely dragged down its F1-score (0.3892) and MCC (0.1246), making it unreliable.|
|Random Forest (Ensemble)|Random Forest demonstrated the strongest performance across almost every evaluation metric, leading with an accuracy of 81.84% and the highest AUC (0.7709). It maintained a solid precision of 67.37% alongside a good recall (36.11%). This optimal trade-off produced the best overall MCC (0.3989) and a top-tier F1-score (0.4702).|
|Overall Winner for your dataset?|Random Forest (Ensemble) performed the best overall because it achieved the highest accuracy, AUC, F1-score, and MCC, while maintaining a good balance between precision and recall. This makes it the most reliable model for this dataset.|

## Streamlit App Link
