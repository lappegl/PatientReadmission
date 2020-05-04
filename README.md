# Early Readmission Prediction
Using machine learning to predict patients' risk of being readmitted within 30 days of their discharge.

Being able to predict a patient's likelihood of being readmitted within 30 days would be helpful in monitoring patients' health, evaluating treatment care, and forecasting future demand on medical resources.

It's a challenging problem to model, as there are so many variables impacting our health that cannot be easily captured--patient habits, living conditions/environment, and virtually every other aspect of daily life.

Nonetheless, we can make use of what medical data we have to begin modeling patient risk and perhaps--at least to some extent--quality of care.


# Contents

* Data
  * Raw data
  * Exploratory analysis
  * Data cleansing / feature engineering

* Machine Learning
  * LightGBM
  * TensorFlow
  * TensorFlow - undersampled data
  * Logistic Regression
  * Logistic Regression - undersampled data

* Conclusion
  * Model evaluation comparisons
  * Final thoughts
  
  
# Data

Head of raw data (click to view full image)
![img](https://i.imgur.com/WiyTHTp.png)

This is the format of the primary file--a CSV in which one row represents a single patient admission (encounter) at a hospital. Admissions are given IDs (encounter_id) and patients have IDs as well.

There are patient demographic/info variables, birth date, sex, gender, ethnicity.

A number of features are actually codes which are defined in a reference table. For example: **admission_type**, **discharge_disposition**, **diag_\***

Over 20 columns are dedicated to noting the role of various medications in the encounter. These columns are named after the drug, and values indicate whether the drug was used, and whether the dosage was steady or increased, decreased, or steady over the duration of the admission.

Finally, the target variable we're predicting is contained in the column **readmitted** which contains one of ["NO", ">30", "<30"]

* **NO** indicates that there was no readmission on record for this patient. That is, this row was the first and last encounter for this patient.

* **\>30** indicates that the patient was readmitted after 30 days of being discharged.

* **<30** indicates that the patient was readmitted within 30 days (this is what we're trying to predict)


![img](https://i.imgur.com/bzWyKDC.png)

![img](https://i.imgur.com/4CQEKrV.png)

![img](https://i.imgur.com/pMOr20I.png)




# Machine Learning

Building predictive models to identify patients at risk of being readmitted within 30 days after discharge.

This is where the fun begins!

While the data is pretty clean in its raw format, there is still a lot of work to be done to prepare it for training any models.

You may have seen a lot of "?" marks in the first 5 rows I showed at the beginning. These signify missing values, and, additionally, different columns have a few different varieties of null value options! To make matters even more complicated, the columns that indicate codes/IDs are actually shown to link to "null" or "missing" descriptions in reference tables!
All these things must be accounted for (and I handle this in the notebook).

One of my favorite parts about machine learning problems is feature engineering / transformation. It's an exercise that requires a great deal of both logic and creativity.

What I found in the data is that encounters with **readmitted** being <30 or \>30, these patients often appeared multiple times throughout the dataset (obviously, because being readmitted implies multiple admissions).

This fact led me to calculating two additional features:

1. time since previous admission

2. number of previous admissions

This was a bit tricky because at each point I had to filter the data to the patient_id and only looking at data prior to whatever encounter I calculated these variables for.

The methods I defined to add these features:
![img](https://i.imgur.com/vRMxtp8.png)


## LightGBM

LightGBM is a very powerful gradient boosting decision-tree-based model that is--as the name suggests--lightweight on memory is very efficient and fast to train.

It can handle imbalanced data and doesn't require data to be one-hot encoded--it can deal with categorical data directly.

It's a go-to of mine, so I figured I'd start here!

Results:

![img](https://i.imgur.com/9dnrOPw.png)

Keep in mind that approximately 88% of the data is NOT labeled as early-readmission.

As such, the null accuracy is ~88%. Even so, LightGBM results show perfect precision but terrible recall. Before taking much else away from this, let's look into other models.


## TensorFlow

Moving on, I transformed categorical features into one-hot-encoded variables:

> X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, stratify=y, test_size=0.2, random_state=18)

Results:
![img](https://i.imgur.com/bBL5G8t.png)

While less accurate than LightGBM, the neural network model has significantly greater recall--at least relative to LightGBM.

## TensorFlow - Resampled

Because 88% of the data is a negative class, I decided to look into undersampling non-early-readmission data in order to present the neural network with an evenly-split training data set.

I used **imblearn** to accomplish this.

Results:

![img](https://i.imgur.com/Pv3yL0J.png)

While less accurate, training on resampled data had a big impact on recall.


## Logistic Regression

Logistic regression is still very popular in a variety of binary classification problems in analytics. I figured it would be appropriate to compare the other models to Logistic Regression.

Results:

![img](blob:https://imgur.com/1a40ee19-25e7-478d-ade5-f3e2ec6548d4)

Pretty good accuracy, but here it falls into the same problem as LightGBM--higher precision but almost worthless as a result of extremely low recall.

## Logistic Regression - Resampled

Just like the neural network model, I wanted to see how under-sampling the training data would impact the Logistic Regression model.

![img](https://i.imgur.com/1mnBGUL.png)

This model has the lowest accuracy we've seen yet, but the highest recall, catching nearly 60% of all early-readmissions.



# Conclusion

## Model Evaluation Comparison

#### Accuracy

![img](https://i.imgur.com/p1h14ob.png)

#### Precision

![img](https://i.imgur.com/yQ2uuL4.png)

#### Recall

![img](https://i.imgur.com/KwYPdyS.png)

#### ROC-AUC

![img](https://i.imgur.com/Js5wHiv.png)

## Discussion and final thoughts

So, which model is best?

Well, it depends. It's never really as simple as pointing to the highest accuracy, precision, recall, ROC-AUC score or false/true positive rates.

The impact of a model's evaluation metrics are dependent on the context of the problem.

In this case, we can imagine that this model would be used by hospitals to assess risk, plan future resource allocation, perform studies on treatment success, etc.

I believe it could come down to a few key issues:

1. What action (if any) is taken when a patient is deemed at-risk? Are the additional expenses allocated to monitoring this patient? (How much might be wasted on false-positives?)
2. What are the consequences of failing to predict earely readmission? Are they at an even greater risk due to a lack of monitoring or attention that might otherwise have been given to them? Is the hospital under-estimating demand of their services (a potential shortage of healthcare capacity)?

These are real matters that must be explored and defined to assign appropriate weight of implications to any model.






