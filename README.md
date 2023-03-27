# School Project related to cardiovacular disease

The goal of this project was analyse data related to cardiovascular disease then build a model
to predict future symptoms and classify them as either positive or negative for cardiovascuslar disease.

## EDA
The proects starts with an EDA in the file (explorative_analysis)

Here I have checked the statistics about the data, like how many particpants there are in the study, the values for each column and what they mean.
The number of partipants that are positive and negative for cardiovascular disease. 

I have also visualised some of the statistics related to the dataset for an easy quick understanding of the data.

For ease I choose not to implement any data processing or cleaning in this file since that could have potentially resulted in a file too long with endless scrolling.

## Feature engineering

In this file I have implemented data cleaning, tried to catch the potential outliers in the dataset. Good to note that an outlier is a data point in a dataset that significantly deviates from the other observations.

My target for outliers in this dataset were height and BMI. 

For BMI, I did some reading found that BMIs under 16 and over 42 are extremely high and people within this or rather out of this range could be in need of immediate attention. My end goal is for semeone who seemingly looks or believes to be healthy, can check the presence of cardiovascular disease in their body.

I also checked for outliers within heaight. data points seemed to indicate for example that a participant could be 250 cm tall or even 80 cm tall. While there is nothing wrong with with these different heights, they may tend to affect the overall concept of height within the dataset and may thereby influence the outcome of any machine learning model or models in the dataset. To further strengthen my inclination to my selected limits for height, I checked https://www.scb.se/hitta-statistik/artiklar/2018/varannan-svensk-har-overvikt-eller-fetma/, which is the state authority for statistics to check the average heights of people within Sweden where I believe i will most likely be using my algorithm. From the statistics SCB, I also applied a standard deviation of 10.

The next outliers that looked at were systolic and diastolic blood pressures. I have tried to exclude participants that could very well be alreafy very sick as some blood pressures are as high as 16020 for systolic bloodpressure or 10000 for diastolic bloodpressure. While these could be erroneous, I have no domain expertise to make this conclusion and therefore exclude systolic bloodpressure beyond the range 90 - 200 and diastolic bloodpressures beyond 60 - 140.

After created two subsets from my original dataset, I went on to test different classification model that could predict with better certainty if a new person has the presence of cardiovascular disease.

## Models

For my models I used logistic classification, linear svc classification, KNN classification, and decision tree classification. 

For all the models;

first I split my data into train/validation/test sets.

This train is used for training my chosen models. It is the data on which the model is fit or trained. The model learns the patterns in the data and tries to capture the underlying relationship between the input features and the output labels.

The validation set is used to tune the hyperparameters of the model using a gridsearch and pipeline. Hyperparameters are the settings that control the behavior of the model such as learning rate, regularization strength, or number of hidden layers. 

The test set is used to evaluate the final performance of the model. It is used to estimate how well the model generalizes to unseen data.

From the evaluations of the models, I then choose 3 models to use in voting classification.

## Deployment and prediction

After choosing the model that performed best, the logistic regression, I then created a production file called production_model.py from which i did a prediction for 100 random data points from the entire dataset. The prediction model used here is trained on the entire dataset. 

## Others

The code also contains a deployed version in app.py which is connected to a frontend NextJs application and currently only runs locally while I continue working on its deployment.