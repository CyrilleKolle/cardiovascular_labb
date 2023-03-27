import pandas as pd
import joblib

test_samples = pd.read_csv("./data/test_samples.csv")
logistic_regression_model = joblib.load("./models/model.pkl")
explanatory_variables, response_variable = test_samples.drop('cardio', axis=1), test_samples['cardio']

# Predict the outcome probabilities for each class
outcome_probabilities = logistic_regression_model.predict_proba(explanatory_variables)

# Select the probability values for class 1
class1_probabilities = outcome_probabilities[:, 1]

# Create a DataFrame with the prediction indices, predicted probabilities of class 0 and class 1
threshold = 0.5
dataframe = pd.DataFrame({'prediction': explanatory_variables['id'].apply(lambda x: f'Id: {x:03}'),
                          'probability_class0': outcome_probabilities[:, 0],
                          'probability_class1': class1_probabilities})

# Add a new column with the predicted result based on the threshold
dataframe['predicted_result'] = ['probability class 1' if p >= threshold else 'probability class 0' for p in class1_probabilities]

# Print the first few rows of the DataFrame
print(dataframe.head())

# Save the dataframe to a CSV file
dataframe.to_csv('./data/prediction.csv', index=False)
