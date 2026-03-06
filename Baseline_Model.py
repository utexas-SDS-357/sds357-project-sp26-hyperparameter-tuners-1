# imports 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# load dataset
df = pd.read_csv("/Users/vidhisapru/Downloads/df_with_dummy_final.csv")

# drop na values 
df = df.dropna(subset=["subject_age", "WealthIndex", "outcome_encd"])

# outcome variable from df 
y = df["outcome_encd"]

# predictors 
X = df[[
    "subject_age",
    "WealthIndex",
    "sex_2.0",
    "race_2.0",
    "race_3.0",
    "race_4.0",
    "race_5.0",
    "race_6.0"
]]

# INTERACTIONS

# wealth x age
X["wealth_age"] = df["WealthIndex"] * df["subject_age"]

# wealth x sex
X["wealth_sex"] = df["WealthIndex"] * df["sex_2.0"]

# wealth x race
X["wealth_race2"] = df["WealthIndex"] * df["race_2.0"]
X["wealth_race3"] = df["WealthIndex"] * df["race_3.0"]
X["wealth_race4"] = df["WealthIndex"] * df["race_4.0"]
X["wealth_race5"] = df["WealthIndex"] * df["race_5.0"]
X["wealth_race6"] = df["WealthIndex"] * df["race_6.0"]

# multinomial logistic regression
model = LogisticRegression(solver="lbfgs", max_iter=3000)
print("Starting model fitting...")
model.fit(X, y)
print("model done")

# coeff 
coef_table = pd.DataFrame(model.coef_, columns=X.columns)
print(coef_table)

# intercepts
print("Intercepts:")
print(model.intercept_)

# accuracy 
print("Training accuracy:", model.score(X,y))

# confusion matrix 
y_pred = model.predict(X)
print(confusion_matrix(y, y_pred))

# precision, recall, F1
print(classification_report(y, y_pred))