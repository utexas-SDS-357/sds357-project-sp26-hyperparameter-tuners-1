# imports 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# 80/20 split with stratification on outcome
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# INTERACTIONS

# interaction term train - age
X_train["wealth_age"] = X_train["WealthIndex"] * X_train["subject_age"]

# interaction term train - sex
X_train["wealth_sex"] = X_train["WealthIndex"] * X_train["sex_2.0"]

# interaction term train - race
X_train["wealth_race2"] = X_train["WealthIndex"] * X_train["race_2.0"]
X_train["wealth_race3"] = X_train["WealthIndex"] * X_train["race_3.0"]
X_train["wealth_race4"] = X_train["WealthIndex"] * X_train["race_4.0"]
X_train["wealth_race5"] = X_train["WealthIndex"] * X_train["race_5.0"]
X_train["wealth_race6"] = X_train["WealthIndex"] * X_train["race_6.0"]

# interaction term test - age 
X_test["wealth_age"] = X_test["WealthIndex"] * X_test["subject_age"]

# interaction term test - sex
X_test["wealth_sex"] = X_test["WealthIndex"] * X_test["sex_2.0"]

# interaction term test - race
X_test["wealth_race2"] = X_test["WealthIndex"] * X_test["race_2.0"]
X_test["wealth_race3"] = X_test["WealthIndex"] * X_test["race_3.0"]
X_test["wealth_race4"] = X_test["WealthIndex"] * X_test["race_4.0"]
X_test["wealth_race5"] = X_test["WealthIndex"] * X_test["race_5.0"]
X_test["wealth_race6"] = X_test["WealthIndex"] * X_test["race_6.0"]

# multinomial logistic regression
model = LogisticRegression(solver="lbfgs", max_iter=3000, class_weight="balanced")
print("Starting model fitting...")
model.fit(X_train, y_train)
print("model done")

# coeff 
coef_table = pd.DataFrame(model.coef_, columns=X_train.columns)
print(coef_table)

# intercepts
print("Intercepts:")
print(model.intercept_)

# accuracy 
print("Training accuracy:", model.score(X_test,y_test))

# confusion matrix 
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# precision, recall, F1
print(classification_report(y_test, y_pred))