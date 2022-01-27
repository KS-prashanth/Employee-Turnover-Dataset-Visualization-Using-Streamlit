from re import L
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.image("t8.png", width=500, use_column_width="auto")


st.write("## Employee Turnover Classification Using Machine Learning Algorithms")
st.write("### Employee Turnover Dataset")

st.sidebar.image("M.png")
classifier_name = st.sidebar.selectbox(
    "Select classifier",
    ("KNN", "SVM", "Random Forest", "Logestic Regression", "Naive Bayes"),
)


##############################################################################
##############################################################################
##  DATA Cleaning

hr = pd.read_csv("HR.csv")
hr = hr.rename(columns={"sales": "department"})
hr["department"].unique()
hr["department"] = np.where(
    hr["department"] == "support", "technical", hr["department"]
)
hr["department"] = np.where(hr["department"] == "IT", "technical", hr["department"])
hr["department"].unique()
cat_vars = ["department", "salary"]

for var in cat_vars:
    cat_list = "var" + "_" + var
    cat_list = pd.get_dummies(hr[var], prefix=var)
    hr1 = hr.join(cat_list)
    hr = hr1

hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)

cols = [
    "satisfaction_level",  # 0
    "last_evaluation",  # 1
    "time_spend_company",  # 4
    "Work_accident",  # 5
    "promotion_last_5years",  # 7
    "department_RandD",  # 8
    "department_hr",  # 10
    "department_management",  # 11
    "salary_high",  # 16
    "salary_low",  # 17
]

X = hr[cols]
y = hr["left"]

################################################################################################
################################################################################################
hr.drop(hr.columns[[2, 3, 6, 9, 12, 13, 14, 15, 18]], axis=1, inplace=True)
st.dataframe(hr)

st.write("Shape of dataset:", X.shape)
st.write("Number of Features :", len(cols))
st.write("number of Target classes:", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    elif clf_name == "Logestic Regression":

        return 0
    else:
        return 0
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=1,
        )
    elif clf_name == "Logestic Regression":
        clf = LogisticRegression()
    else:
        clf = clf = GaussianNB()
    return clf


clf = get_classifier(classifier_name, params)


#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy =", acc)

###############################################################
##About
# with st.container():
#     st.title("About")
#     st.image("Prashanth.jpg", caption="Karnati Sai Prashanth", width=150)
    

