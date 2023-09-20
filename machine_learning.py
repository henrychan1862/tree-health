import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

def extract_train_test_data(df, target):
    """functions extract train and test data"""
    # extract features 
    X_nominal = df[target].copy()
    X = pd.get_dummies(data=X_nominal, drop_first=True)

    # extract labels
    Y = df.health.values
    Y_refined = np.array( [0 if i == "Good" else 1 for i in Y] )

    #preview
    print(f"Shape of feature dataset:{X.shape}. Shape of label dataset: {Y_refined.shape}")
    print(f"TProportion of labels: \n {pd.DataFrame(Y_refined).value_counts(normalize=True)}")

    return train_test_split(X, Y_refined, test_size=0.2, random_state=162, stratify=Y)


def dt_model_report(model, x_train, x_test, y_train, y_test):
    """function trains a learner and rate its performance using test set"""
    # classify using decision tree 
    clf = model(random_state=162, class_weight={1:10})
    clf.fit(x_train, y_train)
    print("Confusion matrix: \n {}".format(confusion_matrix(y_test, clf.predict(x_test))))
    print(f"Test f1 score for decision tree model is {f1_score(y_test,clf.predict(x_test)):.3}")

    #obtain feature importances defined by decrease in impurity
    fig = plt.figure(figsize=(10, 5))
    feature_importance_df = pd.DataFrame({"Feature":x_train.columns, "Importance":clf.feature_importances_}).sort_values("Importance", ascending=False)[:10]
    sns.barplot(y="Feature", x="Importance", data=feature_importance_df)
    plt.show()

    return clf

def nb_model_report(model, x_train, x_test, y_train, y_test):
    """function trains a learner and rate its performance using test set"""
    # classify using decision tree 
    clf = model(class_prior =  [1.1, 0.9])
    clf.fit(x_train, y_train)
    print("Confusion matrix: \n {}".format(confusion_matrix(y_test, clf.predict(x_test))))
    print(f"Test f1 score for categorical naive bayes is {f1_score(y_test,clf.predict(x_test)):.3}")

    return clf

def pr_curve(model, x_test, y_test):
    """function plots precision-recall curve"""
    y_score = model.predict_proba(x_test)[:, 1]
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=model.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    return 