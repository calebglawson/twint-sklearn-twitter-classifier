from joblib import dump
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse
import numpy as np
import pandas as pd
import sqlite3


def fetch_data(database):
    conn = sqlite3.connect(database)
    df = pd.read_sql_query("select * from user_stats;", conn)

    # Preserve the DF for displaying results later.
    df_bkp = df

    # Remove the non-numeric types from the DF.
    df = df[['following_watchlist', 'watchlist_completion', 'likes_watchlist', 'retweets_watchlist', 'mentions_watchlist', 'watchword_in_bio',
             'is_on_watchlist']]

    conn.close()

    return df, df_bkp


def tt_split(df):
    x_train, x_test, y_train, y_test = train_test_split(df.drop('is_on_watchlist', axis=1), df['is_on_watchlist'],
                                                        test_size=0.25)
    return x_train, x_test, y_train, y_test


def find_best_params(x_train, y_train, n_iter):
    print("Finding the best param values, this may take some time.")

    params = {"C": stats.uniform(1, 1000000),
              "gamma": stats.uniform(0.1, 1)}

    model = RandomizedSearchCV(SVC(
        probability=True), param_distributions=params, n_iter=n_iter, n_jobs=-1, cv=5, verbose=3)
    model.fit(x_train, y_train)

    print("Best Params: " + str(model.best_params_))

    return model


def train_model(C, gamma, x_train, y_train, n_iter):
    if C == None or gamma == None:
        model = find_best_params(x_train, y_train, n_iter)
    else:
        model = SVC(probability=True, C=C,
                    gamma=gamma)
        model.fit(x_train, y_train)

    return model


def persist_model_to_disk(model, model_output):
    dump(model, model_output)
    print("Model preserved to disk: " + model_output)


def predict(model, x_test):
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    return pred, proba


def display_stats(pred, y_test):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred))
    print("Classification Report")
    print(classification_report(y_test, pred))


def write_excel(path, df, worksheet, mode):
    if len(df.index) > 0:
        # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        writer = pd.ExcelWriter(path, mode=mode)
        df.to_excel(writer, worksheet)
        writer.save()


def persist_test_results_to_disk(x_test, y_test, pred, proba, df_bkp, test_output):
    results = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    results = results.reset_index()
    results = df_bkp.loc[results['index']]
    results["pred"] = pred
    results["proba"] = proba

    write_excel(test_output, results, "test results", "w")

    print("Test results saved to: " + test_output)


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database", help="name of the db holding the twitter user stats and labels")
    parser.add_argument(
        "--C", help="C value param for SVC", type=float)
    parser.add_argument(
        "--gamma", help="gamma value param for SVC", type=float)
    parser.add_argument(
        "--n_iter", help="number of iterations for randomized optimal param search", type=int, default=50)
    parser.add_argument(
        "--model_output", help="filename of the output model", default="model")
    parser.add_argument(
        "--test_output", help="filename of the test results", default="test_results")

    args = parser.parse_args()

    if ".joblib" not in args.model_output:
        args.model_output = args.model_output + ".joblib"

    if ".xlsx" not in args.test_output:
        args.test_output = args.test_output + ".xlsx"

    df, df_bkp = fetch_data(args.database)
    # Feature scaling did not improve performance, so it was not included.
    x_train, x_test, y_train, y_test = tt_split(df)
    model = train_model(args.C, args.gamma, x_train, y_train, args.n_iter)
    persist_model_to_disk(model, args.model_output)
    pred, proba = predict(model, x_test)
    display_stats(pred, y_test)
    persist_test_results_to_disk(
        x_test, y_test, pred, proba, df_bkp, args.test_output)
