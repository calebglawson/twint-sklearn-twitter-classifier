'''This script takes the DB built with gather_data.py and turns it into a machine learning model.'''

import argparse
import sqlite3
from joblib import dump
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


def fetch_data(database):
    '''Retrieve and filter data for numeric columns.'''

    conn = sqlite3.connect(database)
    dataframe = pd.read_sql_query("select * from user_stats;", conn)

    # Preserve the DF for displaying results later.
    bkp = dataframe

    # Remove the non-numeric types from the DF.
    dataframe = dataframe[['following_watchlist', 'watchlist_completion', 'likes_watchlist',
                           'retweets_watchlist', 'mentions_watchlist', 'watchword_in_bio',
                           'is_on_watchlist']]

    conn.close()

    return dataframe, bkp


def tt_split(dataframe):
    '''Split DB data into train and test data sets.'''

    x_tr, x_tst, y_tr, y_tst = train_test_split(dataframe.drop('is_on_watchlist', axis=1),
                                                dataframe['is_on_watchlist'],
                                                test_size=0.25)
    return x_tr, x_tst, y_tr, y_tst


def find_best_params(x_tr, y_tr, n_iter):
    '''Do a randomized search in order to find the best params that fit the model.'''

    print("Finding the best param values, this may take some time.")

    params = {"C": stats.uniform(1, 1000000),
              "gamma": stats.uniform(0.1, 1)}

    mdl = RandomizedSearchCV(SVC(
        probability=True), param_distributions=params, n_iter=n_iter, n_jobs=-1, cv=5, verbose=3)
    mdl.fit(x_tr, y_tr)

    print("Best Params: " + str(mdl.best_params_))

    return mdl


def train_model(c_val, gamma, x_tr, y_tr, n_iter):
    '''Train the model with the user-provided params.'''

    if c_val is None or gamma is None:
        mdl = find_best_params(x_tr, y_tr, n_iter)
    else:
        mdl = SVC(probability=True, C=c_val,
                  gamma=gamma)
        mdl.fit(x_tr, y_tr)

    return mdl


def persist_model_to_disk(mdl, model_output):
    '''Dump the model to a file on the disk to be consumed by predict.py.'''

    dump(mdl, model_output)
    print(f"Model preserved to disk: {model_output}")


def predict(mdl, x_tst):
    '''Run predictions against the test data.'''

    prd = mdl.predict(x_tst)
    prob = mdl.predict_proba(x_tst)[:, 1]

    return prd, prob


def display_stats(prd, y_tst):
    '''Print out the model performance stats to the user.'''

    print("Confusion Matrix")
    print(confusion_matrix(y_tst, prd))
    print("Classification Report")
    print(classification_report(y_tst, prd))


def write_excel(path, dataframe, worksheet, mode):
    '''Write data frame to Excel file.'''

    if not dataframe.empty:
        # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        writer = pd.ExcelWriter(path, mode=mode)
        dataframe.to_excel(writer, worksheet)
        writer.save()


def persist_test_results_to_disk(x_tst, y_tst, prd, prob, bkp, test_output):
    '''Persist the test results to disk.'''

    results = pd.concat([pd.DataFrame(x_tst), pd.DataFrame(y_tst)], axis=1)
    results = results.reset_index()
    results = bkp.loc[results['index']]
    results["pred"] = prd
    results["proba"] = prob

    write_excel(test_output, results, "test results", "w")

    if not results.empty:
        print("Test results saved to: " + test_output)
    else:
        print("No results to output.")


# MAIN
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "database", help="name of the db holding the twitter user stats and labels")
    PARSER.add_argument(
        "--C", help="C value param for SVC", type=float)
    PARSER.add_argument(
        "--gamma", help="gamma value param for SVC", type=float)
    PARSER.add_argument(
        "--n_iter", help="number of iterations for randomized optimal param search",
        type=int, default=50)
    PARSER.add_argument(
        "--model_output", help="filename of the output model", default="model")
    PARSER.add_argument(
        "--test_output", help="filename of the test results", default="test_results")

    ARGS = PARSER.parse_args()

    if ".joblib" not in ARGS.model_output:
        ARGS.model_output = ARGS.model_output + ".joblib"

    if ".xlsx" not in ARGS.test_output:
        ARGS.test_output = ARGS.test_output + ".xlsx"

    DF, DF_BKP = fetch_data(ARGS.database)
    # Feature scaling did not improve performance, so it was not included.
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = tt_split(DF)
    MODEL = train_model(ARGS.C, ARGS.gamma, X_TRAIN, Y_TRAIN, ARGS.n_iter)
    persist_model_to_disk(MODEL, ARGS.model_output)
    PRED, PROBA = predict(MODEL, X_TEST)
    display_stats(PRED, Y_TEST)
    persist_test_results_to_disk(
        X_TEST, Y_TEST, PRED, PROBA, DF_BKP, ARGS.test_output)
