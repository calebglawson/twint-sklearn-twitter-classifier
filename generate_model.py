'''This script takes the DB built with gather_data.py and turns it into a machine learning model.'''

import argparse
import sqlite3
from pathlib import Path
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

    x_train, x_test, y_train, y_test = train_test_split(dataframe.drop('is_on_watchlist', axis=1),
                                                        dataframe['is_on_watchlist'],
                                                        test_size=0.25)
    return x_train, x_test, y_train, y_test


def find_best_params(x_train, y_train, n_iter):
    '''Do a randomized search in order to find the best params that fit the model.'''

    print("Finding the best param values, this may take some time.")

    params = {"C": stats.uniform(1, 1000000),
              "gamma": stats.uniform(0.1, 1)}

    model = RandomizedSearchCV(SVC(
        probability=True), param_distributions=params, n_iter=n_iter, n_jobs=-1, cv=5, verbose=3)
    model.fit(x_train, y_train)

    print(f"Best Params: {model.best_params_}")

    return model


def train_model(c_val, gamma, x_train, y_train, n_iter):
    '''Train the model with the user-provided params.'''

    model = None

    if c_val is None or gamma is None:
        model = find_best_params(x_train, y_train, n_iter)
    else:
        model = SVC(probability=True, C=c_val,
                    gamma=gamma)
        model.fit(x_train, y_train)

    return model


def persist_model_to_disk(model, model_output):
    '''Dump the model to a file on the disk to be consumed by predict.py.'''

    dump(model, model_output)
    print(f"Model preserved to disk: {model_output}")


def predict(model, x_test):
    '''Run predictions against the test data.'''

    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    return pred, proba


def display_stats(pred, y_test):
    '''Print out the model performance stats to the user.'''

    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred))
    print("Classification Report")
    print(classification_report(y_test, pred))


def write_excel(path, dataframe, worksheet, mode):
    '''Write data frame to Excel file.'''

    if not dataframe.empty:
        # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        writer = pd.ExcelWriter(path, mode=mode)
        dataframe.to_excel(writer, worksheet)
        writer.save()


def persist_test_results_to_disk(x_test, y_test, pred, proba, bkp, test_output):
    '''Persist the test results to disk.'''

    results = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    results = results.reset_index()
    results = bkp.loc[results['index']]
    results["pred"] = pred
    results["proba"] = proba

    write_excel(test_output, results, "test results", "w")

    if not results.empty:
        print(f"Test results saved to: {test_output}")
    else:
        print("No results to output.")


def fetch_args():
    '''Fetch arguments with argparse if called from command line.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database", help="name of the db holding the twitter user stats and labels")
    parser.add_argument(
        "--C", help="C value param for SVC", type=float)
    parser.add_argument(
        "--gamma", help="gamma value param for SVC", type=float)
    parser.add_argument(
        "--n_iter", help="number of iterations for randomized optimal param search",
        type=int, default=100)

    args = parser.parse_args()

    return args


def massage(args):
    '''Format filenames for args.'''

    name = Path(args.database).stem
    directory = Path(f"./{name}/")
    args.model_output = directory / f"{name}.joblib"
    args.test_output = directory / f"{name}_test_results.xlsx"

    if args.database.suffix != ".db":
        args.database = args.database / ".db"

    return args


def run(args):
    '''Main function.'''

    args = massage(args)
    dataframe, df_bkp = fetch_data(args.database)
    # feature scaling did not improve performance, so it was not included.
    x_train, x_test, y_train, y_test = tt_split(dataframe)
    model = train_model(args.C, args.gamma, x_train, y_train, args.n_iter)
    persist_model_to_disk(model, args.model_output)
    pred, proba = predict(model, x_test)
    display_stats(pred, y_test)
    persist_test_results_to_disk(
        x_test, y_test, pred, proba, df_bkp, args.test_output)


# MAIN
if __name__ == '__main__':
    ARGS = fetch_args()
    run(ARGS)
