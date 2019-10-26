import argparse
from joblib import load
import pandas as pd
import sqlite3


def load_model(model_filename):
    model = load(model_filename)
    return model


def fetch_data(database):
    conn = sqlite3.connect(database)
    df = pd.read_sql_query("select * from user_stats;", conn)

    df_bkp = df
    df = df[['following_watchlist', 'watchlist_completion', 'likes_watchlist', 'retweets_watchlist',
             'mentions_watchlist', 'tweet_watchword_ratio', 'neg', 'neu', 'pos', 'compound', 'watchword_in_bio']]

    conn.close()

    return df, df_bkp


def get_predictions(df, model):
    pred = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    return pred, proba


def output_results(df_bkp, pred, proba, output_filename):
    results = pd.DataFrame(df_bkp)
    results["pred"] = pred
    results["proba"] = proba
    results = results.sort_values(
        by=['proba'], ascending=False)

    results.to_csv(output_filename)

    print("Predictions saved to: " + output_filename)

# MAIN


parser = argparse.ArgumentParser()

parser.add_argument(
    "database", help="name of the db holding the twitter user stats")
parser.add_argument(
    "model_filename", help="filename of the joblibbed model")
parser.add_argument(
    "--output_filename", help="filename of the predicted output", default="results.csv")

args = parser.parse_args()

if ".csv" not in args.output_filename:
    args.output_filename = args.output_filename + ".csv"

df, df_bkp = fetch_data(args.database)
model = load_model(args.model_filename)
pred, proba = get_predictions(df, model)
output_results(df_bkp, pred, proba, args.output_filename)
