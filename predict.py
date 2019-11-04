import argparse
from joblib import load
import pandas as pd
import sqlite3


def load_model(model):
    model = load(model)
    return model


def fetch_data(database):
    conn = sqlite3.connect(database)
    df = pd.read_sql_query("select * from user_stats;", conn)

    df_bkp = df
    df = df[['following_watchlist', 'watchlist_completion', 'likes_watchlist', 'retweets_watchlist',
             'mentions_watchlist', 'watchword_in_bio']]

    conn.close()

    return df, df_bkp


def get_predictions(df, model):
    pred = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    return pred, proba


def write_excel(path, df, worksheet, mode):
    if len(df.index) > 0:
        writer = pd.ExcelWriter(path, mode=mode)
        df.to_excel(writer, worksheet)
        writer.save()


def output_results(df_bkp, pred, proba, output):
    results = pd.DataFrame(df_bkp)
    results["pred"] = pred
    results["proba"] = proba
    results = results.sort_values(
        by=['proba'], ascending=False)

    write_excel(output, results, "predictions", "w")

    print("Predictions saved to: " + output)

# MAIN


parser = argparse.ArgumentParser()

parser.add_argument(
    "database", help="name of the db holding the twitter user stats")
parser.add_argument(
    "model", help="filename of the joblibbed model")
parser.add_argument(
    "--output", help="filename of the predicted output", default="predictions")

args = parser.parse_args()

if ".xlsx" not in args.output:
    args.output = args.output + ".xlsx"

df, df_bkp = fetch_data(args.database)
model = load_model(args.model)
pred, proba = get_predictions(df, model)
output_results(df_bkp, pred, proba, args.output)
