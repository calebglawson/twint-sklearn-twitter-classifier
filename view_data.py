import argparse
import numpy as np
import os
import pandas as pd
import sqlite3


def detect_file_header(filename, header):
    f = open(filename, "r")
    lines = f.readlines()

    if header not in lines[0]:
        f.close()
        f = open(filename, "w")
        lines.insert(0, header+"\n")
        f.writelines(lines)
        f.close()


def load_watchlist(filename, header):

    detect_file_header(filename, header)

    watchlist = pd.read_csv(filename)
    watchlist[header] = watchlist[header].apply(lambda x: x.lower())
    watchlist = tuple(watchlist[header].values)
    return watchlist


def load_csv(filename, header):
    detect_file_header(filename, header)
    return pd.read_csv(filename)


def connect_to_db(filename):
    return sqlite3.connect(filename)


def get_user(username, db, filepath):
    username = username.lower()
    query = "select * from users where lower(username) = '" + username + "'"
    result = pd.read_sql_query(query, db)
    result.to_csv(filepath + "user_info.csv")

    # Get the user's ID for future queries
    id = result["id"].values[0]

    return id


def get_following_watchlist(username, watchlist, db, filepath):
    query = "select user from following_names where lower(follows) = '" + \
        username + "' and lower(user) in " + str(watchlist) + ""
    result = pd.read_sql_query(query, db)

    if len(result.index) > 0:
        result.to_csv(filepath + "following_watchlist.csv")

    return len(result.index)


def get_watchlist_favorites(user_id, watchlist, db, filepath):
    query = "select t.screen_name, t.tweet, t.date, t.time, t.timezone, t.replies_count, t.likes_count, t.retweets_count, t.link from favorites f join tweets t on f.tweet_id = t.id where f.user_id = " + \
        str(user_id) + " and lower(t.screen_name) in " + str(watchlist)
    result = pd.read_sql_query(query, db)

    if len(result.index) > 0:
        result.to_csv(filepath + "watchlist_favorites.csv")

    return len(result.index)


def get_watchlist_mentions(user_id, watchlist, db, filepath):
    query = "select t.screen_name, t.tweet, t.mentions, t.date, t.time, t.timezone, t.replies_count, t.likes_count, t.retweets_count, t.link from tweets t where user_id = " + \
        str(user_id) + " and mentions <> ''"
    result = pd.read_sql_query(query, db)

    if len(result.index) > 0:
        watchlist = list(watchlist)

        result["contains_watchlist"] = result["mentions"].apply(
            lambda x: any([value for value in x.lower().split(',') if value in watchlist]))

        result = result[result["contains_watchlist"]].drop(
            ["contains_watchlist"], axis=1)

    if len(result.index) > 0:
        result.to_csv(
            filepath + "watchlist_mentions.csv")

    return len(result.index)


def get_watchlist_retweets(user_id, watchlist, db, filepath):
    query = "select t.screen_name, t.tweet, t.date, t.time, t.timezone, t.replies_count, t.likes_count, t.retweets_count, t.link from retweets r join tweets t on r.tweet_id = t.id where r.user_id = " + \
        str(user_id) + " and lower(t.screen_name) in " + str(watchlist)
    result = pd.read_sql_query(query, db)

    if len(result.index) > 0:
        result.to_csv(filepath + "watchlist_retweets.csv")

    return len(result.index)


def get_watchword_tweets(user_id, watchwords, db, filepath):
    query = "select t.screen_name, t.tweet, t.date, t.time, t.timezone, t.replies_count, t.likes_count, t.retweets_count, t.link from tweets t where user_id = " + \
        str(user_id)
    tweets = pd.read_sql_query(query, db)

    query = "select t.screen_name, t.tweet, t.date, t.time, t.timezone, t.replies_count, t.likes_count, t.retweets_count, t.link from retweets r join tweets t on r.tweet_id = t.id where r.user_id = " + \
        str(user_id)
    retweets = pd.read_sql_query(query, db)

    result = pd.concat([tweets, retweets])
    result["tweet"] = result["tweet"].apply(lambda x: x.lower())

    mask = []
    for tweet in result["tweet"]:
        ww_found = False
        for watchword in watchwords["watchwords"]:
            if watchword in tweet:
                ww_found = True
                break
        mask.append(ww_found)

    result = result[mask]

    if len(result.index) > 0:
        result.to_csv(filepath + "tweets_with_watchwords.csv")

    return len(result.index)


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "database", help="db to fetch the data from")
    parser.add_argument("watchlist", help="csv file, users of interest")
    parser.add_argument(
        "--tweet_watchwords", help="csv of watchwords to look for in tweets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--username', help="username of single user to fetch data on")
    group.add_argument(
        '--input_file',  help="csv containing usernames to fetch data on")

    args = parser.parse_args()

    if "_twint_data.db" not in args.database:
        args.database = args.database[:-3] + "_twint_data.db"
        print(args.database)

    if args.username != None:
        users = [args.username]

    if args.input_file != None:
        users = load_csv(args.input_file, "screen_names")
        users = users["screen_names"].values

    db = connect_to_db(args.database)
    watchwords = load_csv(args.tweet_watchwords, "watchwords")
    watchlist = load_watchlist(args.watchlist, "screen_names")

    for user in users:
        user = user.lower()
        filepath = "./" + user + "/"

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        user_id = get_user(user, db, filepath)
        following_watchlist = get_following_watchlist(
            user, watchlist, db, filepath)
        watchlist_favorites = get_watchlist_favorites(
            user_id, watchlist, db, filepath)
        watchlist_retweets = get_watchlist_retweets(
            user_id, watchlist, db, filepath)
        watchlist_mentions = get_watchlist_mentions(
            user_id, watchlist, db, filepath)
        watchword_tweets = get_watchword_tweets(
            user_id, watchwords, db, filepath)

        print("\n" + user)
        print("Watchlist follow count: " + str(following_watchlist))
        print("Watchlist favorite count: " + str(watchlist_favorites))
        print("Watchlist retweet count: " + str(watchlist_retweets))
        print("Watchlist mention count: " + str(watchlist_mentions))
        print("Watchword tweets: " + str(watchword_tweets))

    print("\nAll files output in working directory.")
    db.close()
