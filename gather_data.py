import argparse
import numpy as np
import pandas as pd
import sqlite3
import twint


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except:
        return None


def create(conn):
    try:
        sql = '''CREATE TABLE user_stats ( 	id	TEXT, 	username	TEXT, 	last_updated	INTEGER, 	following_watchlist	REAL, 	watchlist_completion	REAL, 	likes_watchlist	REAL, 	retweets_watchlist	REAL, 	mentions_watchlist	REAL, watchword_in_bio INGEGER, 	is_on_watchlist	INTEGER, 	PRIMARY KEY(id) )'''
        cur = conn.cursor()
        cur.execute(sql)
    except:
        print("Failed to create")


def insert(conn, user_stats):
    try:
        sql = '''INSERT INTO user_stats (id,username,last_updated,following_watchlist,watchlist_completion,likes_watchlist,retweets_watchlist,mentions_watchlist, watchword_in_bio, is_on_watchlist) VALUES (?,?,current_timestamp,?,?,?,?,?,?,?)'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except:
        print("Failed to insert")


def update(conn, user_stats):
    try:
        sql = '''UPDATE user_stats SET username = ?, last_updated = current_timestamp, following_watchlist = ?, watchlist_completion = ?, likes_watchlist = ?, retweets_watchlist = ?, mentions_watchlist = ?, watchword_in_bio = ?, is_on_watchlist = ? WHERE id = ?'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except:
        print("Failed to update")


def exists(conn, id):
    sql = '''SELECT * from user_stats WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, id)
    rows = cur.fetchall()
    return len(rows) > 0


def exists_table(conn):
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='user_stats'"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return len(rows) > 0


def detect_file_header(filename, header):
    f = open(filename, "r")
    lines = f.readlines()

    if header not in lines[0]:
        f.close()
        f = open(filename, "w")
        lines.insert(0, header+"\n")
        f.writelines(lines)
        f.close()


def import_csv(filename, header):

    detect_file_header(filename, header)

    data = pd.read_csv(filename)
    data[header] = data[header].apply(
        lambda x: x.lower())

    if header == "watchwords":
        try:
            data = data[header].values
        except:
            data = []

    return data


def fetch_user_info(username, db):
    c = twint.Config()
    c.Username = username
    c.Pandas = True
    c.Hide_output = True
    c.Database = db
    twint.run.Lookup(c)
    Users_df = twint.storage.panda.User_df
    return Users_df


def get_user_info(username, db):
    Users_df = fetch_user_info(user, db)

    try:
        Users_df["username"] = Users_df["username"].apply(lambda x: x.lower())
        user_id = str(Users_df[Users_df["username"] ==
                               user]['id'].values[0])
        private = Users_df[Users_df["username"]
                           == user]['private'].values[0]
        num_following = Users_df[Users_df["username"]
                                 == username]['following'].values[0]
        bio = Users_df[Users_df["username"]
                       == username]['bio'].values[0].lower()
    except:
        print("failed to fetch")
        user_id = None
        private = 1
        num_following = 0
        bio = None

    return user_id, private, num_following, bio


def fetch_following(username, db):
    c = twint.Config()
    c.Username = username
    c.Pandas = True
    c.Hide_output = True
    c.Database = db
    twint.storage.panda.Follow_df = pd.DataFrame()
    twint.run.Following(c)

    Followers_df = twint.storage.panda.Follow_df

    try:
        follower_list = pd.DataFrame(Followers_df['following'][username])
        follower_list[0].apply(lambda x: x.lower())
    except:
        follower_list = pd.DataFrame()

    return follower_list


def calculate_following_stats(num_following, username, follower_list, watchlist):
    following_watchlist = 0
    watchlist_completion = 0

    if num_following > 0:
        watchlist_following = np.intersect1d(follower_list, watchlist)

        if len(watchlist_following) > 0:
            following_watchlist = len(watchlist_following)/num_following
            watchlist_completion = len(watchlist_following)/len(watchlist)

    return following_watchlist, watchlist_completion


def following(username, num_following, watchlist, db):

    try:
        follower_list = fetch_following(username, db)
        following_watchlist, watchlist_completion = calculate_following_stats(
            num_following, username, follower_list, watchlist)
    except:
        print("Failed to fetch following")
        following_watchlist = 0
        watchlist_completion = 0

    return following_watchlist, watchlist_completion


def fetch_likes(username, limit, db):
    c = twint.Config()
    c.Username = username
    c.Store_object = True
    c.Hide_output = True
    c.Database = db
    c.Limit = limit
    twint.output.tweets_list = []
    twint.run.Favorites(c)

    faves = twint.output.tweets_list

    build = []

    for fave in faves:
        build.append(fave.username)

    df = pd.DataFrame(build, columns=['username'])
    df['username'] = df['username'].apply(lambda x: x.lower())

    return df


def calculate_like_stats(df, watchlist):
    watchlist_intersect = df[df["username"].isin(
        watchlist['screen_names'])]["username"].count() / len(df["username"])

    if np.isnan(watchlist_intersect):
        watchlist_intersect = 0

    return watchlist_intersect


def likes(username, limit, watchlist, db):
    try:
        df = fetch_likes(username, limit, db)
        likes_watchlist = calculate_like_stats(df, watchlist)
    except:
        likes_watchlist = 0
        print("Failed to fetch likes")

    return likes_watchlist


def fetch_tweets(username, limit, db):
    c = twint.Config()
    c.Username = username
    c.Store_object = True
    c.Limit = limit
    c.Retweets = True
    c.Hide_output = True
    c.Database = db
    twint.output.tweets_list = []
    twint.run.Profile(c)

    faves = twint.output.tweets_list

    retweets = []
    mentions = []
    all_tweets = []

    for fave in faves:
        if fave.retweet == 1:
            retweets.append(fave.username.lower())

        if fave.username.lower() == username and len(fave.mentions) > 0:
            for mention in fave.mentions:
                mentions.append(mention.lower())

        all_tweets.append(fave.tweet.lower())

    retweets = pd.DataFrame(retweets, columns=['username'])
    mentions = pd.DataFrame(mentions, columns=['username'])

    return retweets, mentions, all_tweets


def calculate_tweet_stats(retweets, mentions, watchlist, all_tweets):
    watchlist_intersect_retweets = retweets[retweets["username"].isin(
        watchlist['screen_names'])]["username"].count() / len(retweets["username"])

    watchlist_intersect_mentions = mentions[mentions["username"].isin(
        watchlist['screen_names'])]["username"].count() / len(mentions["username"])

    if np.isnan(watchlist_intersect_retweets):
        watchlist_intersect_retweets = 0

    if np.isnan(watchlist_intersect_mentions):
        watchlist_intersect_mentions = 0

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def tweets(username, limit, watchlist, db):
    try:
        retweets, mentions, all_tweets = fetch_tweets(username, limit, db)
        watchlist_intersect_retweets, watchlist_intersect_mentions = calculate_tweet_stats(
            retweets, mentions, watchlist, all_tweets)
    except:
        watchlist_intersect_retweets, watchlist_intersect_mentions = (
            0, 0)
        print("Failed to fetch tweets")

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def calculate_bio_stats(bio, watchwords):
    watchword_in_bio = 0

    if len(watchwords) != 0:
        for watchword in watchwords:
            if watchword in bio:
                watchword_in_bio = 1
                break

    return watchword_in_bio

# MAIN


parser = argparse.ArgumentParser()
parser.add_argument(
    "input", help="twitter users from whom to fetch the stats about their relationship to the watchlist users")
parser.add_argument(
    "watchlist", help="group of twitter users who are interesting")
parser.add_argument(
    "output", help="specify output database")
parser.add_argument(
    "--bio_watchwords", help="list of watchwords to look for in tweets unique to the target group")
parser.add_argument(
    "--fetch_limit", help="number of tweets to fetch when calculating statistics", default=100, type=int)

args = parser.parse_args()

if ".db" not in args.output:
    args.output = args.output + ".db"

db = create_connection(args.output)
twint_db = args.output.replace(".db", "_twint_data.db")
watchlist = import_csv(args.watchlist, "screen_names")
intake = import_csv(args.input, "screen_names")

bio_watchwords = import_csv(args.bio_watchwords, "watchwords")

if exists_table(db) != True:
    create(db)

for user in intake["screen_names"]:

    user = user.lower()
    print(user)

    user_id, private, following_count, bio = get_user_info(user, twint_db)

    # If I  can't get user ID or if profile is private, stats are useless.
    if user_id != None and private != 1:

        following_watchlist, watchlist_completion = following(
            user, following_count, watchlist, twint_db)
        likes_watchlist = likes(user, args.fetch_limit, watchlist, twint_db)
        retweets_watchlist, mentions_watchlist = tweets(
            user, args.fetch_limit, watchlist, twint_db)
        watchword_in_bio = calculate_bio_stats(bio, bio_watchwords)

        if user in watchlist['screen_names'].values:
            is_on_watchlist = 1
        else:
            is_on_watchlist = 0

        if is_on_watchlist == 0 or (is_on_watchlist == 1 and (following_watchlist != 0 or watchlist_completion != 0 or likes_watchlist != 0 or retweets_watchlist != 0 or mentions_watchlist != 0)):
            if exists(db, (user_id,)):
                user_row = (user, following_watchlist, watchlist_completion,
                            likes_watchlist, retweets_watchlist, mentions_watchlist, watchword_in_bio, is_on_watchlist, user_id)
                update(db, user_row)
            else:
                user_row = (user_id, user, following_watchlist, watchlist_completion,
                            likes_watchlist, retweets_watchlist, mentions_watchlist, watchword_in_bio, is_on_watchlist)
                insert(db, user_row)

            db.commit()

        else:
            print("Skipped insert: " + user)

db.close()
