import argparse
import pandas as pd
import sqlite3
import twint
from multiprocessing import Pool, cpu_count
from time import sleep
from os import getpid, mkdir, path
from datetime import datetime
import traceback

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, isolation_level=None)
        return conn
    except Exception as e:
        print(e)
        return None

def create(conn):
    try:
        sql = '''CREATE TABLE user_stats ( 	id	TEXT, 	username	TEXT, 	last_updated	INTEGER, 	following_watchlist	REAL, 	watchlist_completion	REAL, 	likes_watchlist	REAL, 	retweets_watchlist	REAL, 	mentions_watchlist	REAL, watchword_in_bio INGEGER, 	is_on_watchlist	INTEGER, 	PRIMARY KEY(id) )'''
        cur = conn.cursor()
        cur.execute(sql)
    except Exception as e:
        print("Failed to create")
        print(e)

def insert(conn, user_stats):
    try:
        sql = '''INSERT INTO user_stats (id,username,last_updated,following_watchlist,watchlist_completion,likes_watchlist,retweets_watchlist,mentions_watchlist, watchword_in_bio, is_on_watchlist) VALUES (?,?,current_timestamp,?,?,?,?,?,?,?)'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except Exception as e:
        print("Failed to insert")
        print(e)


def update(conn, user_stats):
    try:
        sql = '''UPDATE user_stats SET username = ?, last_updated = current_timestamp, following_watchlist = ?, watchlist_completion = ?, likes_watchlist = ?, retweets_watchlist = ?, mentions_watchlist = ?, watchword_in_bio = ?, is_on_watchlist = ? WHERE id = ?'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except Exception as e:
        print("Failed to update")
        print(e)


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


# I only made this for myself since twint Pandas support is broken at the moment.
def twint_obj_list_to_df(twint_list):
    df = pd.DataFrame()

    if len(twint_list) > 0:
        # All attributes without __ are meant for humans
        interest_attr = []
        for attr in dir(twint_list[0]):
            if "__" not in attr:
                interest_attr.append(attr)

        # Array of dict objects
        twobs = []
        # Build the array of dict objects for pandas
        for obj in twint_list:
            twob = {}
            for attr in interest_attr:
                twob[attr] = obj.__getattribute__(attr)

                if "str" in str(type(obj.__getattribute__(attr))):
                    twob[attr] = obj.__getattribute__(attr).lower()

            twobs.append(twob)

        # Convert to DF
        df = pd.DataFrame(twobs)

    return df


def write_excel(path, df, columns, worksheet, mode):
    if len(df.index) > 0:
        writer = pd.ExcelWriter(path, mode=mode) # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        df.to_excel(writer, worksheet, columns=columns)
        writer.save()


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

    if filename != None:
        detect_file_header(filename, header)

        data = pd.read_csv(filename)
        data[header] = data[header].apply(
            lambda x: x.lower())

        if header == "watchwords":
            try:
                data = data[header].values
            except Exception as e:
                data = []
                print(e)
    else:
        if header == "watchwords":
            data = []
        else:
            data = pd.DataFrame()

    return data


def fetch_user_info(username):
    c = twint.Config()
    c.Username = username
    c.Pandas = True
    c.Hide_output = True

    twint.run.Lookup(c)
    Users_df = twint.storage.panda.User_df
    return Users_df


def get_user_info(username, directory):
    Users_df = fetch_user_info(username)

    try:
        Users_df["username"] = Users_df["username"].apply(lambda x: x.lower())
        user_id = str(Users_df[Users_df["username"] ==
                               username]['id'].values[0])
        private = Users_df[Users_df["username"]
                           == username]['private'].values[0]
        num_following = Users_df[Users_df["username"]
                                 == username]['following'].values[0]
        bio = Users_df[Users_df["username"]
                       == username]['bio'].values[0].lower()

        file_path = directory + username + ".xlsx"
        cols = ['name', 'username', 'followers', 'following',
            'bio', 'tweets', 'likes', 'join_date']
        write_excel(file_path,
                    Users_df[Users_df["username"] == username], cols, "user info", "w")
    except:
        print("Failed to fetch " + username + " user info.")
        user_id = None
        private = 1
        num_following = 0
        bio = None

    return user_id, private, num_following, bio


def fetch_following(username):
    c = twint.Config()
    c.Username = username
    c.Pandas = True
    c.Hide_output = True

    Followers_df = pd.DataFrame()
    twint.storage.panda.Follow_df = pd.DataFrame()

    attempts = 0
    while len(Followers_df.index) == 0 and attempts < attempt_limit:
        try:
            attempts += 1
            twint.run.Following(c)
            Followers_df = twint.storage.panda.Follow_df
        except Exception as e:
            print(e)
            print("Fetch following failed for " + username +
                  " sleeping for " + str(attempts * sleep_time) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * sleep_time)

    follower_list = pd.DataFrame(columns=[0])
    if len(Followers_df.index) > 0:
        for index, row in Followers_df.iteritems(): # pylint: disable=unused-variable
            follower_list = pd.DataFrame(row[0])
            follower_list[0] = follower_list[0].apply(lambda x: x.lower())

    return follower_list


def calculate_following_stats(num_following, username, follower_list, watchlist, directory):
    following_watchlist=0
    watchlist_completion=0

    if num_following > 0:
        watchlist_following=follower_list[follower_list[0].isin(
            watchlist["screen_names"])]

        file_path = directory + username + ".xlsx"
        cols=[0]
        write_excel(file_path, watchlist_following, cols,
                    "following watchlist", "a")

        if len(watchlist_following) > 0:
            following_watchlist=len(watchlist_following[0])/num_following
            watchlist_completion=len(
                watchlist_following[0])/len(watchlist["screen_names"])

    return following_watchlist, watchlist_completion


def following(username, num_following, watchlist, directory):
    follower_list = fetch_following(username)
    following_watchlist, watchlist_completion = calculate_following_stats(
        num_following, username, follower_list, watchlist, directory)

    return following_watchlist, watchlist_completion


def fetch_likes(username, limit):
    c = twint.Config()
    c.Username = username
    c.Store_object = True
    c.Hide_output = True
    c.Limit = limit

    tweets = []
    twint.output.tweets_list = []

    attempts = 0
    while len(tweets) == 0 and attempts < attempt_limit:
        try:
            attempts += 1            
            twint.run.Favorites(c)
            tweets = twint.output.tweets_list
        except Exception as e:
            print(e)
            print("Fetching likes failed for " + username + " sleeping for " + str(attempts * sleep_time) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * sleep_time)            
    
    df = twint_obj_list_to_df(tweets)

    return df


def calculate_like_stats(df, watchlist, username, directory):
    watchlist_intersect = 0

    if len(df.index) > 0:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"]!=username]

        df_on_watchlist = df[df["username"].isin(
            watchlist['screen_names'])]

        if len(df_on_watchlist.index) > 0 and len(df.index) > 0:
            watchlist_intersect = df_on_watchlist["username"].count(
            ) / len(df.index)

        file_path = directory + username + ".xlsx"
        cols = ['tweet','username','likes_count','retweets_count','replies_count','datestamp','timestamp','timezone','link']
        write_excel(file_path, df_on_watchlist, cols, "watchlist likes", "a")

    return watchlist_intersect


def likes(username, limit, watchlist, directory):
    df = fetch_likes(username, limit)
    likes_watchlist = calculate_like_stats(df, watchlist, username, directory)

    return likes_watchlist


def fetch_tweets(username, limit):
    c = twint.Config()
    c.Username = username
    c.Store_object = True
    c.Limit = limit
    c.Retweets = True
    c.Profile_full = True
    c.Hide_output = True

    tweets = []
    twint.output.tweets_list = []

    attempts = 0
    while len(tweets) == 0 and attempts < attempt_limit:
        try:
            attempts += 1
            twint.run.Profile(c)
            tweets = twint.output.tweets_list
        except Exception as e:
            print(e)            
            print("Fetching tweets failed for " + username + " sleeping for " + str(attempts * sleep_time) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * sleep_time)

    df = twint_obj_list_to_df(tweets)

    if len(df.index) > 0:
        # The retweets flag is broken in Twint 2.1.7
        retweets = df[df["username"] != username]
        # Pandas dataframe is broken for tweets in Twint 2.1.7 & 6
        mentions = df[df["mentions"] != "[]"]
        all_tweets = df
    else:
        # If df idx length is 0, then user may have no tweets.
        retweets = df
        mentions = df
        all_tweets = df

    return retweets, mentions, all_tweets


def calculate_retweets(retweets, watchlist, username, directory):
    watchlist_intersect_retweets = 0

    if len(retweets.index) > 0:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"]!=username]

        watchlist_retweets = retweets[retweets["username"].isin(
            watchlist['screen_names'])]

        if len(watchlist_retweets.index) > 0 and len(retweets.index) > 0:
            watchlist_intersect_retweets = watchlist_retweets["username"].count(
            ) / len(retweets["username"])

        file_path = directory + username + ".xlsx"
        cols = ['tweet','username','likes_count','retweets_count','replies_count','datestamp','timestamp','timezone','link']
        write_excel(file_path, watchlist_retweets, cols, "watchlist retweets", "a")

    return watchlist_intersect_retweets


def calculate_mentions(mentions, watchlist, username, directory):
    watchlist_intersect_mentions = 0

    if len(mentions.index) > 0:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"]!=username]

        filtered_mentions = []
        for index, row in mentions.iterrows():  # pylint: disable=unused-variable
            if row["username"] == username:
                for mention in row["mentions"]:
                    if mention in watchlist['screen_names'].values and mention != username:
                        filtered_mentions.append(row)
                        break

        filtered_mentions = pd.DataFrame(filtered_mentions)

        mention_usernames = []
        for tweet in mentions[mentions["username"]==username]["mentions"].values:
            for mention in tweet:
                if mention != username:
                    mention_usernames.append(mention)

        mention_usernames = pd.DataFrame(
            mention_usernames, columns=['username'])

        if len(mention_usernames.index) > 0:
            watchlist_intersect_mentions = mention_usernames[mention_usernames["username"].isin(
                watchlist['screen_names'])]["username"].count() / len(mention_usernames.index)

        file_path = directory + username + ".xlsx"
        cols = ['tweet','mentions','username','likes_count','retweets_count','replies_count','datestamp','timestamp','timezone','link']
        write_excel(file_path, filtered_mentions, cols, "watchlist mentions", "a")

    return watchlist_intersect_mentions


def find_watchword_tweets(all_tweets, watchwords, username, directory):

    if len(all_tweets.index) > 0 and len(watchwords) > 0:    
        filtered_tweets = []
        for index, row in all_tweets.iterrows():  # pylint: disable=unused-variable
            for watchword in watchwords:
                if watchword in row["tweet"]:
                    filtered_tweets.append(row)
                    break
        
        filtered_tweets = pd.DataFrame(filtered_tweets)

        file_path = directory + username + ".xlsx"
        cols = ['tweet','username','likes_count','retweets_count','replies_count','datestamp','timestamp','timezone','link']
        write_excel(file_path, filtered_tweets, cols, "watchword tweets", "a")


def calculate_tweet_stats(retweets, mentions, all_tweets, watchlist, watchwords, username, directory):
    watchlist_intersect_retweets = calculate_retweets(
        retweets, watchlist, username, directory)
    watchlist_intersect_mentions = calculate_mentions(
        mentions, watchlist, username, directory)
    find_watchword_tweets(all_tweets, watchwords, username, directory)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def tweets(username, limit, watchlist, watchwords, directory):
    retweets, mentions, all_tweets = fetch_tweets(username, limit)
    watchlist_intersect_retweets, watchlist_intersect_mentions = calculate_tweet_stats(
        retweets, mentions, all_tweets, watchlist, watchwords, username, directory)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def calculate_bio_stats(bio, watchwords):
    watchword_in_bio = 0

    if len(watchwords) != 0:
        for watchword in watchwords:
            if watchword in bio:
                watchword_in_bio = 1
                break

    return watchword_in_bio


def transform_work_item(work_item):
    user = work_item["user"]
    directory = work_item["directory"]
    db = create_connection(directory + work_item["db"])
    watchlist = work_item["watchlist"]
    tweet_fetch_limit = work_item["tweet_fetch_limit"]
    tweet_watchwords = work_item["tweet_watchwords"]
    bio_watchwords = work_item["bio_watchwords"]

    return user, directory, db, watchlist, tweet_fetch_limit, tweet_watchwords, bio_watchwords

def transform_stats(user, following_watchlist, watchlist_completion,likes_watchlist,retweets_watchlist,mentions_watchlist,watchword_in_bio,is_on_watchlist):
    result = [{
        "user": user,
        "following_watchlist": following_watchlist,
        "watchlist_completion": watchlist_completion,
        "likes_watchlist": likes_watchlist,
        "retweets_watchlist": retweets_watchlist,
        "mentions_watchlist": mentions_watchlist,
        "watchword_in_bio": watchword_in_bio,
        "is_on_watchlist": is_on_watchlist
    }]
    return result

def process_user(work_item):
    try:
        # work_item needed to be a dictionary because of the way multiprocessing works
        user, directory, db, watchlist, tweet_fetch_limit, tweet_watchwords, bio_watchwords = transform_work_item(
            work_item)

        print("PID " + str(getpid()) + " START " + user + " " + str(datetime.now()))

        user_id, private, following_count, bio = get_user_info(user, directory)
        # If I  can't get user ID or if profile is private, stats are useless.
        if user_id != None and private != 1:

            following_watchlist, watchlist_completion = following(
                user, following_count, watchlist, directory)
            likes_watchlist = likes(
                user, tweet_fetch_limit, watchlist, directory)
            retweets_watchlist, mentions_watchlist = tweets(
                user, tweet_fetch_limit, watchlist, tweet_watchwords, directory)
            watchword_in_bio = calculate_bio_stats(bio, bio_watchwords)

            if user in watchlist['screen_names'].values:
                is_on_watchlist = 1
            else:
                is_on_watchlist = 0

            stats = transform_stats(user, following_watchlist, watchlist_completion,likes_watchlist,retweets_watchlist,mentions_watchlist,watchword_in_bio,is_on_watchlist)
            stats = pd.DataFrame(stats)
            file_path = directory + user + ".xlsx"
            cols = ["user", "following_watchlist", "watchlist_completion","likes_watchlist","retweets_watchlist","mentions_watchlist","watchword_in_bio","is_on_watchlist"]
            write_excel(file_path, stats, cols, "user stats", "a")


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

        print("PID " + str(getpid()) + " END " + user + " " + str(datetime.now()))
    except:
        print(str(getpid()) + " EXCEPTION")
        print(traceback.format_exc())
    
    db.close()


def pool_handler(work):
    p = Pool(cpu_count())
    print("Parallel processing on " + str(cpu_count()) + " cores.")
    p.map_async(process_user, work)
    p.close()
    p.join()

# GLOBAL VARS
# Below are settings intended for a resiliency feature.
attempt_limit = 5
sleep_time = 5000

# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "watchlist", help="group of twitter users who are interesting")
    parser.add_argument(
        "output", help="specify output database")
    parser.add_argument(
        "--bio_watchwords", help="list of watchwords to look for in bio unique to the target group")
    parser.add_argument(
        "--tweet_watchwords", help="list of watchwords to look for in tweets unique to the target group")
    parser.add_argument(
        "--tweet_fetch_limit", help="number of tweets to fetch when calculating statistics", default=100, type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--username', help="username of single user to fetch data on")
    group.add_argument(
        '--userlist',  help="csv containing usernames to fetch data on")

    args = parser.parse_args()

    watchlist = import_csv(args.watchlist, "screen_names")

    if ".db" not in args.output:
        args.output = args.output + ".db"

    # Create an output directory, since several XLSX files will be produced.
    directory = "./" + args.output.replace(".db","") + "/"
    if not path.exists(directory):
        mkdir(directory)

    db = create_connection(directory + args.output)
    if exists_table(db) != True:
        create(db)
    db.close()

    bio_watchwords = import_csv(args.bio_watchwords, "watchwords")
    tweet_watchwords = import_csv(args.tweet_watchwords, "watchwords")

    if args.username != None:
        users = [args.username]

    if args.userlist != None:
        users = import_csv(args.userlist, "screen_names")
        users = users["screen_names"].values

    # Build work queue.
    work = []
    for user in users:
        work_item = {
            "user": user,
            "db": args.output,
            "directory": directory,
            "watchlist": watchlist,
            "tweet_fetch_limit": args.tweet_fetch_limit,
            "tweet_watchwords": tweet_watchwords,
            "bio_watchwords": bio_watchwords
        }

        work.append(work_item)

    # Start work in multiple processes.
    pool_handler(work)

    db.close()
