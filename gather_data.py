'''From a list of twitter users, gather the twitter data, persist it, and compute stats.
Outputs a single SQLite Database the machine learning model and Excel reports for humans.'''


from multiprocessing import Pool, cpu_count
from time import sleep
from os import getpid, mkdir, path
from datetime import datetime
import argparse
import sqlite3
import traceback
import twint
import pandas as pd


def create_connection(db_file):
    '''Create a connection to the stats output DB.'''

    try:
        conn = sqlite3.connect(db_file, isolation_level=None)
        return conn
    except Exception as exception:
        print(exception)
        return None


def create(conn):
    '''Create the stats table.'''

    try:
        sql = '''CREATE TABLE user_stats ( id TEXT,	username TEXT, last_updated INTEGER,
        following_watchlist REAL, watchlist_completion REAL, likes_watchlist REAL,
        retweets_watchlist REAL, mentions_watchlist REAL, watchword_in_bio INGEGER,
        is_on_watchlist INTEGER, PRIMARY KEY(id) )'''
        cur = conn.cursor()
        cur.execute(sql)
    except Exception as exception:
        print("Failed to create DB Schema.")
        print(exception)


def insert(conn, user_stats):
    '''Insert a row into the stats table.'''

    try:
        sql = '''INSERT INTO user_stats (id, username, last_updated, following_watchlist,
        watchlist_completion, likes_watchlist, retweets_watchlist, mentions_watchlist,
        watchword_in_bio, is_on_watchlist) VALUES (?,?,current_timestamp,?,?,?,?,?,?,?)'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except Exception as exception:
        print(f"Failed to insert {user_stats[1]}.")
        print(exception)


def update(conn, user_stats):
    '''Update a row in the stats table.'''

    try:
        sql = '''UPDATE user_stats SET username = ?, last_updated = current_timestamp,
        following_watchlist = ?, watchlist_completion = ?, likes_watchlist = ?,
        retweets_watchlist = ?, mentions_watchlist = ?, watchword_in_bio = ?,
        is_on_watchlist = ? WHERE id = ?'''
        cur = conn.cursor()
        cur.execute(sql, user_stats)
    except Exception as exception:
        print(f"Failed to update {user_stats[0]}.")
        print(exception)


def exists(conn, user_id):
    '''Check if a given user exists in the stats table.
    Used to determine insert or update.'''

    sql = '''SELECT * from user_stats WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, user_id)
    rows = cur.fetchall()
    return len(rows) > 0


def exists_table(conn):
    '''Check if the user_stats table exists.
    Used to determine if schema needs to be created.'''

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='user_stats'"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return len(rows) > 0


def twint_obj_list_to_dataframe(twint_list):
    ''' I only made this for myself since twint Pandas support is broken for tweets, at the moment.
    Takes twint obj and converts it to Pandas dataframe.'''
    dataframe = pd.DataFrame()

    # Sequences evaluate to true if not empty.
    if twint_list:
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

        # Convert to dataframe
        dataframe = pd.DataFrame(twobs)

    return dataframe


def write_excel(directory, dataframe, columns, worksheet, mode):
    '''Write data frame to Excel file.'''
    if not dataframe.index.empty:
        # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        writer = pd.ExcelWriter(directory, mode=mode)
        dataframe.to_excel(writer, worksheet, columns=columns)
        writer.save()


def detect_file_header(filename, header):
    '''Open the CSV and determine if header exists.
    If header does not exist, add one.'''

    csv = open(filename, "r")
    lines = csv.readlines()

    if header not in lines[0]:
        csv.close()
        csv = open(filename, "w")
        lines.insert(0, header+"\n")
        csv.writelines(lines)
        csv.close()


def import_csv(filename, header):
    '''Import the CSV as a dataframe with column as header.'''

    if filename is not None:
        detect_file_header(filename, header)

        data = pd.read_csv(filename)
        data[header] = data[header].apply(
            lambda x: x.lower())

        if header == "watchwords":
            try:
                data = data[header].values
            except Exception as exception:
                data = []
                print(exception)
    else:
        if header == "watchwords":
            data = []
        else:
            data = pd.DataFrame()

    return data


def fetch_user_info(username):
    '''Fetch twitter user profile info with twint.'''

    config = twint.Config()
    config.Username = username
    config.Pandas = True
    config.Hide_output = True

    twint.run.Lookup(config)
    users_dataframe = twint.storage.panda.User_df
    return users_dataframe


def get_user_info(username, file_path):
    '''Get and transform user info.'''

    users_dataframe = fetch_user_info(username)

    try:
        users_dataframe["username"] = users_dataframe["username"].apply(
            lambda x: x.lower())
        user_id = str(users_dataframe[users_dataframe["username"] ==
                                      username]['id'].values[0])
        private = users_dataframe[users_dataframe["username"]
                                  == username]['private'].values[0]
        num_following = users_dataframe[users_dataframe["username"]
                                        == username]['following'].values[0]
        bio = users_dataframe[users_dataframe["username"]
                              == username]['bio'].values[0].lower()

        file_path = file_path + username + ".xlsx"
        cols = ['name', 'username', 'followers', 'following',
                'bio', 'tweets', 'likes', 'join_date']
        write_excel(file_path,
                    users_dataframe[users_dataframe["username"] == username],
                    cols, "user info", "w")

    except Exception as exception:
        print(f"Failed to fetch {username} user info.")
        print(exception)
        user_id = None
        private = 1
        num_following = 0
        bio = None

    return user_id, private, num_following, bio


def fetch_following(username):
    '''Fetch users that the user is following.'''

    config = twint.Config()
    config.Username = username
    config.Pandas = True
    config.Hide_output = True

    followers_dataframe = pd.DataFrame()
    twint.storage.panda.Follow_df = pd.DataFrame()

    attempts = 0
    while followers_dataframe.index.empty and attempts < ATTEMPT_LIMIT:
        try:
            attempts += 1
            twint.run.Following(config)
            followers_dataframe = twint.storage.panda.Follow_df
        except Exception as exception:
            print(exception)
            print("Fetch following failed for " + username +
                  " sleeping for " + str(attempts * SLEEP_TIME) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * SLEEP_TIME)

    follower_list = pd.DataFrame(columns=[0])
    if not followers_dataframe.index.empty:
        for index, row in followers_dataframe.iteritems():  # pylint: disable=unused-variable
            follower_list = pd.DataFrame(row[0])
            follower_list[0] = follower_list[0].apply(lambda x: x.lower())

    return follower_list


def calculate_following_stats(num_following, username, follower_list, wlist, file_path):
    '''Calculate following watchlist ratio and watchlist completion ratio.'''

    following_watchlist = 0
    watchlist_completion = 0

    if num_following > 0:
        watchlist_following = follower_list[follower_list[0].isin(
            wlist["screen_names"])]

        file_path = file_path + username + ".xlsx"
        cols = [0]
        write_excel(file_path, watchlist_following, cols,
                    "following watchlist", "a")

        if not watchlist_following.index.empty:
            following_watchlist = len(watchlist_following[0])/num_following
            watchlist_completion = len(
                watchlist_following[0])/len(wlist["screen_names"])

    return following_watchlist, watchlist_completion


def following(username, num_following, wlist, file_path):
    '''Wrapper for fetching and calculating following stats.'''

    follower_list = fetch_following(username)
    following_watchlist, watchlist_completion = calculate_following_stats(
        num_following, username, follower_list, wlist, file_path)

    return following_watchlist, watchlist_completion


def fetch_likes(username, limit):
    '''Fetch a user's likes with twint.'''

    config = twint.Config()
    config.Username = username
    config.Store_object = True
    config.Hide_output = True
    config.Limit = limit

    fetched_tweets = []
    twint.output.tweets_list = []

    attempts = 0
    while not fetched_tweets and attempts < ATTEMPT_LIMIT:
        try:
            attempts += 1
            twint.run.Favorites(config)
            fetched_tweets = twint.output.tweets_list
        except Exception as exception:
            print(exception)
            print("Fetching likes failed for " + username +
                  " sleeping for " + str(attempts * SLEEP_TIME) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * SLEEP_TIME)

    dataframe = twint_obj_list_to_dataframe(fetched_tweets)

    return dataframe


def calculate_like_stats(dataframe, wlist, username, file_path):
    '''Calculate the watchlist vs non-watchlist tweet like ratio.'''

    watchlist_intersect = 0

    if not dataframe.index.empty:
        # Don't count self.
        wlist = wlist[wlist["screen_names"] != username]

        dataframe_on_watchlist = dataframe[dataframe["username"].isin(
            wlist['screen_names'])]

        if not dataframe_on_watchlist.index.empty and not dataframe.index.empty:
            watchlist_intersect = dataframe_on_watchlist["username"].count(
            ) / len(dataframe.index)

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'username', 'likes_count', 'retweets_count', 'replies_count', 'datestamp',
                'timestamp', 'timezone', 'link']
        write_excel(file_path, dataframe_on_watchlist,
                    cols, "watchlist likes", "a")

    return watchlist_intersect


def likes(username, limit, wlist, file_path):
    '''Wrapper function for fetching and calculating like stats.'''

    dataframe = fetch_likes(username, limit)
    likes_watchlist = calculate_like_stats(
        dataframe, wlist, username, file_path)

    return likes_watchlist


def fetch_tweets(username, limit):
    '''Fetch tweets for stats calculation.'''

    config = twint.Config()
    config.Username = username
    config.Store_object = True
    config.Limit = limit
    config.Retweets = True
    config.Profile_full = True
    config.Hide_output = True

    fetched_tweets = []
    twint.output.tweets_list = []

    attempts = 0
    while not fetched_tweets and attempts < ATTEMPT_LIMIT:
        try:
            attempts += 1
            twint.run.Profile(config)
            fetched_tweets = twint.output.tweets_list
        except Exception as exception:
            print(exception)
            print("Fetching tweets failed for " + username +
                  " sleeping for " + str(attempts * SLEEP_TIME) + " milliseconds.")
            print(traceback.format_exc())
            sleep(attempts * SLEEP_TIME)

    dataframe = twint_obj_list_to_dataframe(fetched_tweets)

    if not dataframe.index.empty:
        # The retweets flag is broken in Twint 2.1.7
        retweets = dataframe[dataframe["username"] != username]
        # Pandas dataframe is broken for tweets in Twint 2.1.7 & 6
        mentions = dataframe[dataframe["mentions"] != "[]"]
        all_tweets = dataframe
    else:
        # If dataframe idx length is 0, then user may have no tweets.
        retweets = dataframe
        mentions = dataframe
        all_tweets = dataframe

    return retweets, mentions, all_tweets


def calculate_retweets(retweets, wlist, username, file_path):
    '''Calculate watchlist vs non-watchlist retweet ratio.'''

    watchlist_intersect_retweets = 0

    if not retweets.index.empty:
        # Don't count self.
        wlist = wlist[wlist["screen_names"] != username]

        watchlist_retweets = retweets[retweets["username"].isin(
            wlist['screen_names'])]

        if not watchlist_retweets.index.empty and not retweets.index.empty:
            watchlist_intersect_retweets = watchlist_retweets["username"].count(
            ) / len(retweets["username"])

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'username', 'likes_count', 'retweets_count', 'replies_count',
                'datestamp', 'timestamp', 'timezone', 'link']
        write_excel(file_path, watchlist_retweets,
                    cols, "watchlist retweets", "a")

    return watchlist_intersect_retweets


def calculate_mentions(mentions, wlist, username, file_path):
    '''Calculate watchlist vs non-watchlist mention ratio.'''

    watchlist_intersect_mentions = 0

    if not mentions.index.empty:
        # Don't count self.
        wlist = wlist[wlist["screen_names"] != username]

        filtered_mentions = []
        for index, row in mentions.iterrows():  # pylint: disable=unused-variable
            if row["username"] == username:
                for mention in row["mentions"]:
                    if mention in wlist['screen_names'].values and mention != username:
                        filtered_mentions.append(row)
                        break

        filtered_mentions = pd.DataFrame(filtered_mentions)

        mention_usernames = []
        for tweet in mentions[mentions["username"] == username]["mentions"].values:
            for mention in tweet:
                if mention != username:
                    mention_usernames.append(mention)

        mention_usernames = pd.DataFrame(
            mention_usernames, columns=['username'])

        if not mention_usernames.index.empty:
            watchlist_intersect_mentions = mention_usernames[mention_usernames["username"].isin(
                wlist['screen_names'])]["username"].count() / len(mention_usernames.index)

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'mentions', 'username', 'likes_count', 'retweets_count', 'replies_count',
                'datestamp', 'timestamp', 'timezone', 'link']
        write_excel(file_path, filtered_mentions,
                    cols, "watchlist mentions", "a")

    return watchlist_intersect_mentions


def find_watchword_tweets(all_tweets, watchwords, username, file_path):
    '''Search for tweets with watchwords and phrases for Excel output only.'''
    if not all_tweets.index.empty and watchwords.size != 0:
        filtered_tweets = []
        for index, row in all_tweets.iterrows():  # pylint: disable=unused-variable
            for watchword in watchwords:
                if watchword in row["tweet"]:
                    filtered_tweets.append(row)
                    break

        filtered_tweets = pd.DataFrame(filtered_tweets)

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'username', 'likes_count', 'retweets_count', 'replies_count', 'datestamp',
                'timestamp', 'timezone', 'link']
        write_excel(file_path, filtered_tweets, cols, "watchword tweets", "a")


def calculate_tweet_stats(retweets, mentions, all_tweets, wlist, watchwords, username, file_path):
    '''Wrapper for calculating various tweet stats.'''

    watchlist_intersect_retweets = calculate_retweets(
        retweets, wlist, username, file_path)
    watchlist_intersect_mentions = calculate_mentions(
        mentions, wlist, username, file_path)
    find_watchword_tweets(all_tweets, watchwords, username, file_path)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def tweets(username, limit, wlist, watchwords, file_path):
    '''Wrapper function for fetching tweets and calculating stats.'''

    retweets, mentions, all_tweets = fetch_tweets(username, limit)
    watchlist_intersect_retweets, watchlist_intersect_mentions = calculate_tweet_stats(
        retweets, mentions, all_tweets, wlist, watchwords, username, file_path)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def calculate_bio_stats(bio, watchwords):
    '''Search the bio for watchwords.'''
    watchword_in_bio = 0

    if watchwords.size == 0:
        for watchword in watchwords:
            if watchword in bio:
                watchword_in_bio = 1
                break

    return watchword_in_bio


def transform_work(item):
    '''Unpack the dictionary passed to the process function.'''
    username = item["user"]
    file_path = item["directory"]
    database = create_connection(file_path + item["db"])
    wlist = item["watchlist"]
    tweet_fetch_limit = item["tweet_fetch_limit"]
    tweet_ww = item["tweet_watchwords"]
    bio_ww = item["bio_watchwords"]

    return username, file_path, database, wlist, tweet_fetch_limit, tweet_ww, bio_ww


def transform_stats(username, following_watchlist, watchlist_completion, likes_watchlist,
                    retweets_watchlist, mentions_watchlist, watchword_in_bio, is_on_watchlist):
    '''Transform stats to dict for conversion to DataFrame and Excel output.'''

    result = [{
        "user": username,
        "following_watchlist": following_watchlist,
        "watchlist_completion": watchlist_completion,
        "likes_watchlist": likes_watchlist,
        "retweets_watchlist": retweets_watchlist,
        "mentions_watchlist": mentions_watchlist,
        "watchword_in_bio": watchword_in_bio,
        "is_on_watchlist": is_on_watchlist
    }]
    return result


def process_user(item):
    '''Main function for processing a user.'''

    try:
        # work_item needed to be a dictionary because of the way multiprocessing works
        username, file_path, database, wlist, tweet_fetch_limit, tweet_ww, bio_ww = transform_work(
            item)

        print("PID " + str(getpid()) + " START " +
              username + " " + str(datetime.now()))

        user_id, private, following_count, bio = get_user_info(
            username, file_path)
        # If I  can't get user ID or if profile is private, stats are useless.
        if user_id is not None and private != 1:

            following_watchlist, watchlist_completion = following(
                username, following_count, wlist, file_path)
            likes_watchlist = likes(
                username, tweet_fetch_limit, wlist, file_path)
            retweets_watchlist, mentions_watchlist = tweets(
                username, tweet_fetch_limit, wlist, tweet_ww, file_path)
            watchword_in_bio = calculate_bio_stats(bio, bio_ww)

            if username in wlist['screen_names'].values:
                is_on_watchlist = 1
            else:
                is_on_watchlist = 0

            stats = transform_stats(username, following_watchlist, watchlist_completion,
                                    likes_watchlist, retweets_watchlist, mentions_watchlist,
                                    watchword_in_bio, is_on_watchlist)
            stats = pd.DataFrame(stats)
            file_path = file_path + username + ".xlsx"
            cols = ["user", "following_watchlist", "watchlist_completion", "likes_watchlist",
                    "retweets_watchlist", "mentions_watchlist", "watchword_in_bio",
                    "is_on_watchlist"]
            write_excel(file_path, stats, cols, "user stats", "a")

            if is_on_watchlist == 0 or (is_on_watchlist == 1 and
                                        (following_watchlist != 0 or watchlist_completion != 0 or
                                         likes_watchlist != 0 or retweets_watchlist != 0 or
                                         mentions_watchlist != 0)):
                if exists(database, (user_id,)):
                    user_row = (username, following_watchlist, watchlist_completion,
                                likes_watchlist, retweets_watchlist, mentions_watchlist,
                                watchword_in_bio, is_on_watchlist, user_id)
                    update(database, user_row)
                else:
                    user_row = (user_id, username, following_watchlist, watchlist_completion,
                                likes_watchlist, retweets_watchlist, mentions_watchlist,
                                watchword_in_bio, is_on_watchlist)
                    insert(database, user_row)

                database.commit()

            else:
                print("Skipped insert: " + username)

        print("PID " + str(getpid()) + " END " +
              username + " " + str(datetime.now()))
    except Exception as exception:
        print(str(getpid()) + " EXCEPTION")
        print(exception)
        print(traceback.format_exc())

    database.close()


def pool_handler(work_items):
    '''Pool handler for distributing work amongst processes.'''

    pool = Pool(cpu_count())
    print("Parallel processing on " + str(cpu_count()) + " cores.")
    pool.map_async(process_user, work_items)
    pool.close()
    pool.join()


# GLOBAL VARS
# Below are settings intended for a resiliency feature.
ATTEMPT_LIMIT = 5
SLEEP_TIME = 5000

# MAIN
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "watchlist", help="group of twitter users who are interesting")
    PARSER.add_argument(
        "output", help="specify output database")
    PARSER.add_argument(
        "--bio_watchwords", help="list of watchwords to look for in bio unique to the target group")
    PARSER.add_argument(
        "--tweet_watchwords",
        help="list of watchwords to look for in tweets unique to the target group")
    PARSER.add_argument(
        "--tweet_fetch_limit", help="number of tweets to fetch when calculating statistics",
        default=100, type=int)
    GROUP = PARSER.add_mutually_exclusive_group(required=True)
    GROUP.add_argument(
        '--username', help="username of single user to fetch data on")
    GROUP.add_argument(
        '--userlist', help="csv containing usernames to fetch data on")

    ARGS = PARSER.parse_args()

    WATCHLIST = import_csv(ARGS.watchlist, "screen_names")

    if ".db" not in ARGS.output:
        ARGS.output = ARGS.output + ".db"

    # Create an output directory, since several XLSX files will be produced.
    DIRECTORY = "./" + ARGS.output.replace(".db", "") + "/"
    if not path.exists(DIRECTORY):
        mkdir(DIRECTORY)

    DB = create_connection(DIRECTORY + ARGS.output)
    if not exists_table(DB):
        create(DB)
    DB.close()

    BIO_WATCHWORDS = import_csv(ARGS.bio_watchwords, "watchwords")
    TWEET_WATCHWORDS = import_csv(ARGS.tweet_watchwords, "watchwords")

    if ARGS.username is not None:
        USERS = [ARGS.username]

    if ARGS.userlist is not None:
        USERS = import_csv(ARGS.userlist, "screen_names")
        USERS = USERS["screen_names"].values

    # Build work queue.
    WORK = []
    for user in USERS:
        work_item = {
            "user": user,
            "db": ARGS.output,
            "directory": DIRECTORY,
            "watchlist": WATCHLIST,
            "tweet_fetch_limit": ARGS.tweet_fetch_limit,
            "tweet_watchwords": TWEET_WATCHWORDS,
            "bio_watchwords": BIO_WATCHWORDS
        }

        WORK.append(work_item)

    # Start work in multiple processes.
    pool_handler(WORK)

    DB.close()
