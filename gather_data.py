'''From a list of twitter users, gather the twitter data, persist it, and compute stats.
Outputs a single SQLite Database the machine learning model and Excel reports for humans.'''


from multiprocessing import Pool, cpu_count
from os import getpid, mkdir, path
from datetime import datetime
from math import ceil
import argparse
import sqlite3
import traceback
import twint
import pandas as pd


def create_connection(db_file):
    '''Create a connection to the stats output DB.'''

    conn = sqlite3.connect(db_file, isolation_level=None)
    return conn


def create(conn):
    '''Create the stats table.'''

    sql = '''CREATE TABLE user_stats ( id TEXT,	username TEXT, last_updated INTEGER,
    following_watchlist REAL, watchlist_completion REAL, likes_watchlist REAL,
    retweets_watchlist REAL, mentions_watchlist REAL, watchword_in_bio INGEGER,
    is_on_watchlist INTEGER, PRIMARY KEY(id) )'''
    cur = conn.cursor()
    cur.execute(sql)


def insert(conn, user_stats):
    '''Insert a row into the stats table.'''

    sql = '''INSERT INTO user_stats (id, username, last_updated, following_watchlist,
    watchlist_completion, likes_watchlist, retweets_watchlist, mentions_watchlist,
    watchword_in_bio, is_on_watchlist) VALUES (?,?,current_timestamp,?,?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, user_stats)


def update(conn, user_stats):
    '''Update a row in the stats table.'''

    sql = '''UPDATE user_stats SET username = ?, last_updated = current_timestamp,
    following_watchlist = ?, watchlist_completion = ?, likes_watchlist = ?,
    retweets_watchlist = ?, mentions_watchlist = ?, watchword_in_bio = ?,
    is_on_watchlist = ? WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, user_stats)


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
    if not dataframe.empty:
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

    data = []

    if filename is not None:
        detect_file_header(filename, header)

        data = pd.read_csv(filename)
        data[header] = data[header].apply(
            lambda x: x.lower())

        if header == "watchwords":
            data = data[header].values
    else:
        if header != "watchwords":
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

    user_id = None
    private = 1
    num_following = 0
    bio = None

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

    except Exception as exception:  # pylint: disable=broad-except
        print(f"Failed to fetch {username} user info.")
        print(exception)

    return user_id, private, num_following, bio


def fetch_following(username):
    '''Fetch users that the user is following.'''

    config = twint.Config()
    config.Username = username
    config.Pandas = True
    config.Hide_output = True

    attempt = 0
    followers_dataframe = pd.DataFrame()

    while followers_dataframe.empty and attempt < MAX_ATTEMPTS:
        twint.storage.panda.Follow_df = pd.DataFrame()
        twint.run.Following(config)
        followers_dataframe = twint.storage.panda.Follow_df

        attempt += 1

    follower_list = pd.DataFrame(columns=[0])
    if not followers_dataframe.empty:
        for index, row in followers_dataframe.iteritems():  # pylint: disable=unused-variable
            follower_list = pd.DataFrame(row[0])
            follower_list[0] = follower_list[0].apply(lambda x: x.lower())

    return follower_list


def calculate_following_stats(num_following, username, follower_list, watchlist, file_path):
    '''Calculate following watchlist ratio and watchlist completion ratio.'''

    following_watchlist = 0
    watchlist_completion = 0

    if num_following > 0:
        watchlist_following = follower_list[follower_list[0].isin(
            watchlist["screen_names"])]

        file_path = file_path + username + ".xlsx"
        cols = [0]
        write_excel(file_path, watchlist_following, cols,
                    "following watchlist", "a")

        if not watchlist_following.empty:
            following_watchlist = len(watchlist_following[0])/num_following
            watchlist_completion = len(
                watchlist_following[0])/len(watchlist["screen_names"])

    return following_watchlist, watchlist_completion


def following(username, num_following, watchlist, file_path):
    '''Wrapper for fetching and calculating following stats.'''

    follower_list = fetch_following(username)
    following_watchlist, watchlist_completion = calculate_following_stats(
        num_following, username, follower_list, watchlist, file_path)

    return following_watchlist, watchlist_completion


def fetch_likes(username, limit):
    '''Fetch a user's likes with twint.'''

    config = twint.Config()
    config.Username = username
    config.Store_object = True
    config.Hide_output = True
    config.Limit = limit

    attempt = 0
    fetched_tweets = []
    twint.output.tweets_list = []

    while not fetched_tweets and attempt < MAX_ATTEMPTS:
        twint.run.Favorites(config)
        fetched_tweets = twint.output.tweets_list

        attempt += 1

    dataframe = twint_obj_list_to_dataframe(fetched_tweets)

    return dataframe


def calculate_like_stats(dataframe, watchlist, username, file_path):
    '''Calculate the watchlist vs non-watchlist tweet like ratio.'''

    watchlist_intersect = 0

    if not dataframe.empty:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"] != username]

        dataframe_on_watchlist = dataframe[dataframe["username"].isin(
            watchlist['screen_names'])]

        if not dataframe_on_watchlist.empty and not dataframe.empty:
            watchlist_intersect = dataframe_on_watchlist["username"].count(
            ) / len(dataframe.index)

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'username', 'likes_count', 'retweets_count', 'replies_count', 'datestamp',
                'timestamp', 'timezone', 'link']
        write_excel(file_path, dataframe_on_watchlist,
                    cols, "watchlist likes", "a")

    return watchlist_intersect


def likes(username, limit, watchlist, file_path):
    '''Wrapper function for fetching and calculating like stats.'''

    dataframe = fetch_likes(username, limit)
    likes_watchlist = calculate_like_stats(
        dataframe, watchlist, username, file_path)

    return likes_watchlist


def fetch_tweets(username, limit):
    '''Fetch tweets for stats calculation.'''

    config = twint.Config()
    config.Username = username
    config.Store_object = True
    config.Retweets = True
    config.Profile_full = True
    config.Hide_output = True
    config.Limit = limit

    attempt = 0
    twint.output.tweets_list = []
    fetched_tweets = []
    retweets = pd.DataFrame()
    mentions = pd.DataFrame()
    all_tweets = pd.DataFrame()

    while not fetched_tweets and attempt < MAX_ATTEMPTS:
        twint.run.Profile(config)
        fetched_tweets = twint.output.tweets_list

        attempt += 1

    dataframe = twint_obj_list_to_dataframe(fetched_tweets)

    if not dataframe.empty:
        # The retweets flag is broken in Twint 2.1.7
        retweets = dataframe[dataframe["username"] != username]
        # Pandas dataframe is broken for tweets in Twint 2.1.7 & 6
        mentions = dataframe[dataframe["mentions"] != "[]"]
        all_tweets = dataframe

    return retweets, mentions, all_tweets


def calculate_retweets(retweets, watchlist, username, file_path):
    '''Calculate watchlist vs non-watchlist retweet ratio.'''

    watchlist_intersect_retweets = 0

    if not retweets.empty:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"] != username]

        watchlist_retweets = retweets[retweets["username"].isin(
            watchlist['screen_names'])]

        if not watchlist_retweets.empty and not retweets.empty:
            watchlist_intersect_retweets = watchlist_retweets["username"].count(
            ) / len(retweets["username"])

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'username', 'likes_count', 'retweets_count', 'replies_count',
                'datestamp', 'timestamp', 'timezone', 'link']
        write_excel(file_path, watchlist_retweets,
                    cols, "watchlist retweets", "a")

    return watchlist_intersect_retweets


def calculate_mentions(mentions, watchlist, username, file_path):
    '''Calculate watchlist vs non-watchlist mention ratio.'''

    watchlist_intersect_mentions = 0

    if not mentions.empty:
        # Don't count self.
        watchlist = watchlist[watchlist["screen_names"] != username]

        filtered_mentions = []
        for index, row in mentions.iterrows():  # pylint: disable=unused-variable
            if row["username"] == username:
                for mention in row["mentions"]:
                    if mention in watchlist['screen_names'].values and mention != username:
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

        if not mention_usernames.empty:
            watchlist_intersect_mentions = mention_usernames[mention_usernames["username"].isin(
                watchlist['screen_names'])]["username"].count() / len(mention_usernames.index)

        file_path = file_path + username + ".xlsx"
        cols = ['tweet', 'mentions', 'username', 'likes_count', 'retweets_count', 'replies_count',
                'datestamp', 'timestamp', 'timezone', 'link']
        write_excel(file_path, filtered_mentions,
                    cols, "watchlist mentions", "a")

    return watchlist_intersect_mentions


def find_watchword_tweets(all_tweets, watchwords, username, file_path):
    '''Search for tweets with watchwords and phrases for Excel output only.'''
    if not all_tweets.empty and watchwords.size != 0:
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


def calculate_tweet_stats(retweets, mentions, all_tweets,
                          watchlist, watchwords, username, file_path):
    '''Wrapper for calculating various tweet stats.'''

    watchlist_intersect_retweets = calculate_retweets(
        retweets, watchlist, username, file_path)
    watchlist_intersect_mentions = calculate_mentions(
        mentions, watchlist, username, file_path)
    find_watchword_tweets(all_tweets, watchwords, username, file_path)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def tweets(username, limit, watchlist, watchwords, file_path):
    '''Wrapper function for fetching tweets and calculating stats.'''

    retweets, mentions, all_tweets = fetch_tweets(username, limit)
    watchlist_intersect_retweets, watchlist_intersect_mentions = calculate_tweet_stats(
        retweets, mentions, all_tweets, watchlist, watchwords, username, file_path)

    return watchlist_intersect_retweets, watchlist_intersect_mentions


def calculate_bio_stats(bio, watchwords):
    '''Search the bio for watchwords.'''
    watchword_in_bio = 0

    if watchwords.size != 0:
        for watchword in watchwords:
            if watchword in bio:
                watchword_in_bio = 1
                break

    return watchword_in_bio


def process_user(item):
    '''Main function for processing a user.'''

    try:
        print(
            f"PID {str(getpid())} START {item['user']} {str(datetime.now())}")

        item["db"] = create_connection(item["directory"] + item["db"])
        stats = {"user": item["user"]}

        user_id, private, following_count, bio = get_user_info(
            item["user"], item["directory"])
        # If I  can't get user ID or if profile is private, stats are useless.
        if user_id is not None and private != 1:

            stats["following_watchlist"], stats["watchlist_completion"] = following(
                item["user"], following_count, item["watchlist"], item["directory"])
            stats["likes_watchlist"] = likes(
                item["user"], item["tweet_fetch_limit"], item["watchlist"], item["directory"])
            stats["retweets_watchlist"], stats["mentions_watchlist"] = tweets(
                item["user"], item["tweet_fetch_limit"],
                item["watchlist"], item["tweet_watchwords"], item["directory"])
            stats["watchword_in_bio"] = calculate_bio_stats(
                bio, item["bio_watchwords"])

            if item["user"] in item["watchlist"]['screen_names'].values:
                stats["is_on_watchlist"] = 1
            else:
                stats["is_on_watchlist"] = 0

            excel_stats = pd.DataFrame([stats])
            file_path = f"{item['directory']}{item['user']}.xlsx"
            cols = ["user", "following_watchlist", "watchlist_completion", "likes_watchlist",
                    "retweets_watchlist", "mentions_watchlist", "watchword_in_bio",
                    "is_on_watchlist"]
            write_excel(file_path, excel_stats, cols, "user stats", "a")

            non_zero_stats = stats["following_watchlist"] != 0 or \
                stats["watchlist_completion"] != 0 or stats["likes_watchlist"] != 0 \
                or stats["retweets_watchlist"] != 0 or stats["mentions_watchlist"] != 0
            if stats["is_on_watchlist"] == 0 or (stats["is_on_watchlist"] == 1 and non_zero_stats):
                if exists(item["db"], (user_id,)):
                    user_row = (item["user"], stats["following_watchlist"],
                                stats["watchlist_completion"], stats["likes_watchlist"],
                                stats["retweets_watchlist"], stats["mentions_watchlist"],
                                stats["watchword_in_bio"], stats["is_on_watchlist"], user_id)
                    update(item["db"], user_row)
                else:
                    user_row = (user_id, item["user"], stats["following_watchlist"],
                                stats["watchlist_completion"], stats["likes_watchlist"],
                                stats["retweets_watchlist"], stats["mentions_watchlist"],
                                stats["watchword_in_bio"], stats["is_on_watchlist"])
                    insert(item["db"], user_row)

                item["db"].commit()

            else:
                print(f"Skipped insert: {item['user']}")

        print(
            f"PID {str(getpid())} END {item['user']} {str(datetime.now())}")
    except Exception as exception:  # pylint: disable=broad-except
        # Okay, I actually need this one, because if anything is uncaught,
        # processes fail silently and do not pick up new work.
        print(f"{str(getpid())} EXCEPTION")
        print(exception)
        print(traceback.format_exc())

    item['db'].close()


def pool_handler(work_items, pool_worker_scaling):
    '''Pool handler for distributing work amongst processes.'''

    num_worker_threads = ceil(float(cpu_count()) * pool_worker_scaling)

    if num_worker_threads <= 0:
        num_worker_threads = 1

    pool = Pool(num_worker_threads)
    print(f"Processing with {str(num_worker_threads)} worker process(es).")
    pool.map_async(process_user, work_items)
    pool.close()
    pool.join()


def build_work_queue(args):
    '''Build queue with work objs for pool.'''
    work = []
    for user in args.userlist:
        work_item = {
            "user": user,
            "db": args.output,
            "directory": args.directory,
            "watchlist": args.watchlist,
            "tweet_fetch_limit": args.tweet_fetch_limit,
            "tweet_watchwords": args.tweet_watchwords,
            "bio_watchwords": args.bio_watchwords
        }

        work.append(work_item)

    return work


def fetch_args():
    '''Use argparse if main.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "watchlist", help="group of twitter users who are interesting")
    parser.add_argument(
        "--bio_watchwords", help="list of watchwords to look for in bio unique to the target group")
    parser.add_argument(
        "--tweet_watchwords",
        help="list of watchwords to look for in tweets unique to the target group")
    parser.add_argument(
        "--pool_worker_scaling",
        help="multiplied against cpu_count to determine number of worker threads",
        default=1.0, type=float)
    parser.add_argument(
        "--tweet_fetch_limit", help="number of tweets to fetch when calculating statistics",
        default=100, type=int)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--username', help="username of single user to fetch data on")
    group.add_argument(
        '--userlist', help="csv containing usernames to fetch data on")

    args = parser.parse_args()
    args.output = None

    return args


def massage(args):
    '''Massage the arg data to build work.'''

    args.watchlist = import_csv(args.watchlist, "screen_names")

    if args.userlist is not None:
        args.output = args.userlist.split("/")[-1]
        args.output = args.userlist.split("\\")[-1]
        args.output = args.output.split('.')[0]

        args.userlist = import_csv(args.userlist, "screen_names")
        args.userlist = args.userlist["screen_names"].values
    else:
        args.output = args.username
        args.userlist = [args.username]

    if ".db" not in args.output:
        args.output = f"{args.output}.db"

    # Create an output directory, since several XLSX files will be produced.
    args.directory = f"./{args.output.replace('.db', '')}/"
    if not path.exists(args.directory):
        mkdir(args.directory)

    database = create_connection(args.directory + args.output)
    if not exists_table(database):
        create(database)
    database.close()

    args.bio_watchwords = import_csv(args.bio_watchwords, "watchwords")
    args.tweet_watchwords = import_csv(args.tweet_watchwords, "watchwords")

    if args.username is not None:
        if args.pool_worker_scaling != 1:
            print(
                "Argument --username is provided, supplied --pool_worker_scaling is ignored.")
        args.pool_worker_scaling = -1

    return args


def run(args):
    '''Main function.'''
    args = massage(args)
    work = build_work_queue(args)
    pool_handler(work, args.pool_worker_scaling)


# GLOBAL
MAX_ATTEMPTS = 5

# MAIN
if __name__ == '__main__':
    ARGS = fetch_args()
    run(ARGS)
