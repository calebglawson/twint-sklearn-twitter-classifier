''' This script fetches followers for a given user and outputs a CSV. '''

from csv import QUOTE_NONNUMERIC
import argparse
import pandas as pd
import twint


def detect_file_header(filename, header):
    ''' Detects if the given header is in the file.
        If the header is not found, one is added. '''

    csv = open(filename, "r")
    lines = csv.readlines()

    if header not in lines[0]:
        csv.close()
        csv = open(filename, "w")
        lines.insert(0, header+"\n")
        csv.writelines(lines)
        csv.close()


def import_csv(filename, header):
    ''' Import the given CSV as a Pandas DataFrame or arrray. '''

    if header == "watchwords":
        data = []
    else:
        data = pd.DataFrame()

    if filename is not None:
        detect_file_header(filename, header)

        data = pd.read_csv(filename)
        data[header] = data[header].apply(
            lambda x: x.lower())

        if header == "watchwords":
            data = data[header].values

    return data


def fetch_following(username, limit):
    ''' Wrapper function for the twint fetch. '''
    config = twint.Config()
    config.Username = username
    config.Pandas = True
    config.Hide_output = False

    if limit is not None:
        config.Limit = limit

    twint.storage.panda.Follow_df = pd.DataFrame()
    twint.run.Followers(config)

    followers_df = twint.storage.panda.Follow_df
    followers_df = pd.DataFrame(followers_df['followers'][username])
    followers_df[0] = followers_df[0].apply(lambda x: x.lower())

    return followers_df


# MAIN
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "username", help="twitter user from whom to fetch the followers")
    PARSER.add_argument("--filter", help="csv file of individuals to filter")
    PARSER.add_argument(
        "--output", help="specify output filename", default="followers.csv")
    PARSER.add_argument("--limit", help="max followers to fetch", type=int)

    ARGS = PARSER.parse_args()

    if ARGS.output is not None:
        if ".csv" not in ARGS.output:
            ARGS.output = f"{ARGS.output}.csv"

    FOLLOWER_LIST = fetch_following(ARGS.username, ARGS.limit)

    if not FOLLOWER_LIST.empty:
        WATCHLIST = import_csv(ARGS.filter, "screen_names")

        if not WATCHLIST.empty:
            FILTERED_FOLLOWERS = []
            for follower in FOLLOWER_LIST[0]:
                if follower not in WATCHLIST['screen_names'].values:
                    FILTERED_FOLLOWERS.append(follower.lower())
            FOLLOWER_LIST = pd.DataFrame(
                FILTERED_FOLLOWERS, columns=['screen_names'])

        if not FOLLOWER_LIST.empty:
            FOLLOWER_LIST.to_csv(ARGS.output, quoting=QUOTE_NONNUMERIC)
            print(f"Output: {ARGS.output}")
        else:
            print("No followers to output.")
