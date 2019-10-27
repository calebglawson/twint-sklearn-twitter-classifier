import argparse
import csv
import pandas as pd
import twint


def detect_file_header(filename, header):
    f = open(filename, "r")
    lines = f.readlines()

    if header not in lines[0]:
        f.close()
        f = open(filename, "w")
        lines.insert(0, header+"\n")
        f.writelines(lines)
        f.close()


def import_watchlist(filename, header):

    detect_file_header(filename, header)

    watchlist = pd.read_csv(filename)
    watchlist[header] = watchlist[header].apply(
        lambda x: x.lower())
    return watchlist


def fetch_following(username, limit):
    c = twint.Config()
    c.Username = username
    c.Pandas = True
    c.Hide_output = False

    if limit != None:
        c.Limit = limit

    twint.storage.panda.Follow_df = pd.DataFrame()
    twint.run.Followers(c)

    Followers_df = twint.storage.panda.Follow_df
    try:
        follower_list = pd.DataFrame(Followers_df['followers'][username])
        follower_list[0] = follower_list[0].apply(lambda x: x.lower())
    except:
        follower_list = pd.DataFrame()

    return follower_list


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "username", help="twitter user from whom to fetch the followers")
    parser.add_argument("--filter", help="csv file of individuals to filter")
    parser.add_argument(
        "--output", help="specify output filename", default="followers.csv")
    parser.add_argument("--limit", help="max followers to fetch", type=int)

    args = parser.parse_args()

    if args.output != None:
        if ".csv" not in args.output:
            args.output = args.output + ".csv"

    follower_list = fetch_following(args.username, args.limit)

    if args.filter != None:
        watchlist = import_watchlist(args.filter, "screen_names")
    else:
        watchlist = pd.DataFrame()
        watchlist['screen_names'] = pd.Series()

    filtered_followers = []
    for follower in follower_list[0]:
        if follower not in watchlist['screen_names'].values:
            filtered_followers.append(follower.lower())

    filtered_followers = pd.DataFrame(
        filtered_followers, columns=['screen_names'])
    filtered_followers.to_csv(args.output, quoting=csv.QUOTE_NONNUMERIC)

    print('Output: ' + args.output)
