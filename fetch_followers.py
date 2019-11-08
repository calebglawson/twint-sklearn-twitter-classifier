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

    max_attempts = 5
    attempt = 0
    follower_list = pd.DataFrame()

    while len(follower_list.index) == 0 and attempt < max_attempts:
        try:
            follower_list = fetch_following(args.username, args.limit)
        except:
            print(f"Attempt {str(attempt)} of {str(max_attempts)} failed.")

        attempt += 1

    if len(follower_list.index) > 0:
        watchlist = import_csv(args.filter, "screen_names")

        if len(watchlist.index) > 0:
            filtered_followers = []
            for follower in follower_list[0]:
                if follower not in watchlist['screen_names'].values:
                    filtered_followers.append(follower.lower())
            follower_list = pd.DataFrame(
                filtered_followers, columns=['screen_names'])

        if len(follower_list.index) > 0:
            follower_list.to_csv(args.output, quoting=csv.QUOTE_NONNUMERIC)
            print('Output: ' + args.output)
        else:
            print("No followers to output.")
