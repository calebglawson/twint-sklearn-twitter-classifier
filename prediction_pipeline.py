'''This script combines gather_data and generate_model to form a simpler pipeline.'''

import argparse
import gather_data
import predict


def massage(args):
    '''Below is used to deduce the name of the DB.'''

    if args.userlist is not None:
        args.output = args.userlist.split("/")[-1]
        args.output = args.userlist.split("\\")[-1]
        args.output = args.output.split('.')[0]
    else:
        args.userlist = args.username
        args.output = args.username

    folder = f".\\{args.output}\\"
    args.database = f"{folder}{args.output}.db"

    return args


def fetch_args():
    '''Use argparse if main.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "watchlist", help="group of twitter users who are interesting")
    parser.add_argument(
        "model", help="filename of the joblibbed model")
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
        '--userlist', help="csv containing usernames to fetch data on")
    group.add_argument(
        '--username', help="username of single user to fetch data on")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = fetch_args()
    ARGS = massage(ARGS)
    gather_data.run(ARGS)
    predict.run(ARGS)
