''' This script converts a CSV of user IDs to screen names. '''
from pathlib import Path
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
        lines.insert(0, header + "\n")
        csv.writelines(lines)
        csv.close()


def fetch_user_info(user_id):
    '''Fetch twitter user profile info with twint.'''

    config = twint.Config()
    config.User_id = user_id
    config.Pandas = True
    config.User_full = True
    config.Hide_output = True

    try:
        twint.run.Lookup(config)
        users_dataframe = twint.storage.panda.User_df
        username = users_dataframe.iloc[-1]["username"]
    except TypeError:
        username = user_id
        print(f"Failed to fetch username for id: {user_id}")

    return username


# MAIN
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "userlist", help="CSV with user ids")

    ARGS = PARSER.parse_args()

    detect_file_header(ARGS.userlist, "user_ids")

    USER_IDS = pd.read_csv(ARGS.userlist)

    ARGS.output = Path(ARGS.userlist).stem

    FAILURES = []
    SUCCESSES = []
    for user in USER_IDS["user_ids"].values:
        user = str(fetch_user_info(user))

        print(user)

        if user.isnumeric():
            FAILURES.append(user)
        else:
            SUCCESSES.append(user)

    SUCCESSES = pd.DataFrame(SUCCESSES, columns=['screen_names'])
    SUCCESSES["screen_names"] = SUCCESSES["screen_names"].apply(
        lambda x: x.lower())
    SUCCESSES.to_csv(f"{ARGS.output}_screen-names.csv")

    print(f"Results output to: {ARGS.output}_screen-names.csv")
    print(f"These records failed: {FAILURES}")
