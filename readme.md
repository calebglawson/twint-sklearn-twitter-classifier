**Twint, the library used to fetch tweets, is incompatible with Python 3.8 at this time.  Please use Python 3.7 or 3.6 with these scripts until this message is removed.**

# Summary
This project takes in a list of twitter accounts, collects stats, trains a classification model, predicts class based on stats, and returns the data collected to create the stats.

**Remember, in order to make quality predictions with a machine learning model, you will need quality data. I have been able to achieve a 94% average accuracy with this tool. With the right input, maybe you can, too.**

# Typical Workflow
## If You Do Not Have a Machine Learning Model
1. Have a list of users that belong to the targed class, also known as a watchlist.
2. Obtain a list with an equal number of users that are neutral / do not belong to the targeted class with fetch_followers.py **Example:** ```python .\fetch_followers.py example_user --filter .\watchlist.csv --output example_user_followers.csv --limit 800```
3. Unify the lists.
4. Run gather_data.py to fetch the stats, supply all of the script params in the command line. **Example:** ```python .\gather_data.py .\users.csv .\watchlist.csv results.db --bio_watchwords .\bio_watchwords.csv```
5. Run generate_model.py on the fetched data to generate a model. **Example:** ```python .\generate_model.py .\results.db --model_output model.joblib --test_output test_results.csv```
## Once You Have a Machine Learning Model
1. Run gather_data.py to fetch stats on a new set of users that you would like to use with the model. **Example:** ```python .\gather_data.py .\watchlist.csv results.db --bio_watchwords .\bio_watchwords.csv --tweet_watchwords ./tweet_watchwords.csv --userlist .\users.csv```
2. Run predict.py on the new set of users and view the results. **Example:** ```python .\predict.py .\results.db .\model.joblib --output results_predictions.csv```
3. Use the file generated from predict.py to find users of interest. Take a deeper look into an individual user by reviewing their report generated be gather_data.py.

# Required Libraries
* argparse
* joblib
* numpy
* openpyxl or xlsxwriter
* pandas
* scipy
* sklearn
* sqlite3
* twint

# Optional Libraries
* matplotlib
* seaborn

# Additional Recommended Tools
* Anaconda
* Jupyter Notebook
* DB Browser (SQLite)
* LibreOffice Calc or Excel

# File Index
## Scripts
* fetch_followers.py
    * Fetch the followers of a given user ID. Optionally filter and limit results. Outputs CSV.
* gather_data.py
    * From a list of twitter users, gather the twitter data, persist it, and compute stats. Outputs a single SQLite Database the machine learning model and Excel reports for humans.
* generate_model.py
    * From a DB outputted by gather_data.py, generate a machine learning model which will classify twitter users in one of two classes. Outputs .joblib of the model and test results.
* predict.py
    * From a DB outputted by gather_data.py, load a machine learning model outputted by generate_model.py and classify the twitter users. Outputs Excel file of the results.
## Database
* *.db
    * The computed stats outputted by gather_data.py

## CSVs
* watchlist.csv
    * This is a list of the users we want to watch for in other user's twitter interactions. Used to compute stats.
* bio_watchwords.csv
    * This is a list of words to watch for in a user's bio. Used to compute stats.
* tweet_watchwords.csv
    * This is a list of words to look for in a user's tweets. Used in data review only.
* users.csv
    * This is an example input file for gather_data.py. Contains screen names.

## Other
* *.joblib
    * This is the joblibbed SVC model.
* Visualize.ipynb
    * This is a Jupyter notebook used to visualize the stats db generated by gather_data.py