**Twint, the library used to fetch tweets, is incompatible with Python 3.8 at this time.  Please use Python 3.7 or 3.6 with these scripts until this message is removed.** [twintproject/twint#569](https://github.com/twintproject/twint/issues/569)

# Summary
This project takes in a list of twitter accounts, collects stats, trains a classification model, predicts class based on stats, and returns the data collected to create the stats.

**Remember, in order to make quality predictions with a machine learning model, you will need quality data. I have been able to achieve a 94% average accuracy with this tool. With the right input, maybe you can, too.**

# Typical Workflow
## If You Do Not Have a Machine Learning Model
1. Have a list of users that belong to the targed class, also known as a watchlist.
2. Obtain a list with an equal number of users that are neutral / do not belong to the targeted class with fetch_followers.py **Example:** ```python .\fetch_followers.py example_user --filter .\watchlist.csv --output example_user_followers.csv --limit 800```
3. Unify the lists.
4. Run model_pipeline.py to gather the data and generate a model. **Example:** ```python .\model_pipeline.py .\watchlist.csv --userlist .\intake.csv --tweet_watchwords .\tweet_watchwords.csv --bio_watchwords .\bio_watchwords.csv```
## Once You Have a Machine Learning Model
1. Run prediction_pipeline.py to gether the data and run predictions against the model. **Example:** ```python .\prediction_pipeline.py .\watchlist.csv .\intake\intake.joblib --userlist .\user_list.csv --tweet_watchwords .\tweet_watchwords.csv --bio_watchwords .\bio_watchwords.csv```
3. Use the file generated from predict.py to find users of interest. Take a deeper look into an individual user by reviewing their report generated by gather_data.py.

# Required Libraries
* argparse
* joblib
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
* model_pipeline.py
    * Automated pipeline that combines gather_data and generate_model to build a new, trained model.
* prediction_pipeline.py
    * Automated pipeline that combines gather_data and predict to make predicitons off an existing model.
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