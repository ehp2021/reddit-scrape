#pip install praw
# pip install pandas


# PROGRESS STATUS
import time
import sys

for i in range(100):
    time.sleep(1)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()

from pprint import pprint
import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
import os

load_dotenv()

# REDDIT SCRAPE START

# https://praw.readthedocs.io/en/stable/
import praw

# Reddit API Setup: https://www.reddit.com/prefs/apps
user_agent = "scraper 1.0 by /u/Coconuts2018"

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    check_for_async=False
)

#filter for BMRN comments
#https://praw.readthedocs.io/en/latest/code_overview/models/subreddit.html#praw.models.Subreddit.search

comments = []
dates = []

for submission in reddit.subreddit("all").search("Biomarin", syntax='lucene', sort='relevance', time_filter='all'):
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments.append(comment.body)
        dates.append(comment.created_utc)

print(len(comments))

import pandas as pd
df = pd.DataFrame({"comments": comments, "date": dates})
df.to_csv("comments.csv", index=False)
df.head(5)


import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []


#auto score each headline
for line in comments:
  pol_score = sia.polarity_scores(line) # -> dict
  pol_score['comment'] = line
  results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()

#default
df['label'] = 0

#greater than 0.2 is positive
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()

# Select columns for output
df_out = df[['comment', 'label', 'compound', 'neg', 'neu', 'pos']]

# Write output to csv file
df_out.to_csv('reddit_labels.csv', encoding='utf-8', index=False)

# Print value counts of the label column
print(df['label'].value_counts())
