
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

df2 = df[['comment', 'label']]

df2.to_csv('reddit_labels.csv', encoding='utf-8', index=False)

df.label.value_counts()
