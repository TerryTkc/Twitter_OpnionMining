import json
import util
import os
import re
import sys
import json
import vaderSentiment
import csv

from vaderSentiment.vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from pytz import timezone

cities = ["new_york_city", "delaware", "cheyenne", "miami", "omaha", "san_jose"]
# cities = ["new_york_city"]

def load_city_dict(city):
    with open("city_data/{}.json".format(city)) as city_file:
        city_dict = json.load(city_file)
        city_file.close()
        return city_dict

# Save into a csv file
def save_csv(folder, header, lists, file_name):
	if not os.path.exists(folder):
		os.makedirs(folder)

	out_file = open("{}/{}.csv".format(folder, file_name),'w')
	mywriter = csv.writer(out_file)

	mywriter.writerow(header)

	for rows in lists:
		mywriter.writerow(rows)

	out_file.close()

# Remove all the duplicates in the json
def remove_duplicates(city_dict, city):
	unique_city = { each['id_str'] : each for each in city_dict }.values()

	return unique_city

# Add the Vader sentiment result to the sentiment dict
def add_sentiment(sentiment_dict, result):
    sentiment_dict["neg"] += result["neg"]
    sentiment_dict["neu"] += result["neu"]
    sentiment_dict["pos"] += result["pos"]
    sentiment_dict["compound"] += result["compound"]


# Normalize the Vader sentiment result
def normalize_sentiment(sentiment_dict, n):
    sentiment_dict["neg"] /= n
    sentiment_dict["neu"] /= n
    sentiment_dict["pos"] /= n
    sentiment_dict["compound"] /= n

# Clean the tweet
def clean_text(tweet):
	tweet = re.sub(r"http\S+|@\S+", "", tweet)  
	return tweet

# Make all the tweets time into their local time
def standardize_timezone(time, city):
	switcher = {

		"miami": "US/Eastern",
		"new_york_city": "US/Eastern",
		"omaha": "US/Central",
		"san_jose": "US/Pacific",
		"cheyenne": "US/Mountain",
		"delaware": "US/Eastern"

	}

	fmt = "%Y-%m-%d %H:%M:%S"
	new_time = datetime.strptime(time, fmt)
	new_time = new_time.astimezone(timezone(switcher.get(city, "NULL")))

	return new_time


def calculate_overall_scores():
	city_name = []
	tweets_count = []
	negative = []
	neutral = []
	positive = []
	compound = []
	polarity = []
	subjectivity = []

	for city in cities:
		city_dict = load_city_dict(city)
		city_dict = remove_duplicates(city_dict, city)

	    
		analyzer = SentimentIntensityAnalyzer()

		sentiment_results = {"neg": 0, "neu": 0, "pos": 0, "compound":0}

		tweet_count = 0

		for tweet in city_dict:
			tweet_text = tweet["text"]
			tweet_text = clean_text(tweet_text)

			if(tweet_text[:2] == 'RT'):
				continue

			tweet_count += 1

			try:
				sentiment = analyzer.polarity_scores(tweet_text)
				add_sentiment(sentiment_results, sentiment)
			except:
				continue

		normalize_sentiment(sentiment_results, tweet_count)

		city_name.append(city)
		tweets_count.append(tweet_count)
		negative.append(round(sentiment_results["neg"],4))
		neutral.append(round(sentiment_results["neu"],4))
		positive.append(round(sentiment_results["pos"],4))
		compound.append(round(sentiment_results["compound"],4))

	folder = "csv_file/Overall"
	header = ['city', 'tweet_count','neg', 'neu', 'pos', 'compound']
	file_name = "00 Overall"
	lists = zip(city_name, tweets_count, negative, neutral, positive, compound)

	save_csv(folder, header, lists, file_name)


# # Calculate each tweet's sentiment result
def tweets_csv(city_dict, city):
	folder = "csv_file/City"
	header = ["created_at", "city", "retweet_count", "favorite_count", "followers_count", "friends_count", "listed_count", "favorites_count", 
		 "statuses_count", "text", "negative", "neutral", "positive", "compound"]
	file_name = "{}_tweets".format(city)


	retweet_count = []
	favorite_count = []
	followers_count = []
	friends_count = []
	listed_count = []
	favorites_count = []
	statuses_count = []


	created_at = []
	text = []
	negative = []
	neutral = []
	positive = []
	compound = []

	city_list = []


	analyzer = SentimentIntensityAnalyzer()

	for tweet in city_dict:
		tweet_text = tweet["text"]
		tweet_text = clean_text(tweet_text)

		if(tweet_text[:2] == 'RT'):
			continue



		try:
			sentiment = analyzer.polarity_scores(tweet_text)
		except:
			continue

		retweet_count.append(tweet["retweet_count"])
		favorite_count.append(tweet["favorite_count"])
		followers_count.append(tweet["followers_count"])
		friends_count.append(tweet["friends_count"])
		listed_count.append(tweet["listed_count"])
		favorites_count.append(tweet["favourites_count"])
		statuses_count.append(tweet["statuses_count"])
		city_list.append(city)

		time = standardize_timezone(tweet["created_at"], city)
		created_at.append(time)
		text.append(tweet_text)
		negative.append(sentiment["neg"])
		neutral.append(sentiment["neu"])
		positive.append(sentiment["pos"])
		compound.append(sentiment["compound"])




	lists = zip(created_at, city_list, retweet_count, favorite_count, followers_count, friends_count, listed_count, favorites_count, 
		 statuses_count, text, negative, neutral, positive, compound)
	save_csv(folder, header, lists, city)





print("START calculating overall scores...")
calculate_overall_scores()
print("COMPLETE calculating overall scores...")

for city in cities:
	print("Start building {} data".format(city))
	tweet_list = []
	tweet_count = 0
	tweet_list_original = []

	city_dict = load_city_dict(city)
	city_dict = remove_duplicates(city_dict, city)
	tweets_csv(city_dict, city)