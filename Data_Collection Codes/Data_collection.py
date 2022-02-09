import json
import twitter
import os
import sys
import util
import ssl
import shutil
import threading

from datetime import datetime
from email.utils import parsedate_tz, mktime_tz



# if os.path.exists("city_data"):
#     shutil.rmtree("city_data")

if not os.path.exists("city_data"):
    os.makedirs("city_data")

def save_and_close(city_file_name, city_data_list):
    with open(file_path.format(city_file_name),"w") as city_file:
        json.dump(city_data_list, city_file)
        city_file.close()

# Refer to https://stackoverflow.com/questions/29147941/convert-time-zone-format-in-python-from-twitter-api

def to_local_time(tweet_time_string):
    timestamp = mktime_tz(parsedate_tz(tweet_time_string))
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def city_collection():
    #threading.Timer(5.0, city_collection).start()
    for city in location_dict["cities"]:
        print(city)

        geocode=location_dict["cities"][city]["latlongmi"]
        city_file_name = location_dict["cities"][city]["name"]

        if not os.path.exists(file_path.format(city_file_name)):
            with open(file_path.format(city_file_name), "w+") as city_file:
                city_file.write("[]")
                city_file.close()

        city_data_list = util.open_json(file_path.format(city_file_name))



        for i in range(0,tweet_count,increment):

            try:
                tweets = twitter_api.search.tweets(geocode=geocode, count=increment, lang='en', result_type='recent')["statuses"]
                twitter_api.search

                for tweet in tweets:

                    tweet_info = {}
                    tweet_info["created_at"] = to_local_time(tweet["created_at"])
                    tweet_info["id_str"] = tweet["id_str"]
                    tweet_info["text"] = tweet["text"]
                    tweet_info["place"] = tweet["place"]
                    #tweet_info["coordinates"] = tweet["coordinates"]
                    tweet_info["retweet_count"] = tweet["retweet_count"]
                    tweet_info["favorite_count"] = tweet["favorite_count"]
                    tweet_info["user_location"] = tweet["user"]["location"]
                    tweet_info["followers_count"] = tweet["user"]["followers_count"]
                    tweet_info["friends_count"] = tweet["user"]["friends_count"]
                    tweet_info["listed_count"] = tweet["user"]["listed_count"]
                    tweet_info["favourites_count"] = tweet["user"]["favourites_count"]
                    tweet_info["statuses_count"] = tweet["user"]["statuses_count"]


                    city_data_list.append(tweet_info)
            except:
                print("Reached tweet limit, ending script")
                save_and_close(city_file_name, city_data_list)
                sys.exit()

            if i%100 == 0:
                print("Gathered {} tweets for the {} location".format(i+increment, city))

        save_and_close(city_file_name, city_data_list)



twitter_api = twitter.Twitter(auth=util.auth())
location_dict = util.open_json("locations.json")

tweet_count = 100
increment = 100
file_path = "city_data/{}.json"

city_collection()



