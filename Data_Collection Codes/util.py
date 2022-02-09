import twitter
import json

auth_file = "login.json"

def open_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        json_file.close()
        return data
        

def auth():
    with open(auth_file) as file:
        credentials = json.load(file)
        return twitter.oauth.OAuth(credentials['OAUTH_TOKEN'], 
                                    credentials['OAUTH_TOKEN_SECRET'],
                                    credentials['CONSUMER_KEY'], 
                                    credentials['CONSUMER_SECRET'])
