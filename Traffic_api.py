import requests
import smtplib 

# API key
api_key ='<key>'

# home address input
#home = input("Enter a home address\n") 
home="domlur" 
# work address input
#work = input("Enter a work address\n") 
work="sarjapur" 
# base url
url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&"

# get response
r = requests.get(url + "origins=" + home + "&destinations=" + work + "&key=" + api_key) 
 
# return time as text and as seconds
time = r.json()["rows"][0]["elements"][0]["duration"]["text"]       
seconds = r.json()["rows"][0]["elements"][0]["duration"]["value"]
distance = r.json()["rows"][0]["elements"][0]["distance"]["text"]

# print the travel time
print("\nThe total travel time from source  to dest is ", time)
print("The total travel time from source to dest is ", distance)
print("\n 17 miles per hours is fast. That is there is no traffic ")
print("NO")


#Output
# From Domlur to Sarjapur
#The total travel time from source  to dest is  1 hour 2 mins
#The total travel time from source to dest is  16.9 mi

#17 miles per hours is fast. That is there is no traffic
#NO
