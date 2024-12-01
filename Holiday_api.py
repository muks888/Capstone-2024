import requests
from datetime import datetime
from typing import List



GOOGLE_CALENDAR_API_KEY='<key>'
if not GOOGLE_CALENDAR_API_KEY:
    raise Exception("The GOOGLE_CALENDAR_API_KEY environment variable is not set.")

def fetch_holidays(country_code: str, year: str):
    url = f"https://www.googleapis.com/calendar/v3/calendars/{country_code}%23holiday%40group.v.calendar.google.com/events"
    params = {
        'key': GOOGLE_CALENDAR_API_KEY,
        'timeMin': f'{year}-01-01T00:00:00Z',
        'timeMax': f'{year}-12-31T23:59:59Z',
        'singleEvents': True,
        'orderBy': 'startTime'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get('items', [])
    

country_code='en.indian'
year=2024
holidays = fetch_holidays(country_code, year)

from datetime import date

# Returns the current local date
today = date.today()
print("Today date is: ", today)
today='2024-01-01' # Testing
print("Today date is: ", today)



for i in holidays:
    if i['start']['date'] == today:
        print("Today is ",i['summary'])
        break

#Output
#Today date is:  2024-12-01
#Today date is:  2024-01-01 (For testing date was set to 1st Jan)
#Today is  New Year's Day
