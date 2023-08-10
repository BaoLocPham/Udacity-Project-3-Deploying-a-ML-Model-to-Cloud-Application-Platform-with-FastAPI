'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-10 09:29:20
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-10 09:29:43
 # @ Description:
    query API 
'''


import requests
from dotenv import find_dotenv, dotenv_values
ENV_CFG = dotenv_values(find_dotenv())
print(ENV_CFG)

data = {
    'age': 43,
    'hours_per_week': 50,
    'workclass': 'Federal-gov',
    'education': 'Doctorate',
    'marital_status': 'Never-married',
    'occupation': 'Prof-specialty',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Female',
    'native_country': 'United-States'
}
r = requests.post(ENV_CFG[
    'API_URL'
], json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
