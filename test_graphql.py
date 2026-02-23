import requests

graphql_url = "https://leetcode.com/graphql/"
headers = {"Content-Type": "application/json"}

payload = {
    "query": """
    query leetcodeProfileInfo($username: String!) {
      matchedUser(username: $username) {
        profile {
          ranking
          reputation
        }
        submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
          }
        }
      }
    }
    """,
    "variables": {"username": "lee215"},
    "operationName": "leetcodeProfileInfo"
}

res = requests.post(graphql_url, json=payload, headers=headers).json()
import pprint
pprint.pprint(res)
