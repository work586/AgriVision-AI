import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY")

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 5,
    "filters[state]": "Karnataka",
}

print("Testing data.gov.in mandi API...")
print("State filter: Karnataka")
print("API key found:", bool(API_KEY))

try:
    response = requests.get(
        URL,
        params=params,
        timeout=(15, 120)
    )

    print("HTTP Status:", response.status_code)

    if response.status_code == 200:
        data = response.json()

        print("API request successful.")
        print("Total records:", data.get("total", "Unknown"))
        print("Returned records:", len(data.get("records", [])))

        for record in data.get("records", []):
            print(record)

    else:
        print("API request failed.")
        print("Response preview:")
        print(response.text[:1000])

except requests.exceptions.Timeout:
    print("The API request timed out.")

except requests.exceptions.RequestException as error:
    print("Network error:", error)

except Exception as error:
    print("Unexpected error:", error)