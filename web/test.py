import requests

url = "http://127.0.0.1:8081/api/chatbot"  # Replace with the actual URL of your server
data = {
            "messages": [
                {"role": "user", "content": "What is 300 times 1000?"},
                {"role": "assistant", "content": "300 multiplied by 1000 is 300000."},
                {"role": "user", "content": "Add 3 to the answer."},
            ],
            "lang": "en",
        }
response = requests.post(url, json=data)  # Send a POST request with JSON data

if response.status_code == 200:
    print("Request successful!")
    # Access the response content if needed (e.g., for testing the response)
    response_data = response.json()
    print(response_data)
else:
    print("Request failed with status code:", response.status_code)