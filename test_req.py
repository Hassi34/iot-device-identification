import requests
from model_serving.src.configs.pydantic_configs import requested_data_example

# URL of your authentication endpoint
url = "http://iot-devices.eastus.cloudapp.azure.com/auth/token"

# Your credentials and other required parameters
data = {
    "grant_type": "password",
    "username": "a@example.com",
    "password": "123",
}

# Make the POST request
response = requests.post(url, data=data)

# If the request is successful, the status code will be 200
if response.status_code == 200:
    # Extract the access token from the response
    access_token = response.json().get("access_token")
    print(access_token)
else:
    print(f"Failed to authenticate: {response.text}")

# URL of your main API endpoint
url = "http://iot-devices.eastus.cloudapp.azure.com/predict/iot-devices"

# Your request data
data = {
    "request_data": requested_data_example
}

# Headers with the access token
headers = {
    "Authorization": f"Bearer {access_token}"
}

# Make the POST request
response = requests.post(url, json=data, headers=headers)

# If the request is successful, the status code will be 200
if response.status_code == 200:
    # Process the response
    print(response.json())
else:
    print(f"Failed to make request: {response.text}")