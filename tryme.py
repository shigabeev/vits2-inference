import requests

# The URL of your FastAPI synthesize_async endpoint
url = "http://localhost:8000/synthesize/"

# The text you want to synthesize
data = {"text": "Пример текста для синтеза речи. Непростой пример. Дед такое не умеет."}

# Make a POST request and stream the response
response = requests.post(url, json=data, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Open a file for writing the binary audio data
    with open("output.wav", "wb") as f:
        print("chunk processed")
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Audio was saved as output.wav")
else:
    print(f"Request failed with status code {response.status_code}")
