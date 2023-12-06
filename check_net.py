import requests

def check_internet_connection():
    try:
        # Try to make a GET request to a well-known website
        response = requests.get("https://www.google.com", timeout=5)

        # Check if the response status code is successful (200 OK)
        if response.status_code == 200:
            print("Internet connection is available.")
        else:
            print("Internet connection is not available. Status code:", response.status_code)

    except requests.ConnectionError:
        print("Could not connect to the internet.")

# Call the function to check internet connectivity
check_internet_connection()
import datasets
datasets.load_dataset('ncbi_disease')
