#!/usr/bin/env python

import requests
import os

# Read the master key from your file
key_path = os.path.expanduser("~/gemma/server_master_key.txt")
with open(key_path, "r") as f:
    master_key = f.read().strip()

# Add the required Authorization header
headers = {
    "Authorization": f"Bearer {master_key}",
    "Content-Type": "application/json"
}

# Now perform the request
response = requests.get("http://localhost:4000/v1/models", headers=headers)
print(f"Status: {response.status_code}")
print(response.json())