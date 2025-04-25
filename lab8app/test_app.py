import requests

# 🔁 Match this with what your model was trained on!
payload = {
    "Gender": "Male",
    "Age": 25,
    "Height": 1.75,
    "Weight": 85.0,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 3.0,
    "NCP": 3.0,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2.0,
    "SCC": "no",
    "FAF": 1.0,
    "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
}

# Send request
res = requests.post("http://127.0.0.1:8000/predict", json=payload)

# Print result
print("Status Code:", res.status_code)
print("Raw Response:", res.text)