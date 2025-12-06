import requests
import json
import time
import os

# Headers
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjkzM2IxYWM1MGEzNmFiYzZiZGI2Yzc2IiwiYXBwbGljYXRpb25faWQiOiI2NGI4NDFmYmViZDQzNzAwMjhhNmNmMjYiLCJpYXQiOjE3NjQ5OTU5OTd9.ihIAoTqQxgj0ky6VdSrESyZGbiaC1181qxVdhXrI8fI",
    "Cookie": "knack.com.connect.sid=s%3A_v1NxCdAnWmv0EbhEoio3Yyp9T4tffF8.sktmav18OWfs9%2BjhSWUsYy8zJenxq8J2pN3pPsdhBu0",
    "x-knack-application-id": "64b841fbebd4370028a6cf26",
    "Referer": "https://www.carbondi.com/"
}

def fetch_countries():
    url = "https://eu-central-1-renderer-read.knack.com/v1/scenes/scene_102/views/view_363/records"
    params = {
        "format": "both",
        "page": 1,
        "rows_per_page": 1000,
        "sort_field": "field_44",
        "sort_order": "asc"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("records", [])

def fetch_emission_factor(country_id):
    url = "https://eu-central-1-renderer-read.knack.com/v1/scenes/scene_273/views/view_926/records"
    params = {
        "format": "both",
        "page": 1,
        "rows_per_page": 1,
        "view-country-details_id": country_id,
        "sort_field": "field_996", # Year
        "sort_order": "desc"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            print("Rate limit hit, sleeping...")
            time.sleep(5)
            return fetch_emission_factor(country_id)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
        if records:
            return records[0]
        return None
    except Exception as e:
        print(f"Error fetching for {country_id}: {e}")
        return None

def main():
    print("Fetching countries...")
    countries = fetch_countries()
    print(f"Found {len(countries)} countries.")

    results = []
    
    for i, country in enumerate(countries):
        country_name = country.get("field_44_raw", "Unknown")
        country_id = country.get("id")
        
        # Extract ISO code
        iso_code = "Unknown"
        raw_code = country.get("field_473_raw")
        if raw_code and isinstance(raw_code, list) and len(raw_code) > 0:
            iso_code = raw_code[0].get("identifier", "Unknown")
            
        print(f"[{i+1}/{len(countries)}] Fetching data for {country_name} ({iso_code})...")
        
        factor_data = fetch_emission_factor(country_id)
        
        emission_factor = 0.0
        year = None
        
        if factor_data:
            emission_factor = factor_data.get("field_1020_raw", 0.0)
            year = factor_data.get("field_996_raw")
            
        results.append({
            "country_name": country_name,
            "country_code": iso_code,
            "emission_factor": emission_factor,
            "year": year
        })
        
        # Sleep to avoid rate limits
        time.sleep(0.5)

    with open("emissions.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Done! Saved to emissions.json")

if __name__ == "__main__":
    main()
