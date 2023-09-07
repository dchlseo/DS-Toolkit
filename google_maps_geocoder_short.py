# Simplified version: add API key, and address string.
# This function will return a list of geo locations (latitude, longitude).

import googlemaps
from tqdm import tqdm_notebook

def getGeo(key, address):
    # Initialize Google Maps client
    gmaps_key = key
    gmaps = googlemaps.Client(key=gmaps_key)

    gmaps_output = gmaps.geocode(address)
    
    if gmaps_output:
        location_output = gmaps_output[0].get('geometry')
        
        lat = location_output['location']['lat']
        lng = location_output['location']['lng']
    
    else:
        # Handle the case when no geocode results are found
        print("Error: No geocode results for address:", address)
        
        lat, lng = None, None
        

    return lat, lng
