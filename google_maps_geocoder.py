# This function will return a list of geo locations (latitude, longitude) for each address row in a dataframe.

import numpy as np
import googlemaps
from tqdm import tqdm_notebook

def getGeo(key, df, colName):
    # Initialize Google Maps client
    gmaps_key = key
    gmaps = googlemaps.Client(key=gmaps_key)

    # Initialize output lists
    lat = []
    lng = []

    for idx, row in tqdm_notebook(df.iterrows()):
        if row[colName].strip():  # Check if the address is non-empty
            target_name = row[colName]

            gmaps_output = gmaps.geocode(target_name)

            if gmaps_output:
                location_output = gmaps_output[0].get('geometry')
                lat.append(location_output['location']['lat'])
                lng.append(location_output['location']['lng'])
            else:
                # Handle the case when no geocode results are found
                print("Error: No geocode results for address:", target_name)
                lat.append(np.nan)
                lng.append(np.nan)

        else:
            # Handle the case when the address is empty
            print("Error: empty address.")
            lat.append(np.nan)
            lng.append(np.nan)

    # Create and return a DataFrame with latitude and longitude
    result_df = df.copy()
    result_df['Latitude'] = lat
    result_df['Longitude'] = lng
    return result_df