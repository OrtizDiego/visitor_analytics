from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import requests

# Haversine formula to calculate distance between two lat-lon points
def haversine(lon1, lat1, lon2, lat2):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lon1_radians = np.radians(lon1)
    lat1_radians = np.radians(lat1)
    lon2_radians = np.radians(lon2)
    lat2_radians = np.radians(lat2)
    
    # Difference in coordinates
    dlon = lon2_radians - lon1_radians
    dlat = lat2_radians - lat1_radians
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_radians) * np.cos(lat2_radians) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

# Function to filter the dataset
def filter_by_radius(dataframe, event_location, radius):
    """
    dataframe: pandas DataFrame containing 'latitude' and 'longitude' columns
    event_location: tuple of (latitude, longitude) for the event location
    radius: radius in kilometers
    """
    # Apply the haversine function to each row in the dataframe
    distances = dataframe.apply(lambda row: haversine(event_location[1], event_location[0],
                                                      row['central_longitude'], row['central_latitude']), axis=1)
    
    # Filter the dataframe for rows where the distance is less than or equal to the radius
    return dataframe[distances <= radius]

# Read in our dataset
mastercard = pd.read_csv('MasterCardData\GeoInsights_Synthetic_Output.csv')
geo_data = pd.read_csv('MasterCardData\GeoInsights_Hackathon_Quads_GeoInfo.csv', delimiter='|')
merged_df = pd.merge(mastercard, geo_data, on='quad_id', how='inner')

app = Flask(__name__)
CORS(app)

# Example DataFrame
data = {
    'latitude': [34.0522, 36.1699, 37.7749],
    'longitude': [-118.2437, -115.1398, -122.4194],
    'value': [10, 20, 30]  # This could be time or any other numerical value that the slider controls
}
df = pd.DataFrame(data)

@app.route('/data', methods=['GET'])
def get_data():
    # Let's say you want to filter the DataFrame based on a value provided by a query parameter
    filter_value = requests.args.get('value', default=None, type=int)

    # Filter the DataFrame if a filter value is provided
    if filter_value:
        filtered_df = df[df['value'] <= filter_value]
    else:
        filtered_df = df

    # Convert the filtered DataFrame to JSON
    data_json = filtered_df.to_json(orient='records')
    return jsonify(data_json)

@app.route('/heatmap-data/<slider_value>')
def heatmap_data(slider_value):
    # Fetch or generate your heatmap data based on the slider value
    data = get_heatmap_data(slider_value)
    return jsonify(data)

def get_heatmap_data(slider_value):
    # This function should interact with your data source to retrieve and filter data
    return [
        {"lat": 35.6895, "lng": 139.6917, "intensity": float(slider_value)},
        # Add more points...
    ]

if __name__ == '__main__':
    app.run(debug=True)
