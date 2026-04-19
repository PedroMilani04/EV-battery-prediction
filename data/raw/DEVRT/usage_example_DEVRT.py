import pandas as pd
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
import os
import glob


def load_and_describe(file_path):
    try:
        # Loading the dataset
        # 'na_values' ensures empty strings are treated as NaN
        df = pd.read_csv(file_path, skip_blank_lines=True)
        
        print("--- Dataset Overview ---")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        print("--- Column Types and Non-Null Counts ---")
        print(df.info())
        
        print("\n--- Summary Statistics (Numerical) ---")
        print(df.describe())
        
        print("\n--- Missing Values Per Column ---")
        print(df.isnull().sum())
        
        print("\n--- First 5 Rows ---")
        print(df.head())
        
        return df

    except Exception as e:
        print(f"An error occurred: {e}")



def plot_tracks_static(df, lat_col='latitude', lon_col='longitude', track_id_col=None):
    """Build a static plot of the track."""
    plt.figure(figsize=(10, 8))
    
    if track_id_col and track_id_col in df.columns:
        for tid, group in df.groupby(track_id_col):
            plt.plot(group[lon_col], group[lat_col], alpha=0.6, label=f'ID {tid}')
    else:
        plt.scatter(df[lon_col], df[lat_col], s=1, alpha=0.5, color='blue')

    plt.title("Track visualisation")
    plt.xlabel("Longitud")
    plt.ylabel("Latitude")
    plt.gca().set_aspect('equal', adjustable='datalim') # Mantiene la proporción geográfica
    plt.grid(True)
    plt.show()

def plot_tracks_interactive(df, lat_col='latitude', lon_col='longitude', data_col='cumul_dist', output_file='interactive_map.html'):
    """Build an HTML file with a map showing the track."""
    # Centering the map
    avg_lat = df[lat_col].mean()
    avg_lon = df[lon_col].mean()
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles='CartoDB Positron')


    # Determine the data range
    minim = df[data_col].min()
    maxim = df[data_col].max()
    
    # If float data type:
    if df[data_col].dtype == 'float64' or df[data_col].dtype == 'int64':
        # Define a linear gradient colormap
        colormap = cm.linear.YlOrRd_09.scale(minim, maxim) # Adjust scale as needed
        colormap.caption = 'Gradient Legend - [' + data_col + '] ' # Add a caption
        colormap.add_to(m) # Add colormap to the map
        print('Selected data column is float64 or int64')
        # Add points to the map 
        for index, row in df.dropna(subset=[lat_col, lon_col, data_col]).iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=2,
                #color='red',
                color=colormap(row[data_col]),
                fill=True
            ).add_to(m)        
        m.save(output_file)
        print(f"\nInteractive map saved as: {output_file}")
    else:
        print('Selected data column IS NOT Float')
        # Add points to the map 
        for index, row in df.dropna(subset=[lat_col, lon_col, data_col]).iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=2,
                color='red',
                fill=True
            ).add_to(m)
        m.save(output_file)
        print(f"\nInteractive map saved as: {output_file}")
    
def plot_elevation_profile(df, output_file='elevation_profile.png'):
    plt.rcParams["figure.figsize"] = [10.50, 6.00]
    plt.rcParams["figure.autolayout"] = True
    fig,ax = plt.subplots()
    ax.plot(df['altitude'], label="Elevation Profile", color='orange')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.25), loc="lower left")
    ax.set_xlabel("Step")
    ax.set_ylabel("Elevation (m)")
    plt.savefig(output_file)
    print(f"\nElevation profile saved as: {output_file}")
    
    
def concatenate_csv_files(folder_path, output_filename):
    # Create a pattern to match all csv files in the directory
    file_pattern = os.path.join(folder_path, "*.csv")
    
    # List all files matching the pattern
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No CSV files found in the specified directory.")
        return

    # Use a list comprehension to read all CSVs into DataFrames
    # This is faster and more memory-efficient than appending in a loop
    df_list = [pd.read_csv(file) for file in all_files]
    
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Save the result to a new CSV
    combined_df.to_csv(output_filename, index=False)
    print(f"Successfully combined {len(all_files)} files into {output_filename}")



# --- Configuration ---
# Use '.' for the current folder or provide a full path like 'C:/Users/Documents/Data'
target_folder = '.' 
output_file = 'combined_data.csv'    
input_file = '20230420_DACIA_ZARAUTZ_DONOSTIA_040.csv'

#IF NEEDED CONCATENATE ALL TRACK FILES IN SPECIFIED FOLDER INTO ONE CSV:
concatenate_csv_files('DACIA SPRING', target_folder+'/'+output_file)
    
# Example usage:
df = load_and_describe(input_file) # FOR SPECIFIC TRACK FILE
#df = load_and_describe(target_folder+'/'+output_file) # FOR COMBINED DATA FROM ALL TRACK FILES 

if df is not None:
    #UNCOMMENT TO PLOT TRACK(s) IN CHART: 
    #plot_tracks_static(df, lat_col='latitude', lon_col='longitude')
    #UNCOMMENT TO PLOT TRACK(s) IN INTERACTIVE MAP (select the desired data column in data_col):
    plot_tracks_interactive(df, data_col='altitude')
    #UNCOMMENT TO PLOT ELEVATION PROFILE OF TRACK(s): 
    plot_elevation_profile(df)
    