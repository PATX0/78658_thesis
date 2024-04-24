import pandas as pd
from datetime import datetime

def preprocessor_BERT(df):

    #drops any entry that is empty and is not a string
    #df = df.dropna(subset=['reviews'])
    df = df[df['reviews'].apply(lambda x: isinstance(x, str))]

    max_sequence_length = 512  # Maximum sequence length (limit for BERT)

    # Iterate through each row and drop rows with review text length exceeding the limit of BERT (512)
    for index, row in df.iterrows():
        if len(row['reviews']) > max_sequence_length:
            df.drop(index, inplace=True)

        
def normalize_sentiment(df, output_csv):
    """

    Normalizes the sentiment labels in the DataFrame to numerical scores and overwrites the original CSV file with the normalized data.
    """
    sentiment_map = {
        'very_bad': 1,
        'bad': 2,
        'neutral': 3,
        'good': 4,
        'very_good': 5
    }
    df['sentiment'] = df['sentiment'].map(sentiment_map)
    df.to_csv(output_csv, index=False)
    print(f"Normalized DataFrame saved to {output_csv}")
    


# Function to load CSV, modify, and save with the same name, adding rating to the Playstore  csvs
def process_and_save_csv(path):
    try:
        # Load the DataFrame
        df = pd.read_csv(path)
        
        # Add 'rating' column if it doesn't exist
        if 'rating' not in df.columns:
            df['rating'] = pd.NA  # Using pandas' NA for missing data handling

        # Convert 'timestamp' to datetime and format to '%Y-%m-%d'
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Save the DataFrame back to the same CSV
        df.to_csv(path, index=False)
        print(f"Processed and saved: {path}")

    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")


def main():
    #preprocessor_BERT(df)
    
    # List of exchanges, countries, and sources
    exchanges = ['Binance', 'Coinbase', 'Kucoin']
    countries = ['US', 'UA', 'NG', 'CN', 'BR']
    sources = ['AS']  # PlayStore (PS) and AppStore (AS)

    # Iterate over each file and process
    for exchange in exchanges:
        for country in countries:
            for source in sources:
                # Construct the file path
                filename = f"{source}_{exchange}_{country}_bert.csv"
                path = f"csvs/{exchange.lower()}/{filename}"
                # Process and save each file
                process_and_save_csv(path)


if __name__ == '__main__':
    main()