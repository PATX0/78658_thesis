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


def convert_timestamp(entry):
    # Attempt to convert entry to numeric (float), errors='coerce' will convert failures to NaN
    numeric_entry = pd.to_numeric(entry, errors='coerce')
    print(numeric_entry)
        # Check if the conversion was successful (not NaN)
    if pd.notna(numeric_entry):
        return pd.to_datetime(entry, unit='ms').strftime('%Y-%m-%d')
    # Convert string to datetime object
    return entry
        
def normalize_sentiment(df, output_csv):
    """
    Normalizes the sentiment labels in the DataFrame to numerical scores and overwrites the original CSV file with the normalized data.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the sentiment labels.
    output_csv (str): Path to the CSV file where the normalized DataFrame will be saved.
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
    

def main():
    # Define file paths
    paths = [
        #'csvs/playstore/playstore_binance_reviews_UA.csv',
        #'csvs/playstore/playstore_binance_reviews_UA.csv',
        #'csvs/playstore/coinbase/AS_Coinbase_CN_bert.csv',
        'csvs/playstore/playstore_coinbase_reviews_UA.csv',
        #'csvs/playstore/kucoin/AS_Kucoin_CN_bert.csv',
        'csvs/playstore/playstore_kucoin_reviews_UA.csv'
    ]

    # Load each CSV and normalize sentiment
    for path in paths:
        df = pd.read_csv(path)
        # Apply conversion function to the timestamp column
        df['timestamp'] = df['timestamp'].apply(convert_timestamp)
        df.to_csv(path, index=False)
        #preprocessor_BERT(df)
        #normalize_sentiment(df, path)  # Pass both the DataFrame and the path for saving


if __name__ == '__main__':
    main()