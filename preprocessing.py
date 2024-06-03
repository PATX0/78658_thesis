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
    aux0 = []
    aux = []
    aux1 = []
    # List of exchanges, countries, and platforms
    exchanges = ['Binance', 'Coinbase', 'Kucoin']
    countries = ['US', 'UA', 'NG', 'CN', 'BR']
    platforms = ['AS','PS']  # PlayStore (PS) and AppStore (AS)

    # THIS PART WAS USED TO standardize the timestamps and add the 'rating' columns so we can concat the csvs
    for exchange in exchanges:
        for country in countries:
            for platform in platforms:
                path = f"csvs/{exchange}/{platform}_{exchange}_{country}_bert.csv"
                # Process and save each file
                process_and_save_csv(path)

    # THIS PART IS TO COMBINE THE REVIEWS FROME EACH PLATFORM
    for platform in platforms:
        filename = f"{platform}_kucoin_CN_bert.csv"
        path = f"csvs/kucoin/{filename}"
        df = pd.read_csv(path)
        aux0.append(df)
    combined_df = pd.concat(aux0, ignore_index=True)
    combined_df.to_csv("csvs/kucoin/kucoinCN.csv", index=False)

    #THIS PART IS TO COMBINE THE REVIEWS FROM EACH EXCHANGE, repeat execution for each
    for country in countries:
        path = f"csvs/kucoin/kucoin{country}.csv"
        db = pd.read_csv(path)
        aux.append(db)
    kucoin_total = pd.concat(aux, ignore_index=True)
    kucoin_total.to_csv('csvs/kucoin/kucoinTotal.csv', index=False)

    #THIS PART IS TO COMBINE THE REVIEWS FROM EACH COUNTRY, repeat execution for each
    for platform in platforms:
        for exchange in exchanges:
            path = f"csvs/NG/{platform}_{exchange}_NG_bert.csv"
            df1 = pd.read_csv(path)
            aux1.append(df1)
    country_total = pd.concat(aux1, ignore_index=True)
    country_total.to_csv('csvs/NG/NGtotal.csv', index=False)
    

    
if __name__ == '__main__':
    main()