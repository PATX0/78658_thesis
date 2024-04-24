import pandas as pd


def preprocessor_BERT(df):
    #drops any entry that is empty and is not a string
    df = df.dropna(subset=['reviews'])
    df = df[df['reviews'].apply(lambda x: isinstance(x, str))]

    max_sequence_length = 512  # Maximum sequence length (limit for BERT)

    # Iterate through each row and drop rows with review text length exceeding the limit of BERT (512)
    for index, row in df.iterrows():
        if len(row['reviews']) > max_sequence_length:
            df.drop(index, inplace=True)

        
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
        #'csvs/appstore/binance/AS_Binance_CN_bert.csv',
        'csvs/appstore/binance/AS_Binance_NG_bert.csv',
        #'csvs/appstore/coinbase/AS_Coinbase_CN_bert.csv',
        'csvs/appstore/coinbase/AS_Coinbase_NG_bert.csv',
        #'csvs/appstore/kucoin/AS_Kucoin_CN_bert.csv',
        'csvs/appstore/kucoin/AS_Kucoin_NG_bert.csv'
    ]

    # Load each CSV and normalize sentiment
    for path in paths:
        df = pd.read_csv(path)
        #preprocessor_BERT(df)
        normalize_sentiment(df, path)  # Pass both the DataFrame and the path for saving


if __name__ == '__main__':
    main()