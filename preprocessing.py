import pandas as pd
import openai
import os
import json
from tqdm import tqdm
import torch
import re
import time
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util
import numpy as np


def load_dataset():
    # Load the parquet file
    data_path = '/home/janulm/Documents/projects/datathon/data.parquet'
    df = pd.read_parquet(data_path)
    
    return df


def get_home_page_url(df):
    # Extract and save home URLs for all rows
    df['home_url'] = df['page_url'].apply(lambda x: x.strip().split('/')[2].replace('www.', '') if x.strip().startswith('http') else x.strip().split('/')[0].replace('www.', ''))
    
    # Drop the website_url column
    df = df.drop('website_url', axis=1)
    
    return df


def compute_embeddings(df, model, batch_size=32, start_index=0, end_index=1000, model_batch_size=128):
    """
    Compute embeddings for the text column in batches
    
    Args:
        df: DataFrame containing the data
        model: SentenceTransformer model
        batch_size: Number of texts to encode at once
        num_rows: Number of rows to process
    
    Returns:
        DataFrame with embeddings added
    """
    # Create a subset of the data
    if start_index is None:
        subset_df = df.copy()
    else:
        # make sure start_index is not greater than end_index
        if start_index > end_index:
            raise ValueError("start_index must be less than end_index")
        # make sure the end_index is not greater than the length of the dataframe
        if end_index > len(df):
            end_index = len(df)
        subset_df = df.iloc[start_index:end_index].copy()
    print("done here")
    # Extract texts for embedding
    texts = subset_df['text'].tolist()
    print("done here 2")
    # Measure time
    start_time = time.time()
    print("done here 3")
    # Compute embeddings in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=model_batch_size)
        all_embeddings.append(embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print("done here 4")
    print("embeddings.shape: ", embeddings.shape)
    print("embeddings.dtype: ", embeddings.dtype)

    

    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Time taken to compute embeddings: {timedelta(seconds=time_taken)}")

    # Add embeddings to the dataframe
    subset_df['embd_vector'] = embeddings.tolist()
    
    return subset_df


def save_embeddings(df, filename="embeddings_dataset.parquet"):
    """Save the dataframe with embeddings to disk"""
    df.to_parquet(filename)
    print(f"Dataset with embeddings saved to {filename}")
    


def main(start_index=0, end_index=100000):
    # Load dataset
    df = load_dataset()
    
    print("ORIGINAL DATASET. Starting with 700k rows")
    df = get_home_page_url(df)
    print("Done computing home page urls")

    # Print information about the dataset
    print(f"Total pages in dataset: {len(df)}")
    print("KEYS:", df.iloc[0].keys())

    for i in range(5):
        print("TIMESTAMP:", df.iloc[i]['timestamp'])
        print("HOME URL:", df.iloc[i]['home_url'])
        print("TEXT:", df.iloc[i]['text'][:10])
        print("PAGE URL:", df.iloc[i]['page_url'])

    # Initialize model and tokenizer ONCE for all processing
    print("\nLoading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = model.to(device)
    
    # Compute embeddings for the first 1000 rows
    embedded_df = compute_embeddings(df, model, batch_size=512, model_batch_size=128, start_index=start_index, end_index=end_index)
    
    # Print the shape of one embedding vector to verify
    print(f"Embedding vector shape: {np.array(embedded_df['embd_vector'].iloc[0]).shape}")
    
    # print the embedded_df
    print("embedded_df.head(): ", embedded_df.head())
    
    # Save the embeddings
    save_embeddings(embedded_df, f"embeddings_dataset_{start_index}_{end_index}.parquet")


if __name__ == "__main__":
    for i in range(8):
        main(start_index=i*100000, end_index=(i+1)*100000)
