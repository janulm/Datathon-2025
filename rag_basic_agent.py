import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# Import OpenAI if not already imported
import openai
import os
import asyncio
import time


"""
Feature List: 

1. UI, to have a chat window, 
2. now if asking counterquestions, dont do RAG again, but simply answer. Currently its one time only...



"""

def load_list_parquets(file_paths=["/home/janulm/Documents/projects/datathon/embeddings_dataset_0_100000.parquet","/home/janulm/Documents/projects/datathon/embeddings_dataset_100000_200000.parquet"]):
    datasets = [pd.read_parquet(file_path) for file_path in file_paths]
    
    concat_df = pd.concat(datasets)
    print("Concat df shape: ", concat_df.shape)
    print("Concat df columns: ", concat_df.columns.tolist())
    
    return concat_df

def load_embeddings_dataset(file_path="embeddings_dataset_20000.parquet"):
    # Load the embeddings dataset
    print(f"Loading embeddings from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Show the head of the dataframe
    print("\nHead of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isna().sum())
    
   

    # Analyze embedding vectors
    assert 'embd_vector' in df.columns, "embd_vector column not found in dataframe"
    # Get the shape of the first embedding vector
    sample_embd = np.array(df['embd_vector'].iloc[0], dtype=np.float32)   
    print(f"\nEmbedding vector shape: {sample_embd.shape}")
    print(f"Embedding vector dtype: {sample_embd.dtype}")

    return df

def async_get_completion(client, message, max_tokens=400):
    """Async function to get completion from OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        max_tokens=max_tokens,
    )
    return response

def prompt_expansion(prompt, concept_dict, expansion_prompt, client, num_tokens_prompt_expansion):
    
    # go with the openai api to expand the prompt and the concept_specific prompts for each concept. 
    # if the return for a concept contains "REMOVE", then remove the concept 
    # we want to return a list of concept_prompt_descriptions

    print("Expanding prompt input prompt: ", prompt, "\n\n")

    tasks = []
    for concept, concept_prompt in concept_dict.items():
        task = expansion_prompt.replace("INPUT_PROMPT", prompt).replace("CONCEPT", concept_prompt)
        tasks.append(task)
    
    # use the openai api to expand the prompt and the concept_specific prompts for each concept. 
    # if the return for a concept contains "REMOVE", then remove the concept 
    # we want to return a list of concept_prompt_descriptions
    list_of_prompts = []

    #print("Number of tasks", len(tasks),"Tasks: ", tasks, "\n\n")

    # Process tasks sequentially since we're using synchronous client
    for task in tasks:
        messages = [{"role": "user", "content": task}]
        response = async_get_completion(client, messages, num_tokens_prompt_expansion)
        
        message_content = response.choices[0].message.content
        # check if the message_content contains "REMOVE"
        #print("Message content: ", message_content, "\n\n")
        if "REMOVE" not in message_content:
            list_of_prompts.append(message_content)
        else: 
            #print("REMOVE found in message_content: ", message_content, "\n\n")
            pass
    
    return list_of_prompts


def get_home_url_subpage_link_dict(cos_scores,k,min_good_subpage_links,above_min_threshold,df):

    # Get the top 3 similarity scores from the entire matrix
    # Reshape the matrix to a 1D tensor
    flat_scores = cos_scores.flatten()
    
    # Get the top k values from the flattened tensor
    top_k_values, top_k_flat_indices = torch.topk(flat_scores, k=k)

    # now take at least min_good_subpage_links and at most max_good_subpage_links but only if the score is above the threshold, but minimum min_good_subpage_links
    #print("Top k values: ", top_k_values)
    #print("Top k flat indices: ", top_k_flat_indices)


    # Convert flat indices back to 2D indices (prompt_idx, document_idx)
    prompt_indices = top_k_flat_indices // cos_scores.shape[1]
    document_indices = top_k_flat_indices % cos_scores.shape[1]
    
    
    
    # now for each document_index, get the home_url and the subpage_link and group them by home_url and count the number of distinct subpage_links
    home_url_subpage_link_dict = {}
    
    set_of_home_urls = set()

    for i in range(k):
        #print("Document index: ", document_indices[i])
        subpage_idx = int(document_indices[i])
        #print("Subpage index: ", subpage_idx)
        home_url = df.iloc[subpage_idx]['home_url']
        subpage_link = df.iloc[subpage_idx]['page_url']
        set_of_home_urls.add(home_url)
        
    #print("Home URL subpage link dict: ", home_url_subpage_link_dict)

    for home_url in set_of_home_urls:
        home_url_subpage_link_dict[home_url] = []
        # for each home_url, get all the subpage_links
        all_subpage_links = df[df['home_url'] == home_url]['page_url'].tolist()
        home_url_subpage_link_dict[home_url].extend(all_subpage_links)
    
    print("Home URL subpage link dict: ", home_url_subpage_link_dict)
    return home_url_subpage_link_dict


def expand_sub_urls(home_url_subpage_link_dict,df):
    print("Expanding sub_urls")
    # now expand the sub_urls with the main home_url if its not in the sub_urls already
    for home_url, subpage_link_list in home_url_subpage_link_dict.items():
        # how do we find the subpage_link that is the most similar to the home_url?
        # use all rows in the df with the same home_url and find the subpage_link that is the shortest
        
        list_of_rows_with_same_home_url = df[df['home_url'] == home_url]
        # find the index where the string of the subpage link is the shortest
        shortest_subpage_link_index = list_of_rows_with_same_home_url['page_url'].apply(len)
        # get the subpage_link
        #print("We think this is the shortest subpage link index: ", shortest_subpage_link_index)
        idx_min = shortest_subpage_link_index.idxmin()
        #print("We think this is the shortest subpage link: ", idx_min)
        shortest_subpage_link = df.iloc[idx_min]['page_url']
        # add the shortest subpage_link to the subpage_link_list if not already there
        
        #print("Searching for home_url: ", home_url, " and found this subpage_link: ", shortest_subpage_link)
        if not any(shortest_subpage_link == link for link in subpage_link_list):
            subpage_link_list.append(shortest_subpage_link)

    print("Expanded sub_urls: ", home_url_subpage_link_dict)
    return home_url_subpage_link_dict


def get_intermediate_answers(input_prompt, intermediate_answer_prompt, home_url_subpage_link_dict, df, num_tokens_intermediate_answer, client=None):
    print("Getting intermediate answers")
    # Get intermediate answers for each home URL by processing its subpage URLs
    intermediate_answers = {}
    
    # Prepare tasks for parallel processing
    tasks = []
    home_urls = []
    
    for home_url, subpage_links in home_url_subpage_link_dict.items():
        # Retrieve the text content for each subpage link
        subpage_texts = []
        for subpage_link in subpage_links:
            # Find the row in dataframe matching the subpage link
            matching_rows = df[df['page_url'] == subpage_link]
            if not matching_rows.empty:
                text = matching_rows.iloc[0]['text']
                subpage_texts.append(f"URL: {subpage_link}\n{text}")
        
        # Concatenate all texts for this home URL
        all_subpage_text = "\n\n---\n\n".join(subpage_texts)
        
        # Prepare the prompt by replacing placeholders
        prompt = intermediate_answer_prompt.replace("INPUT_PROMPT", input_prompt).replace("SUBPAGE_LINK_LIST_TEXTS", all_subpage_text)
        
        # Add to tasks list
        tasks.append(prompt)
        home_urls.append(home_url)
    
    # Process tasks sequentially (will be run in parallel by OpenAI client)
    for i, task in enumerate(tasks):
        home_url = home_urls[i]
        messages = [{"role": "user", "content": task}]
        
        try:
            response = async_get_completion(client, messages, max_tokens= num_tokens_intermediate_answer)
            answer = response.choices[0].message.content
            intermediate_answers[home_url] = answer
            print(f"Processed intermediate answer for {home_url}")
        except Exception as e:
            print(f"Error processing {home_url}: {str(e)}")
            intermediate_answers[home_url] = f"Error processing this URL: {str(e)}"
    
    return intermediate_answers


def combine_intermediate_answers(context_text, input_prompt, final_answer_prompt, num_tokens_final_answer, client):
    print("Compute final answer")
    
    
    # Prepare the prompt by replacing placeholders
    prompt = final_answer_prompt.replace("INPUT_PROMPT", input_prompt).replace("CONTEXTUAL_INFORMATION", context_text)
    
    # Make OpenAI API call
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=num_tokens_final_answer,  # Allow for a longer final answer
        )
        final_answer = response.choices[0].message.content
        print("Generated final answer")
    except Exception as e:
        print(f"Error generating final answer: {str(e)}")
        final_answer = f"Error generating final answer: {str(e)}"
    
    return final_answer


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #df = load_embeddings_dataset("embeddings_dataset_20000.parquet")
    list_of_paths = [f"/home/janulm/Documents/projects/datathon/embeddings_dataset_{i*100000}_{(i+1)*100000}.parquet" for i in range(8)]
    df = load_list_parquets(list_of_paths)

    num_tokens_final_answer = 10000


    # TODO MAKE THESE PARAMS WORKING: 
    max_good_subpage_links = 1
    
    
      # Initialize OpenAI client
    client = openai.OpenAI(api_key="haha")
    

    # get the full matrix torch tensor of the embeddings
    embeddings_matrix = torch.tensor(np.array(df['embd_vector'].tolist()), dtype=torch.float32)
    embeddings_matrix = embeddings_matrix.to(device)

    print(f"Embeddings matrix shape: {embeddings_matrix.shape}")

    
    # load the model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = model.to(device)


    # Example: Find documents related to banks
    input_prompt = "How can a veteran invest money in a bank?"
    input_prompt = "Please tell me about AI companies in US"

    # TODO WAIT FOR USER INPUT HERE
    input_prompt = input("Please enter your prompt: ")

    

    
    
    # Just get the embedding of the input prompt
    embed_prompt = model.encode([input_prompt], convert_to_tensor=True)
    # make into a tensor
    embed_prompt = embed_prompt.clone().detach()
    # print the shape of the embed_prompts
    print(f"Embed prompts shape: {embed_prompt.shape}")
    
    # move to the device
    embed_prompt = embed_prompt.to(device)


    # compute the cosine similarity between the embeded_prompt and the embeddings_matrix
    cos_scores = util.pytorch_cos_sim(embed_prompt, embeddings_matrix)
    # print the shape of the cos_scores
    print(f"Cos scores shape: {cos_scores.shape}")


    # get the top 10 values of the cos_scores:
    # group them by their home_url, and then print out the number of distinct home_urls,
    # and the number of distinct subpages for each home_url.


    home_url_subpage_link_dict = get_home_url_subpage_link_dict(cos_scores,max_good_subpage_links,None,None,df)


    # now we have a dict of home_url and subpage_link_list

    # then we want to combine all the intermediate answers into a final answer
    
    final_answer_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
    Please combine the following usefull contextual information to complete the answer to the prompt: <<CONTEXTUAL_INFORMATION>> 
    Here is the original prompt: <<INPUT_PROMPT>> 
    
    Please provide a final answer to the prompt.
    """

    context_text = ""
    for home_url, subpage_link_list in home_url_subpage_link_dict.items():
        for subpage_link in subpage_link_list:

            # get the df text from the row with the subpage_link == subpage_link
            matching_rows = df[df['page_url'] == subpage_link]
            if not matching_rows.empty:
                text = matching_rows.iloc[0]['text']
                context_text += f"URL: {subpage_link}\n{text}\n\n---\n\n"


    # now often the context is way too long so we cut it first 9000000 chars
    print("Context text length: ", len(context_text))
    context_text = context_text[:450000]
    print("Context text length after cutting: ", len(context_text))
    final_answer = combine_intermediate_answers(context_text, input_prompt, final_answer_prompt, num_tokens_final_answer, client)

    print("Final answer \n\n: ", final_answer, "\n\n")


    # Initialize message history with the initial conversation including intermediate answers
    message_history = [
        {"role": "user", "content": input_prompt},
        {"role": "system", "content": context_text},
        {"role": "assistant", "content": final_answer}
    ]
    
    # DONT SHOW INTERMEDIATE ANSWERS TO THE USER
    # Print intermediate answers for the user to reference
    #print("\nIntermediate answers for reference:")
    #for home_url, answer in intermediate_answers.items():
    #    print(f"\nSource: [{home_url}]")
    #    print(f"Answer: {answer}")
    #    print("\n---")
    
    # Loop to handle follow-up questions
    while True:
        follow_up = input("\nAsk a follow-up question (or type 'exit' to quit): ")
        
        if follow_up.lower() == 'exit':
            print("Exiting conversation. Goodbye!")
            break
        
        # Add the follow-up question to message history
        message_history.append({"role": "user", "content": follow_up})
        
        # Get response using the entire conversation context
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                max_tokens=num_tokens_final_answer
            )
            
            follow_up_answer = response.choices[0].message.content
            print(f"\nAnswer: {follow_up_answer}\n")
            
            # Add the response to message history
            message_history.append({"role": "assistant", "content": follow_up_answer})
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
    
if __name__ == "__main__":
    
    main()