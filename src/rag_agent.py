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


def load_list_parquets(file_paths=["some_path"]):
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
    list_of_prompts = [prompt]

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
    
    #print("Top 3 similarity scores across all prompts:")
    #for i in range(k):
    #    print(f"Prompt index: {prompt_indices[i]}, Document index: {document_indices[i]}, Score: {top_k_values[i]:.4f}")
    
    #print("Mean similarity score: ", cos_scores.mean())
    
    # Here you can implement the logic to group by home_url and count subpages
    # This would require accessing the DataFrame using the document indices
    
    # now for each document_index, get the home_url and the subpage_link and group them by home_url and count the number of distinct subpage_links
    home_url_subpage_link_dict = {}
    for i in range(k):
        #print("Document index: ", document_indices[i])
        subpage_idx = int(document_indices[i])
        #print("Subpage index: ", subpage_idx)
        home_url = df.iloc[subpage_idx]['home_url']
        subpage_link = df.iloc[subpage_idx]['page_url']
        
        if home_url not in home_url_subpage_link_dict:
            home_url_subpage_link_dict[home_url] = []
        # 
        home_url_subpage_link_dict[home_url].append(subpage_link)
    
    #print("Home URL subpage link dict: ", home_url_subpage_link_dict)

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


def combine_intermediate_answers(intermediate_answers, input_prompt, combination_prompt, num_tokens_final_answer, client=None):
    print("Combining intermediate answers")
    
    # Format the intermediate answers into a string
    formatted_answers = []
    for home_url, answer in intermediate_answers.items():
        formatted_answers.append(f"Source: [{home_url}]\nAnswer: {answer}")
    
    all_answers = "\n\n---\n\n".join(formatted_answers)
    
    # Prepare the prompt by replacing placeholders
    prompt = combination_prompt.replace("INPUT_PROMPT", input_prompt).replace("INTERMEDIATE_ANSWERS", all_answers)
    
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
    num_tokens_prompt_expansion = 200
    num_tokens_intermediate_answer = 2000
    num_tokens_final_answer = 10000


    # TODO MAKE THESE PARAMS WORKING: 
    max_good_subpage_links = 20
    
    
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
    input_prompt = "Please find companies in the banking industry and that also have a product in the field of AI. Please tell me about the company and the products they offer."
    input_prompt = "Please tell me about AI companies in US"

    # TODO WAIT FOR USER INPUT HERE
    input_prompt = input("Please enter your prompt: ")

    concept_dict = {
        "location": "Location of the company.",
        "industry": "Industry of the company.",
        "company": "Company of the company.",
        "product": "Product of the company.",
        "service": "Service of the company.",
        "technology": "Technology of the company.",
    }

    expansion_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
    For this purpose we take the input prompt and expand with respect to the meaning of a specific concept. 
    Please elaborate on how the input prompt is related to the concept and what concept specific answer you would want to get given this prompt. 
    The original prompt is: <<INPUT_PROMPT>> and the concept is: <<CONCEPT>>: Please directly output the expanded prompt without any additional text or answer REMOVE if you dont think the concept is relevant to answer the prompt."""

    list_of_prompts = prompt_expansion(input_prompt, concept_dict, expansion_prompt, client, num_tokens_prompt_expansion)
    
    print("List of expanded prompts: ", list_of_prompts, "\n\n")

    
    # embed the expanded prompts
    embed_prompts = model.encode(list_of_prompts, convert_to_tensor=True)
    # make into a tensor
    embed_prompts = embed_prompts.clone().detach()
    # print the shape of the embed_prompts
    print(f"Embed prompts shape: {embed_prompts.shape}")
    
    # move to the device
    embed_prompts = embed_prompts.to(device)


    # compute the cosine similarity between the embed_prompts and the embeddings_matrix
    cos_scores = util.pytorch_cos_sim(embed_prompts, embeddings_matrix)
    # print the shape of the cos_scores
    print(f"Cos scores shape: {cos_scores.shape}")


    # get the top 10 values of the cos_scores:
    # group them by their home_url, and then print out the number of distinct home_urls,
    # and the number of distinct subpages for each home_url.


    home_url_subpage_link_dict = get_home_url_subpage_link_dict(cos_scores,max_good_subpage_links,None,None,df)

    # now expand the sub_urls with the main home_url if its not in the sub_urls already
    home_url_subpage_link_dict = expand_sub_urls(home_url_subpage_link_dict,df)

    
    

    # now we have a dict of home_url and subpage_link_list
    # we now want to compute our intermediate answers given the list of subpage_link_list_texts and the original prompt
    
    intermediate_answer_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
    Given this prompt: <<INPUT_PROMPT>> and this list of webpages that correspond to a company that might be relevant to answer the prompt. 
    Please try answer the prompt with the information provided in the webpages or state the that webpages are not relevant to the prompt: <<SUBPAGE_LINK_LIST_TEXTS>> 
    Please return your answer in a structured format such that this intermediate answer can be combined into a final answer lateron. Please directly output the intermediate answer without any additional text."""

    intermediate_answers = get_intermediate_answers(input_prompt, intermediate_answer_prompt, home_url_subpage_link_dict, df, num_tokens_intermediate_answer, client)

    # then we want to combine all the intermediate answers into a final answer
    
    combination_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
    Please combine the following intermediate answers into a final answer. It might be the case that some intermediate answers are not relevant to the prompt and they will state that. Please use the URL as a source for the answer where parts of each intermediate answer are used and output the URL in the answer.
    Here is the original prompt: <<INPUT_PROMPT>> and here are the intermediate answers: <<INTERMEDIATE_ANSWERS.>> 
    
    Please provide a final answer to the prompt based on the intermediate answers. Please directly output the final answer without any additional text. Please start from the intermediate answers which you think are most relevant to the prompt and give an answer in natural language."""
    final_answer = combine_intermediate_answers(intermediate_answers, input_prompt, combination_prompt, num_tokens_final_answer, client)

    print("Final answer \n\n: ", final_answer, "\n\n")

    # Create a formatted string of intermediate answers for context
    formatted_intermediate_answers = "\n\n--- INTERMEDIATE ANSWERS FOR REFERENCE ---\n\n"
    for home_url, answer in intermediate_answers.items():
        formatted_intermediate_answers += f"Source: [{home_url}]\nAnswer: {answer}\n\n---\n\n"
    
    # Initialize message history with the initial conversation including intermediate answers
    message_history = [
        {"role": "user", "content": input_prompt},
        {"role": "system", "content": formatted_intermediate_answers},
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