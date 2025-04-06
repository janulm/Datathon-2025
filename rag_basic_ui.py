import gradio as gr
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import openai
import os
import time

# Import necessary functions from rag_agent
from rag_basic_agent import (
    load_embeddings_dataset,
    load_list_parquets,
    get_home_url_subpage_link_dict,
    expand_sub_urls,
    async_get_completion
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = model.to(device)
df = None
embeddings_matrix = None
client = None
message_history = None

# Configuration
num_tokens_final_answer = 10000
max_good_subpage_links = 20  # Number of top results to consider

# Final answer prompt
final_answer_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
Please combine the following useful contextual information to complete the answer to the prompt: <<CONTEXTUAL_INFORMATION>> 
Here is the original prompt: <<INPUT_PROMPT>> 

Please provide a final answer to the prompt. Use the URLs as sources for your answer where appropriate.
"""

def initialize_system():
    global df, embeddings_matrix, client
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key="haha")
    
    # Load dataset
    #df = load_embeddings_dataset("embeddings_dataset_20000.parquet")
    list_of_paths = [f"/home/janulm/Documents/projects/datathon/embeddings_dataset_{i*100000}_{(i+1)*100000}.parquet" for i in range(8)]
    df = load_list_parquets(list_of_paths)
    # Create embeddings matrix
    embeddings_matrix = torch.tensor(np.array(df['embd_vector'].tolist()), dtype=torch.float32)
    embeddings_matrix = embeddings_matrix.to(device)
    
    print(f"System initialized. Embeddings matrix shape: {embeddings_matrix.shape}")
    return "System initialized and ready for queries!"

def combine_intermediate_answers(context_text, input_prompt, final_answer_prompt, num_tokens_final_answer, client):
    print("Computing final answer")
    
    # Prepare the prompt by replacing placeholders
    prompt = final_answer_prompt.replace("INPUT_PROMPT", input_prompt).replace("CONTEXTUAL_INFORMATION", context_text)
    
    # Make OpenAI API call
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=num_tokens_final_answer,
        )
        final_answer = response.choices[0].message.content
        print("Generated final answer")
    except Exception as e:
        print(f"Error generating final answer: {str(e)}")
        final_answer = f"Error generating final answer: {str(e)}"
    
    return final_answer

def process_query(question):
    global df, embeddings_matrix, client, model, message_history
    
    # Check if system is initialized
    if df is None or embeddings_matrix is None or client is None:
        return "System not initialized. Please initialize first."
    
    print(f"Processing query: {question}")
    
    # Embed the query
    embed_prompt = model.encode([question], convert_to_tensor=True)
    embed_prompt = embed_prompt.clone().detach().to(device)
    
    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(embed_prompt, embeddings_matrix)
    
    # Get relevant documents
    home_url_subpage_link_dict = get_home_url_subpage_link_dict(cos_scores, max_good_subpage_links, None, None, df)
    home_url_subpage_link_dict = expand_sub_urls(home_url_subpage_link_dict, df)
    
    # Extract text from relevant documents to create context
    context_text = ""
    for home_url, subpage_link_list in home_url_subpage_link_dict.items():
        for subpage_link in subpage_link_list:
            # Find the row in dataframe matching the subpage link
            matching_rows = df[df['page_url'] == subpage_link]
            if not matching_rows.empty:
                text = matching_rows.iloc[0]['text']
                context_text += f"URL: {subpage_link}\n{text}\n\n---\n\n"
    
    # Get final answer

    # now often the context is way too long so we cut it first 9000000 chars
    print("Context text length: ", len(context_text))
    context_text = context_text[:30000]
    print("Context text length after cutting: ", len(context_text))

    final_answer = combine_intermediate_answers(context_text, question, final_answer_prompt, num_tokens_final_answer, client)
    
    # Initialize or update chat history
    message_history = [
        {"role": "user", "content": question},
        {"role": "system", "content": context_text},
        {"role": "assistant", "content": final_answer}
    ]
    
    return final_answer

def process_follow_up(follow_up_question):
    global client, message_history
    
    if message_history is None:
        return "Please ask an initial question first."
    
    print(f"Processing follow-up question: {follow_up_question}")
    
    # Add the follow-up question to message history
    message_history.append({"role": "user", "content": follow_up_question})
    
    # Get response using the entire conversation context
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,
            max_tokens=num_tokens_final_answer
        )
        
        follow_up_answer = response.choices[0].message.content
        
        # Add the response to message history
        message_history.append({"role": "assistant", "content": follow_up_answer})
        
        return follow_up_answer
        
    except Exception as e:
        error_message = f"Error generating answer: {str(e)}"
        print(error_message)
        return error_message

def clear_chat():
    global message_history
    message_history = None
    return None

# Create Gradio Interface
with gr.Blocks(title="Basic RAG Chatbot") as demo:
    gr.Markdown("# Basic RAG Chatbot")
    gr.Markdown("Ask questions about companies, products, or technologies. The system searches through a database of web pages to find relevant information.")
    
    # Initialize the system right away
    initialize_system()
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about companies, products, or technologies...",
                    scale=9
                )
                submit_btn = gr.Button("Submit", scale=1)
            clear = gr.Button("Clear Chat")
    
    # Setup the UI interactions
    def respond(message, history):
        # If this is the first question or after clearing chat
        if message_history is None:
            bot_message = process_query(message)
        else:
            # This is a follow-up question
            bot_message = process_follow_up(message)
        
        history.append((message, bot_message))
        return "", history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, None, chatbot)
    
    # Add examples
    gr.Examples(
        examples=[
            "Please find companies in the banking industry that also have a product in the field of AI.",
            "Please tell me about AI companies in US",
            "What are some companies working on blockchain technology?"
        ],
        inputs=msg
    )

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(share=True)  # share=True creates a public link
