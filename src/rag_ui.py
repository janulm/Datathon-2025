import gradio as gr
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import openai
import os
import time

# Import functions from rag_agent
from rag_agent import (
    load_embeddings_dataset,
    load_list_parquets,
    prompt_expansion,
    get_home_url_subpage_link_dict,
    expand_sub_urls,
    get_intermediate_answers,
    combine_intermediate_answers,
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
num_tokens_prompt_expansion = 200
num_tokens_intermediate_answer = 2000
num_tokens_final_answer = 10000
max_good_subpage_links = 30

# Concept dictionary for prompt expansion
concept_dict = {
    "location": "Location of the company.",
    "industry": "Industry of the company.",
    "company": "Company of the company.",
    "product": "Product of the company.",
    "service": "Service of the company.",
    "technology": "Technology of the company.",
}

# Prompts
expansion_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
For this purpose we take the input prompt and expand with respect to the meaning of a specific concept. 
Please elaborate on how the input prompt is related to the concept and what concept specific answer you would want to get given this prompt. 
The original prompt is: <<INPUT_PROMPT>> and the concept is: <<CONCEPT>>: Please directly output the expanded prompt without any additional text or answer REMOVE if you dont think the concept is relevant to answer the prompt."""

intermediate_answer_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
Given this prompt: <<INPUT_PROMPT>> and this list of webpages that correspond to a company that might be relevant to answer the prompt. 
Please try answer the prompt with the information provided in the webpages or state the that webpages are not relevant to the prompt: <<SUBPAGE_LINK_LIST_TEXTS>> 
Please return your answer in a structured format such that this intermediate answer can be combined into a final answer lateron. Please directly output the intermediate answer without any additional text."""

combination_prompt = """You are part of an AI Agent that is trying to answer a user prompt. 
Please combine the following intermediate answers into a final answer. It might be the case that some intermediate answers are not relevant to the prompt and they will state that. Please use the URL as a source for the answer where parts of each intermediate answer are used and output the URL in the answer. If multiples companies or products or things are mentioned that match the prompt query, please mention them in the answer ordered by relevance to the prompt.
Here is the original prompt: <<INPUT_PROMPT>> and here are the intermediate answers: <<INTERMEDIATE_ANSWERS.>> 

Please provide a final answer to the prompt based on the intermediate answers. Please directly output the final answer without any additional text. Please start from the intermediate answers which you think are most relevant to the prompt and give an answer in natural language."""

# Global state variables for tracking progress
current_status = ""
chat_history_for_updates = None

def set_status(status):
    """Updates the current status and updates the UI if possible"""
    global current_status, chat_history_for_updates
    current_status = status
    print(f"Status update: {status}")

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

def update_progress(history, message, progress_text):
    """Updates the progress in the chat history"""
    history[-1] = (message, progress_text)
    return history

def process_initial_query(question):
    global df, embeddings_matrix, client, model, message_history, current_status
    
    # Check if system is initialized
    if df is None or embeddings_matrix is None or client is None:
        return "System not initialized. Please initialize first."
    
    print(f"Processing initial query: {question}")
    
    # Set status for expanding prompts
    set_status("Expanding prompt concepts... 🔄")
    
    # Expand the prompt
    list_of_prompts = prompt_expansion(question, concept_dict, expansion_prompt, client, num_tokens_prompt_expansion)
    print("Prompt expansion complete.")
    
    # Set status for finding information
    set_status("Finding relevant information... 🔍")
    
    # Embed the expanded prompts
    embed_prompts = model.encode(list_of_prompts, convert_to_tensor=True)
    embed_prompts = embed_prompts.clone().detach().to(device)
    
    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(embed_prompts, embeddings_matrix)
    
    # Get relevant documents
    home_url_subpage_link_dict = get_home_url_subpage_link_dict(cos_scores, max_good_subpage_links, None, None, df)
    home_url_subpage_link_dict = expand_sub_urls(home_url_subpage_link_dict, df)
    
    # Set status for generating intermediate answers
    set_status("Generating intermediate answers from sources... ⚙️")
    
    # Get intermediate answers
    intermediate_answers = get_intermediate_answers(question, intermediate_answer_prompt, home_url_subpage_link_dict, df, num_tokens_intermediate_answer, client)
    
    # Set status for creating final answer
    set_status("Creating final answer from all sources... 🧠")
    
    # Combine intermediate answers
    final_answer = combine_intermediate_answers(intermediate_answers, question, combination_prompt, num_tokens_final_answer, client)
    
    # Create a formatted string of intermediate answers for context
    formatted_intermediate_answers = "\n\n--- INTERMEDIATE ANSWERS FOR REFERENCE ---\n\n"
    for home_url, answer in intermediate_answers.items():
        formatted_intermediate_answers += f"Source: [{home_url}]\nAnswer: {answer}\n\n---\n\n"
    
    # Initialize or update chat history
    message_history = [
        {"role": "user", "content": question},
        {"role": "system", "content": formatted_intermediate_answers},
        {"role": "assistant", "content": final_answer}
    ]
    
    print("Initial query processing complete.")
    set_status("Complete ✓")
    
    return final_answer

def process_follow_up(follow_up_question):
    global client, message_history, current_status
    
    if message_history is None:
        return "Please ask an initial question first."
    
    print(f"Processing follow-up question: {follow_up_question}")
    set_status("Generating response... 🔄")
    
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
        
        set_status("Complete ✓")
        return follow_up_answer
        
    except Exception as e:
        error_message = f"Error generating answer: {str(e)}"
        print(error_message)
        set_status("Error ❌")
        return error_message

def status_checker():
    """Function to check status updates and update the UI"""
    global current_status, chat_history_for_updates
    last_status = ""
    
    # This is a loop that will continuously check for status updates
    if chat_history_for_updates is not None and current_status != last_status:
        last_status = current_status
        
        # Only update if we have an active query
        if len(chat_history_for_updates) > 0:
            message = chat_history_for_updates[-1][0]
            chat_history_for_updates[-1] = (message, current_status)
            
            # Return updated history and reset for next check
            return chat_history_for_updates
    
    # If no updates, return None to indicate no changes
    return None

def respond(message, history):
    global message_history, current_status, chat_history_for_updates
    
    # Reset status at the beginning of each new query
    current_status = "Processing your question... Please wait."
    
    # Create initial message with status
    processing_history = history + [(message, current_status)]
    chat_history_for_updates = processing_history
    yield "", processing_history
    
    # Process based on whether this is an initial query or follow-up
    if message_history is None:
        # Process the initial query with status updates at each stage
        
        # Update 1: Expanding prompts
        current_status = "Expanding prompt concepts... 🔄"
        chat_history_for_updates[-1] = (message, current_status)
        yield "", chat_history_for_updates
        
        # Expand prompts - use positional arguments to match the original function
        list_of_prompts = prompt_expansion(message, concept_dict, expansion_prompt, client, num_tokens_prompt_expansion)
        
        # Update 2: Finding information
        current_status = "Finding relevant information... 🔍"
        chat_history_for_updates[-1] = (message, current_status)
        yield "", chat_history_for_updates
        
        # Process embeddings and find documents
        embed_prompts = model.encode(list_of_prompts, convert_to_tensor=True)
        embed_prompts = embed_prompts.clone().detach().to(device)
        cos_scores = util.pytorch_cos_sim(embed_prompts, embeddings_matrix)
        home_url_subpage_link_dict = get_home_url_subpage_link_dict(cos_scores, max_good_subpage_links, None, None, df)
        home_url_subpage_link_dict = expand_sub_urls(home_url_subpage_link_dict, df)
        
        # Update 3: Generating intermediate answers
        current_status = "Generating intermediate answers from sources... ⚙️"
        chat_history_for_updates[-1] = (message, current_status)
        yield "", chat_history_for_updates
        
        # Get intermediate answers - use positional arguments
        intermediate_answers = get_intermediate_answers(message, intermediate_answer_prompt, 
                                                     home_url_subpage_link_dict, df, 
                                                     num_tokens_intermediate_answer, client)
        
        # Update 4: Creating final answer
        current_status = "Creating final answer from all sources... 🧠"
        chat_history_for_updates[-1] = (message, current_status)
        yield "", chat_history_for_updates
        
        # Get final answer - use positional arguments
        answer = combine_intermediate_answers(intermediate_answers, message, 
                                           combination_prompt, num_tokens_final_answer, client)
        
        # Create a formatted string of intermediate answers for context
        formatted_intermediate_answers = "\n\n--- INTERMEDIATE ANSWERS FOR REFERENCE ---\n\n"
        for home_url, answer_text in intermediate_answers.items():
            formatted_intermediate_answers += f"Source: [{home_url}]\nAnswer: {answer_text}\n\n---\n\n"
        
        # Initialize or update chat history
        message_history = [
            {"role": "user", "content": message},
            {"role": "system", "content": formatted_intermediate_answers},
            {"role": "assistant", "content": answer}
        ]
    else:
        # Update for follow-up question
        current_status = "Generating response... 🔄"
        chat_history_for_updates[-1] = (message, current_status)
        yield "", chat_history_for_updates
        
        # Process the follow-up
        answer = process_follow_up(message)
    
    # Update with the actual answer directly
    chat_history_for_updates[-1] = (message, answer)
    final_history = history + [(message, answer)]
    yield "", final_history
    
    # Clear the reference
    chat_history_for_updates = None

def clear_chat():
    global message_history
    message_history = None
    return None

# Create Gradio Interface
with gr.Blocks(title="Concept RAG Chatbot") as demo:
    gr.Markdown("# Concept RAG Chatbot")
    gr.Markdown("Ask questions about companies and their products, services, or technologies.")
    
    # Initialize the system right away
    initialize_system()
    
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Your question", placeholder="Ask about companies, products, or technologies...")
    clear = gr.Button("Clear")
    
    # Setup the UI interactions
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, None, chatbot, queue=False)
    
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
