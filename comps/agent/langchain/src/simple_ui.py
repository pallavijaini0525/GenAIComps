import gradio as gr
import requests
import os

AGENT_SERVICE_ENDPOINT = "http://localhost:9095"

def chat(input_text):
    print(AGENT_SERVICE_ENDPOINT)
    
    # Send the request to the API
    response = requests.post(AGENT_SERVICE_ENDPOINT + "/v1/chat/completions", json={"query": input_text})
    
    # Print the raw response for debugging
    print("Response:", response.text)
    
    # Get the raw text response
    response_text = response.text
    
    # Determine if the response uses single or double quotes for data
    data_start_single = response_text.find("data: '")
    data_start_double = response_text.find('data: "')
    
    # Find the start index based on the quote type
    if data_start_single != -1:
        data_start = data_start_single + len("data: '")
        delimiter = "<|eot_id|>"
    elif data_start_double != -1:
        data_start = data_start_double + len('data: "')
        delimiter = '<|eot_id|>"'
    else:
        # If neither is found, return the full response with a message
        extracted_data = "Data not found in the response."
        log = response_text
        return extracted_data, log
    
    # Find the end index
    data_end = response_text.find(delimiter, data_start)
    if data_end == -1:
        data_end = len(response_text)  # If <|eot_id|> not found, take till end of response
    
    # Clean and return the extracted data and full response
    extracted_data = response_text[data_start:data_end].strip()
    log = response_text
    return extracted_data, log


# Update the Gradio interface to handle multiple outputs
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Input"),
    outputs=[
        gr.Textbox(label="Output"),  # Display extracted data
        gr.Textbox(label="Log")      # Display the full response
    ],
    title="Chat Interface",
    description="Type your query and get the response along with detailed logs."
)

interface.launch(share=True)