import json
import boto3
text1 = "Translate to German:  My name is Arthur"
text2 = "A step by step recipe to make bolognese pasta:"

newline, bold, unbold = '\n', '\033[1m', '\033[0m'
endpoint_name = 'jumpstart-dft-hf-text2text-flan-t5-xl'
def query_endpoint(encoded_text):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-text', Body=encoded_text)
    return response

def parse_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_text']
    return generated_text

for text in [text1, text2]:
    query_response = query_endpoint(text.encode('utf-8'))
    generated_text = parse_response(query_response)
    print (f"Inference:{newline}"
            f"input text: {text}{newline}"
            f"generated text: {bold}{generated_text}{unbold}{newline}")
    
"""
This model also supports many advanced parameters while performing inference. They include:

max_length: Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a positive integer.
num_return_sequences: Number of output sequences returned. If specified, it must be a positive integer.
num_beams: Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.
no_repeat_ngram_size: Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
temperature: Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.
early_stopping: If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
do_sample: If True, sample the next word as per the likelihood. If specified, it must be boolean.
top_k: In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.
top_p: In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.
seed: Fix the randomized state for reproducibility. If specified, it must be an integer.
We may specify any subset of the parameters mentioned above while invoking an endpoint. Next, we show an example of how to invoke endpoint with these arguments
"""
#Input must be a json
payload = {"text_inputs":"write me an article of aws sage maker", "max_length":200, "num_return_sequences":3, "top_k":50, "top_p":0.95, "do_sample":True}

def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))

def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_texts']
    return generated_text

generated_texts = parse_response_multiple_texts(query_response)
print(generated_texts)