from flask import Flask, request, render_template
from langdetect import detect
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os
import chardet  # To detect file encoding
import re
from transformers import pipeline
import openai

openai.api_key = "OPENAI_API_KEY"
# Initialize Flask app
app = Flask(__name__)

# Initialize BERT tokenizer and model for multilingual support
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Initialize T5-base model for paraphrasing (more efficient version)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')  # Switched to a smaller version
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize PEGASUS model for better paraphrasing
pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)


def paraphrase_with_gpt(text):
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=f"Paraphrase the following text:\n{text}\n", 
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Function to preprocess text (basic text cleaning)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Function to paraphrase text using T5
def paraphrase(text):
    # Specify a max_length for tokenization and ensure truncation is done properly
    input_ids = t5_tokenizer.encode(f"paraphrase: {text}", return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Generate paraphrased text using the model
    output_ids = t5_model.generate(
        input_ids, 
        max_length=512,  # Limit the output length to 512 tokens
        num_beams=5, 
        num_return_sequences=3,  # Generate multiple paraphrases
        temperature=1.0,  # Increase creativity for diverse outputs
        early_stopping=True
    )
    
    # Decode and return the paraphrased text
    paraphrased_texts = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return paraphrased_texts  # Return a list of paraphrased outputs



# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()

# Function to calculate similarity using BERT embeddings
def calculate_similarity_with_bert(doc1, doc2):
    embedding1 = get_bert_embedding(doc1)
    embedding2 = get_bert_embedding(doc2)
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cosine_similarity.item() * 100

# Load a pre-trained pipeline for AI detection
ai_detector = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_ai_generated(text):
    result = ai_detector(text, candidate_labels=["AI Generated", "Human Written"])
    label = result['labels'][0]  # Take the label with the highest score
    return label == "AI Generated"

# Function to read and preprocess file with correct encoding
def read_file_with_encoding(file):
    raw_data = file.read()
    detected_encoding = chardet.detect(raw_data)
    encoding = detected_encoding['encoding']
    try:
        text = raw_data.decode(encoding)
        return text
    except (UnicodeDecodeError, TypeError):
        return raw_data.decode('ISO-8859-1')

# Function to categorize the plagiarism similarity score
def categorize_plagiarism_score(similarity_score):
    if similarity_score < 15:
        return "Very Low"
    elif similarity_score < 40:
        return "Low"
    elif similarity_score < 70:
        return "Moderate"
    else:
        return "High"

# Function to count words in a text
def count_words(text):
    words = text.split()
    return len(words)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    result = None
    error = None
    input_text = request.form.get('text', "")
    plagiarism_level = None
    ai_generated = None
    paraphrased_text = None

    uploaded_file = request.files.get('file')

    try:
        # If file is uploaded
        if uploaded_file and uploaded_file.filename != '':
            uploaded_content = read_file_with_encoding(uploaded_file)
            uploaded_content = preprocess_text(uploaded_content)

            word_count = count_words(uploaded_content)
            if word_count < 100:
                result = "The uploaded document must contain at least 100 words."
            else:
                with open('reference.txt', 'r') as ref_file:
                    reference_content = preprocess_text(ref_file.read())
                similarity_score = calculate_similarity_with_bert(uploaded_content, reference_content)
                plagiarism_level = categorize_plagiarism_score(similarity_score)
                ai_generated = is_ai_generated(uploaded_content)
                paraphrased_text = paraphrase(uploaded_content)  # Paraphrase the uploaded content
                result = f"Similarity Score: {similarity_score:.2f}% (Plagiarism Level: {plagiarism_level})"

        # If text is entered
        elif input_text.strip():
            user_text = preprocess_text(input_text)

            word_count = count_words(user_text)
            if word_count < 100:
                result = "Your text must contain at least 100 words."
            else:
                with open('reference.txt', 'r') as ref_file:
                    reference_content = preprocess_text(ref_file.read())
                similarity_score = calculate_similarity_with_bert(user_text, reference_content)
                plagiarism_level = categorize_plagiarism_score(similarity_score)
                ai_generated = is_ai_generated(user_text)
                paraphrased_text = paraphrase(user_text)  # Paraphrase the entered text
                result = f"Similarity Score: {similarity_score:.2f}% (Plagiarism Level: {plagiarism_level})"

        # If no input is provided
        else:
            error = "Please provide a document or text to check."

    except Exception as e:
        error = f"Error: {str(e)}"

    return render_template(
        'upload.html',
        result=result,
        error=error,
        input_text=input_text,
        ai_generated=ai_generated,
        paraphrased_text=paraphrased_text
    )

@app.route('/paraphrase', methods=['POST'])
def paraphrase_text():
    input_text = request.form.get('paraphrase-text', "")
    error = None
    paraphrased_text = None

    try:
        if input_text.strip():
            paraphrased_text = paraphrase(input_text)
        else:
            error = "Please provide text to paraphrase."
    except Exception as e:
        error = f"Error: {str(e)}"

    return render_template(
        'upload.html',
        paraphrased_text=paraphrased_text,  # Pass paraphrased text as a list
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
