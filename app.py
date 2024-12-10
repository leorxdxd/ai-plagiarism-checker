from flask import Flask, request, render_template
from langdetect import detect
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import os
import chardet
import re
import json
from werkzeug.utils import secure_filename
import PyPDF2
import docx

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Initialize models
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # For semantic similarity
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
    text = text.lower().strip()         # Lowercase and strip spaces
    return text


# Function to extract text from uploaded files
def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    text = ""

    if ext == ".pdf":
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join(page.extract_text() for page in reader.pages)
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        text = " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)
            encoding = detected_encoding['encoding']
            text = raw_data.decode(encoding, errors='ignore')

    return preprocess_text(text)

# Function for plagiarism detection
def detect_plagiarism(input_text, reference_texts):
    input_embedding = semantic_model.encode(input_text, convert_to_tensor=True)
    scores = []
    for ref_text in reference_texts:
        ref_embedding = semantic_model.encode(ref_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(input_embedding, ref_embedding)
        scores.append(similarity.item())
    return max(scores)  # Return the highest similarity score

# Route for home page
@app.route('/')
def home():
    return render_template('upload.html')

# Route for plagiarism check
@app.route('/check', methods=['POST'])
def check_content():
    result = None
    error = None
    input_text = request.form.get('text', "").strip()
    uploaded_file = request.files.get('file')
    reference_file = './reference_texts.json'  # Path to stored reference texts

    try:
        if not os.path.exists(reference_file):
            with open(reference_file, 'w') as f:
                json.dump(["This is a sample reference text."], f)  # Create a dummy reference

        with open(reference_file, 'r') as f:
            reference_texts = json.load(f)

        # Extract text from input
        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            input_text = extract_text_from_file(file_path)

        if input_text:
            input_text = preprocess_text(input_text)
            word_count = len(input_text.split())
            
            if word_count < 100:
                error = f"Text must be at least 100 words. Current word count: {word_count}."
            else:
                score = detect_plagiarism(input_text, reference_texts)
                result = (
                    f"Plagiarism Results:\n\n"
                    f"Plagiarism Score: {score * 100:.2f}%\n"
                    f"Unique Content: {100 - (score * 100):.2f}%"
                )
        else:
            error = "No text provided for plagiarism check."

    except Exception as e:
        error = f"Error: {str(e)}"

    return render_template(
        'upload.html',
        result=result,
        error=error,
        word_count=len(input_text.split()) if input_text else 0
    )

# Route for paraphrasing
@app.route('/paraphrase', methods=['POST'])
def paraphrase_text():
    input_text = request.form.get('paraphrase-text', "").strip()
    paraphrased_texts = None
    error = None

    try:
        if input_text:
            # Paraphrasing logic
            input_ids = t5_tokenizer.encode(f"paraphrase: {input_text} </s>", return_tensors="pt").to(device)
            outputs = t5_model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
            paraphrased_texts = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        else:
            error = "Please provide text to paraphrase."
    except Exception as e:
        error = f"Error: {str(e)}"

    return render_template(
        'upload.html',
        paraphrased_texts=paraphrased_texts,
        error=error
    )

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000)
