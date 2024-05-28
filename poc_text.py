import streamlit as st
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import pickle

# Load XLNet model and tokenizer
model_path = 'claim_classifier.pkl'  # Update with your model file path
with open(model_path, 'rb') as file:
    model, tokenizer = pickle.load(file)

# Ensure the model is on CPU
model.to('cpu')

# Load the tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

def classify_claim(claim_text):
    inputs = tokenizer(claim_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Streamlit interface
st.title("Claims Classification with XLNet")
st.write("Enter a claim text to classify:")

user_input = st.text_area("Claim Text")
if st.button("Classify"):
    if user_input:
        result = classify_claim(user_input)
        st.write(f"Predicted class: {result}")
    else:
        st.write("Please enter a claim text.")

