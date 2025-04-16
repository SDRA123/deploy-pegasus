from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
import re
from collections import OrderedDict

nltk.download('punkt')

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "saadr1231/legal-pegasus-mcs-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

app = FastAPI()

class SummaryRequest(BaseModel):
    text: str
    reference_summary: str

def clean_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_document(doc_text, original_summary):
    cleaned_text = clean_text(doc_text)
    target_length = len(original_summary.split())

    doc_sentences = sent_tokenize(cleaned_text)
    chunk = " ".join(doc_sentences[:30])  # Example simplification for chunking

    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=5,
            min_length=20,
            max_length=150,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    return summary

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    result = summarize_document(req.text, req.reference_summary)
    return {"summary": result}
