import os
import numpy as np
from PyPDF2 import PdfReader
import docx
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import gradio as gr
import re

###########################
# CONFIGURATION
###########################

DATA_DIR = "."  # Look in the root directory of the Space
MAX_CHUNKS = 10                   # Smaller batches = faster loading & inference
MAX_RESPONSE_CHARS = 600          # Hard cutoff to prevent runaway generation

###########################
# DATA EXTRACTION & CHUNKING
###########################

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = ''
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + ' '
        return text
    elif ext == '.docx':
        docf = docx.Document(file_path)
        return ' '.join([p.text for p in docf.paragraphs])
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def chunk_text(text, max_len=300):
    sents = text.split('. ')
    chunks, chunk = [], ''
    for s in sents:
        if len(chunk) + len(s) < max_len:
            chunk += s + '. '
        else:
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = s + '. '
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

all_chunks, all_meta = [], []
if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(('.pdf', '.docx', '.txt')):
            continue
        file_path = os.path.join(DATA_DIR, fname)
        subject = 'general'
        try:
            text = extract_text(file_path)
            c = chunk_text(text, max_len=300)
            for chunk in c:
                if len(chunk) > 40:
                    all_chunks.append(chunk)
                    all_meta.append({'subject': subject, 'file': fname})
                    if len(all_chunks) >= MAX_CHUNKS:
                        break
            if len(all_chunks) >= MAX_CHUNKS:
                break
        except Exception as e:
            print(f"Failed {file_path}: {e}")
        if len(all_chunks) >= MAX_CHUNKS:
            break
    print('Chunks loaded:', len(all_chunks))
else:
    print(f"DATA_DIR '{DATA_DIR}' not found or empty! Please upload your textbook files.")

###########################
# EMBEDDINGS & INDEX
###########################

if len(all_chunks) > 0:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encoder.encode(all_chunks, show_progress_bar=False, batch_size=8)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
else:
    encoder, embeddings, index = None, None, None

###########################
# CLEAN-UP FUNCTION
###########################

def clean_and_remove_incomplete_last_sentence(text):
    """
    Removes duplicate sentences and deletes the last (possibly incomplete) sentence
    if it doesn't end with a period (or ! or ?).
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen, unique = set(), []
    for s in sentences:
        st = s.strip()
        s_key = st.lower()
        if st and s_key not in seen:
            unique.append(st)
            seen.add(s_key)
    if unique and not unique[-1].endswith(('.', '!', '?')):
        unique = unique[:-1]
    return ' '.join(unique).strip()

###########################
# ANSWER FUNCTION
###########################

def answer_query(
    question,
    subject=None,
    top_k=1,
    model_name="Qwen/Qwen2-0.5B"
):
    if encoder is None or index is None:
        return "No textbooks found or processed."

    q_emb = encoder.encode([question])
    faiss.normalize_L2(q_emb)
    D, I = index.search(np.asarray(q_emb, dtype='float32'), top_k)
    found = []
    for idx in I[0]:
        if subject:
            if subject.lower() in all_meta[idx]['subject'].lower():
                found.append(all_chunks[idx])
        else:
            found.append(all_chunks[idx])
        if len(found) >= top_k: break
    context = "\n".join(found)

    prompt = (
        "You are an ICSE board AI tutor. Answer ONLY using the textbook excerpt below. "
        "Give a COMPLETE, single, concise answer, ending with proper punctuation. "
        "Do NOT repeat yourself or echo the prompt. End your answer with a full stop.\n\n"
        f"Textbook excerpt:\n{context}\n\n"
        f"Student question: {question}\n"
        "Answer:"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=64,  # or even 80 for slightly longer output
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    output = pipe(prompt)[0]['generated_text']

    answer_part = output.split("Answer:")[-1].strip()
    for stop_word in ["Textbook", "Student question:", "Context:"]:
        if stop_word in answer_part:
            answer_part = answer_part.split(stop_word)[0].strip()
    clean = clean_and_remove_incomplete_last_sentence(answer_part)
    if len(clean) > MAX_RESPONSE_CHARS:
        clean = clean[:MAX_RESPONSE_CHARS] + "\n\n[truncated]"
    if not clean.strip():
        return "Sorry, no answer found in the textbook."
    return clean

###########################
# GRADIO APP
###########################

def gradio_chatbot(user_input, subject=""):
    subject_input = subject.strip() if subject.strip() else None
    return answer_query(user_input, subject=subject_input)

TITLE = "ICSE Textbook Chatbot"
DESC = "Ask any question using ONLY your uploaded ICSE textbooks. The AI answers strictly from your syllabus content."

iface = gr.Interface(
    fn=gradio_chatbot,
    inputs=[
        gr.Textbox(label="Your question", placeholder="Type your question here...", lines=2),
        gr.Textbox(label="Subject (optional)", placeholder="e.g., Physics, Geography", lines=1),
    ],
    outputs="text",
    title=TITLE,
    description=DESC,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True)
