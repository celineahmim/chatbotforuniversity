#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - API Flask pour le chatbot RAG avec mémoire conversationnelle
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import unicodedata
import re
from openai import OpenAI
import numpy as np
import pandas as pd
import faiss
from fuzzywuzzy import fuzz
from tqdm import tqdm
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE      = os.path.dirname(__file__)
EXCEL     = os.path.join(BASE, "excel.xlsx")
TRAIN_JL  = os.path.join(BASE, "train.jsonl")
VALID_JL  = os.path.join(BASE, "valid.jsonl")
NEG_JL    = os.path.join(BASE, "negative.jsonl")

OPENAI_KEY  = "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"
FT_MODEL    = "ft:gpt-3.5-turbo-1106:myself::Bh4XZMK0"
EMBED_MODEL = "text-embedding-ada-002"

TOP_K, SIM_THRESHOLD, FUZZY_THR = 8, 0.20, 80
client = OpenAI(api_key=OPENAI_KEY)

corpus = []
index = None
df = None

def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ",
                  unicodedata.normalize("NFKD", text.lower()))).strip()

def initialize_chatbot():
    global corpus, index, df
    logger.info("\U0001f504 Initialisation du chatbot RAG...")
    try:
        if os.path.exists(EXCEL):
            df = pd.read_excel(EXCEL)
            corpus = [" | ".join(f"{c}: {row[c]}" for c in df.columns)
                      for _, row in df.iterrows()]
            logger.info(f"\U0001f4ca Chargé {len(df)} lignes depuis Excel")
        else:
            logger.warning("\u26a0\ufe0f Fichier Excel non trouvé")
            df = pd.DataFrame()

        for path in (TRAIN_JL, VALID_JL):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            q = obj["messages"][0]["content"].lstrip("Q: ").strip()
                            a = obj["messages"][1]["content"].strip()
                            corpus.append(f"QUESTION: {q} RÉPONSE: {a}")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                logger.info(f"\U0001f4dd Chargé {path}")

        if not corpus:
            logger.error("❌ Aucune donnée trouvée")
            return False

        logger.info(f"📚 Corpus total: {len(corpus)} documents")
        logger.info("⏳ Création des embeddings...")
        vectors = []
        for i in tqdm(range(0, len(corpus), 100), desc="Embeddings"):
            batch = corpus[i:i+100]
            try:
                response = client.embeddings.create(input=batch, model=EMBED_MODEL)
                vectors.extend([d.embedding for d in response.data])
            except Exception as e:
                logger.error(f"Erreur embedding batch {i}: {e}")
                vectors.extend([[0.0] * 1536 for _ in batch])

        vectors = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        logger.info("✅ Chatbot initialisé avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def reformulate_question(question: str) -> str:
    prompt = (
        "Tu reformules les questions pour qu'elles soient autonomes et explicites.\n\n"
        f"Nouvelle question : {question}\nQuestion reformulée :"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur reformulation: {e}")
        return question

def formulate_answer(context: str, question: str, history: list) -> str:
    messages = [{"role": "system", "content": "Tu es un assistant universitaire affilié au departement informatique de l'université Abderahmane mira ( Bejaia ) qui répond toujours de manière précise et complète. Les questions auxquelles tu as le droit de répondre ne concerne QUE les questions qui le département informatique de l'université de Bejaia."}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"Contexte : {context}\n\nQuestion : {question}"})
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur reformulation finale: {e}")
        return context or "Je ne sais pas."

def rerank_passages(question: str, passages: list) -> str:
    if not passages:
        return ""
    if len(passages) == 1:
        return passages[0]
    joined = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = f"Question : {question}\n\nExtraits :\n{joined}\n\nDonne le NUMÉRO le plus pertinent :"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        output = response.choices[0].message.content
        match = re.search(r"\d", output)
        idx = int(match.group()) if match else 0
        return passages[max(0, min(idx, len(passages)-1))]
    except Exception as e:
        logger.error(f"Erreur reranking: {e}")
        return passages[0]

def fuzzy_search_excel(question: str) -> str:
    if df is None or df.empty:
        return None
    best_score, best_idx = 0, None
    for i, row in df.iterrows():
        score = max(fuzz.token_set_ratio(question, str(v)) for v in row.values)
        if score > best_score:
            best_score, best_idx = score, i
    if best_score >= FUZZY_THR:
        return " | ".join(f"{c}: {df.loc[best_idx, c]}" for c in df.columns)
    return None

def answer_question(question: str, history: list) -> str:
    try:
        reformed_q = reformulate_question(question)
        embedding_response = client.embeddings.create(input=reformed_q, model=EMBED_MODEL)
        question_embedding = np.array(embedding_response.data[0].embedding, dtype="float32")[None, :]
        similarities, indices = index.search(question_embedding, TOP_K)
        candidates = [corpus[i] for sim, i in zip(similarities[0], indices[0]) if sim >= SIM_THRESHOLD]
        if not candidates:
            fuzzy_result = fuzzy_search_excel(reformed_q)
            if fuzzy_result:
                candidates.append(fuzzy_result)
        context = rerank_passages(reformed_q, candidates) if candidates else ""
        answer = formulate_answer(context or "N/A", reformed_q, history)
        if (context == "" and "je ne sais" not in answer.lower()) or \
           (context and len(set(normalize_text(context).split()) & set(normalize_text(answer).split())) < 1) or \
           any(k in answer.lower() for k in ["probablement", "je pense", "il se pourrait", "doit être"]):
            return "Je ne sais pas."
        return answer
    except Exception as e:
        logger.error(f"Erreur génération réponse: {e}")
        return "Je rencontre un problème technique. Veuillez réessayer."

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '').strip()
        history = data.get('history', [])
        user_id = data.get('user_id', 'anonymous')
        if not question:
            return jsonify({'error': 'Question vide'}), 400
        logger.info(f"👤 {user_id}: {question}")
        answer = answer_question(question, history)
        logger.info(f"🤖 Réponse: {answer}")
        return jsonify({'answer': answer, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Erreur API chat: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'corpus_size': len(corpus) if corpus else 0,
        'index_ready': index is not None
    })

if __name__ == '__main__':
    print("🚀 Démarrage du serveur chatbot...")
    if initialize_chatbot():
        print("✅ Chatbot prêt !")
        print("🌐 Frontend: http://localhost:5000")
        print("📡 API: http://localhost:5000/api/chat")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Erreur lors de l'initialisation")