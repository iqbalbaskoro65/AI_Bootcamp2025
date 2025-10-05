import json
import ollama
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Memuat model dan data lokal...")

llm_agent = ollama.Client(host="http://localhost:11434")
embedder = SentenceTransformer('BAAI/bge-m3')

# muat dataset dari file CSV
df_recipes = pd.read_csv("dataset_baru.csv")
df_recipes['text_for_embedding'] = df_recipes['Title Cleaned'] + " " + df_recipes['Ingredients Cleaned']

# generasi embedding untuk semua resep
corpus_embeddings = embedder.encode(df_recipes['text_for_embedding'].tolist(), convert_to_tensor=True)

print("Model dan data lokal berhasil dimuat.")
def search_document_local(query, k_top=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # hitung kesamaan kosinus
    similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), corpus_embeddings.cpu().numpy()).flatten()
    
    # dapatkan indeks dari k_top resep paling relevan
    top_indices = np.argsort(similarities)[::-1][:k_top]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': df_recipes.iloc[idx]['text_for_embedding'],
            'original_title': df_recipes.iloc[idx]['Title Cleaned'],
            'distance': 1 - similarities[idx]
        })
    return results

def response_query_local(query):
    retrieved_doc = search_document_local(query)
    if not retrieved_doc:
        return "Maaf, saya tidak dapat menemukan resep yang sesuai dengan permintaan Anda di dalam data resep."
        
    context = "\n".join([f"Resep: {doc['original_title']}\n{doc['text']}" for doc in retrieved_doc])
    prompt = f"kamu adalah seorang chef ahli masakan indonesia. jawab pertanyaan berikut hanya berdasarkan konteks resep yang disediakan. jangan menambahkan informasi dari luar konteks. \n\nkonteks resep:\n{context} \n\npertanyaan: {query}\n\njawaban (gunakan point-point seperti bullet di microsoft word):"
    
    response = llm_agent.chat(model="gemma3:4b", messages=[
        {
            'role': 'user',
            'content':prompt
        }
    ])
    return response['message']['content']

if __name__ == "__main__":
    print("Chatbot Dimulai (menggunakan data lokal)")
    while True:
        query_text = input("Masukkan resep yang ingin anda tahu : ")
        if query_text.lower() in ['exit', 'quit', 'q']:
            print("Closing Chatbot....")
            break

        response = response_query_local(query=query_text)
        print("Resep : ", response)
print("Chatbot Selesai")