import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import requests
import google.generativeai as genai
from collections import deque

# Configure Google Gemini API
GOOGLE_API_KEY = ""  # Leave empty; user enters it in UI
genai.configure(api_key=GOOGLE_API_KEY)

# Store recent queries and responses
context_cache = deque(maxlen=5)  # Keep the last 5 conversations

# Sample book dataset
books = [
    {"title": "Dune", "author": "Frank Herbert", "genre": "Sci-Fi", "description": "A science fiction saga about a desert planet and its people."},
    {"title": "1984", "author": "George Orwell", "genre": "Dystopian", "description": "A dystopian novel about a totalitarian regime that watches everything."},
    {"title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "description": "A hobbit embarks on a journey to reclaim a lost kingdom."}
]

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
descriptions = [book["description"] for book in books]
embeddings = model.encode(descriptions, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search_books(query, genre=None, author=None, top_k=2):
    """
    Searches for books using FAISS and filters results based on genre or author.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k * 3)
    results = [books[idx] for idx in indices[0]]
    if genre:
        results = [book for book in results if book["genre"].lower() == genre.lower()]
    if author:
        results = [book for book in results if book["author"].lower() == author.lower()]
    return results[:top_k]

def generate_book_recommendation_response(query, books, api_key):
    """
    Uses Google Gemini AI to generate a response explaining the book recommendations with streaming output and context caching.
    """
    if not books:
        yield "Sorry, I couldn't find any matching books. Try another query!"
        return

    genai.configure(api_key=api_key)  # Set user-provided API key

    book_summaries = "\n".join([f"- **{book['title']}** by {book['author']} (Genre: {book['genre']})" for book in books])
    history = "\n".join(context_cache)

    prompt = f"""
    You are an expert book recommender. Here is the conversation history:
    
    {history}

    A user asked: "{query}".
    Based on this, here are some book recommendations:

    {book_summaries}

    Now, generate a friendly response explaining why these books were recommended.
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt, stream=True)
        
        response_text = ""
        for chunk in response:
            response_text += chunk.text
            yield chunk.text  # Stream response token-by-token
        
        # Store response in cache
        context_cache.append(f"User: {query}\nBot: {response_text}")
    
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def book_chatbot(query, api_key, genre=None, author=None, top_k=3):
    """
    Handles user queries, retrieves books, and generates a response.
    """
    if not api_key:
        return "❌ Please enter your Google Books API key to use live search!"

    books = search_books(query, genre, author, top_k)
    return generate_book_recommendation_response(query, books, api_key)

with gr.Blocks() as demo:
    gr.Markdown("# 📖 AI Book Recommendation Chatbot (Powered by Google Gemini)")
    
    with gr.Row():
        api_key = gr.Textbox(label="Google API Key", placeholder="Enter your Google API Key here")

    with gr.Row():
        query = gr.Textbox(label="Describe the type of book you want", placeholder="e.g., I love space adventure novels")
    
    with gr.Row():
        genre = gr.Textbox(label="Preferred Genre (Optional)", placeholder="e.g., Sci-Fi")
        author = gr.Textbox(label="Preferred Author (Optional)", placeholder="e.g., Isaac Asimov")

    submit = gr.Button("Get Recommendations 📚")
    output = gr.Markdown()

    submit.click(book_chatbot, inputs=[query, api_key, genre, author], outputs=output)

demo.launch(share=True)
