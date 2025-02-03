import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import requests
import google.generativeai as genai
from collections import deque

# Store recent queries and responses
context_cache = deque(maxlen=5)  # Keep the last 5 conversations

# Sample book dataset
books = [
    {
        "title": "Dune",
        "author": "Frank Herbert",
        "genre": "Sci-Fi",
        "description": "A science fiction saga about a desert planet and its people."
    },
    {
        "title": "1984",
        "author": "George Orwell",
        "genre": "Dystopian",
        "description": "A dystopian novel about a totalitarian regime that watches everything."
    },
    {
        "title": "The Hobbit",
        "author": "J.R.R. Tolkien",
        "genre": "Fantasy",
        "description": "A hobbit embarks on a journey to reclaim a lost kingdom."
    }
]

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for the book descriptions
descriptions = [book["description"] for book in books]
embeddings = model.encode(descriptions, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search_books(query, books_api_key, genre=None, author=None, top_k=2):
    """Searches for books using FAISS and Google Books API."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k * 3)
    results = [books[idx] for idx in indices[0]]

    if genre:
        results = [book for book in results if book["genre"].lower() == genre.lower()]
    if author:
        results = [book for book in results if book["author"].lower() == author.lower()]

    if len(results) < top_k:
        extra_needed = top_k - len(results)
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={extra_needed}&key={books_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("items", []):
                volume_info = item.get("volumeInfo", {})
                results.append({
                    "title": volume_info.get("title", "Unknown Title"),
                    "author": ", ".join(volume_info.get("authors", ["Unknown Author"])),
                    "genre": volume_info.get("categories", ["Unknown Genre"])[0] if "categories" in volume_info else "Unknown",
                    "description": volume_info.get("description", "No description available.")
                })
    return results[:top_k]

def generate_book_recommendation_response(query, books, gen_api_key):
    """Uses Google Gemini AI to generate book summaries and personalized recommendations."""
    if not books:
        return "Sorry, I couldn't find any matching books. Try another query!"

    genai.configure(api_key=gen_api_key)

    book_details = []
    for book in books:
        book_prompt = f"Summarize the book **{book['title']}** by {book['author']} in 2 sentences."
        gemini_model = genai.GenerativeModel("gemini-pro")
        summary_response = gemini_model.generate_content(book_prompt)
        summary = summary_response.text if summary_response else "No summary available."
        book_details.append(
            f"- **{book['title']}** by {book['author']} *(Genre: {book['genre']})*\n  📖 {summary}"
        )

    # Precompute the joined string to avoid using a backslash in the f-string expression
    joined_book_details = "\n".join(book_details)

    recommendation_prompt = f"""
A user asked: "{query}".
Based on their request and reading preferences, here are book recommendations:
{joined_book_details}
Now, suggest additional books they might like beyond these, explaining why.
    """

    try:
        gemini_model = genai.GenerativeModel("gemini-pro")
        response = gemini_model.generate_content(recommendation_prompt)
        generated_text = response.text
        context_cache.append(f"User: {query}\nBot: {generated_text}")
        return generated_text
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def book_chatbot(query, books_api_key, gen_api_key, genre=None, author=None, top_k=3):
    """Handles user queries, retrieves books, and generates a response."""
    if not books_api_key or not gen_api_key:
        return "❌ Please enter both Google Books API Key and Google Gemini API Key!"

    retrieved_books = search_books(query, books_api_key, genre, author, top_k)
    return generate_book_recommendation_response(query, retrieved_books, gen_api_key)

with gr.Blocks() as demo:
    gr.Markdown("# 📖 AI Book Recommendation Chatbot (Powered by Google Gemini)")

    with gr.Row():
        books_api_key = gr.Textbox(
            label="📚 Google Books API Key", 
            placeholder="Enter your Google Books API Key"
        )
        gen_api_key = gr.Textbox(
            label="🤖 Google Gemini API Key", 
            placeholder="Enter your Google Gemini API Key"
        )

    with gr.Row():
        query = gr.Textbox(
            label="🔎 Describe the type of book you want", 
            placeholder="e.g., I love space adventure novels"
        )

    with gr.Row():
        genre = gr.Textbox(
            label="📂 Preferred Genre (Optional)", 
            placeholder="e.g., Sci-Fi"
        )
        author = gr.Textbox(
            label="✍️ Preferred Author (Optional)", 
            placeholder="e.g., Isaac Asimov"
        )

    submit = gr.Button("📚 Get Recommendations")
    output = gr.Markdown()

    submit.click(book_chatbot, inputs=[query, books_api_key, gen_api_key, genre, author], outputs=output)

demo.launch(share=True)
