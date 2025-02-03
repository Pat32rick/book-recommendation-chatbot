import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import requests

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

def fetch_books_from_google(query, api_key, max_results=5):
    """ Fetches books from Google Books API based on a search query. """
    if not api_key:
        return []
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        books = []
        for item in data.get("items", []):
            volume_info = item.get("volumeInfo", {})
            books.append({
                "title": volume_info.get("title", "Unknown Title"),
                "author": ", ".join(volume_info.get("authors", ["Unknown Author"])),
                "genre": volume_info.get("categories", ["Unknown Genre"])[0] if "categories" in volume_info else "Unknown",
                "description": volume_info.get("description", "No description available.")
            })
        return books
    return []

def get_recommendations(query, api_key, genre=None, author=None, top_k=3):
    """ First searches locally using FAISS, then fetches from Google Books API if needed. """
    local_results = search_books(query, genre, author, top_k)
    if len(local_results) < top_k:
        extra_needed = top_k - len(local_results)
        google_books = fetch_books_from_google(query, api_key, max_results=extra_needed)
        local_results.extend(google_books)
    return local_results

def book_chatbot(query, api_key, genre=None, author=None, top_k=3):
    """ Handles user queries and returns book recommendations. """
    if not api_key:
        return "❌ Please enter your Google Books API key to use live search!"
    results = get_recommendations(query, api_key, genre, author, top_k)
    if not results:
        return "Sorry, I couldn't find any matching books. Try another query!"
    response = "**📚 Recommended Books:**\n\n"
    for book in results:
        response += f"🔹 **{book['title']}** by {book['author']} *(Genre: {book['genre']})*\n"
        response += f"📝 {book['description'][:250]}...\n\n"
    return response

with gr.Blocks() as demo:
    gr.Markdown("# 📖 AI Book Recommendation Chatbot")
    with gr.Row():
        api_key = gr.Textbox(label="Google Books API Key", placeholder="Enter your API Key here")
    with gr.Row():
        query = gr.Textbox(label="Describe the type of book you want", placeholder="e.g., I love space adventure novels")
    with gr.Row():
        genre = gr.Textbox(label="Preferred Genre (Optional)", placeholder="e.g., Sci-Fi")
        author = gr.Textbox(label="Preferred Author (Optional)", placeholder="e.g., Isaac Asimov")
    submit = gr.Button("Get Recommendations 📚")
    output = gr.Markdown()
    submit.click(book_chatbot, inputs=[query, api_key, genre, author], outputs=output)

demo.launch(share=True)
