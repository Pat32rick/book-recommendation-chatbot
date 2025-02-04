import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import requests
import google.generativeai as genai
from collections import deque

def generate_book_recommendation_response(query, books, gen_api_key):
    """Uses Google Gemini AI to generate book summaries and personalized recommendations with streaming."""
    if not books:
        yield "Sorry, I couldn't find any matching books. Try another query!"
        return

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
    
    joined_book_details = "\n".join(book_details)
    recommendation_prompt = f"""
    A user asked: "{query}".
    Based on their request and reading preferences, here are book recommendations:
    {joined_book_details}
    Now, suggest additional books they might like beyond these, explaining why.
    """
    
    try:
        gemini_model = genai.GenerativeModel("gemini-pro")
        response = gemini_model.generate_content_stream(recommendation_prompt)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text  # Stream the text chunks as they arrive
    except Exception as e:
        yield f"Error generating recommendations: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 📖 AI Book Recommendation Chatbot (Streaming Enabled)")
    
    with gr.Row():
        books_api_key = gr.Textbox(label="📚 Google Books API Key", placeholder="Enter your Google Books API Key")
        gen_api_key = gr.Textbox(label="🤖 Google Gemini API Key", placeholder="Enter your Google Gemini API Key")
    
    with gr.Row():
        query = gr.Textbox(label="🔎 Describe the type of book you want", placeholder="e.g., I love space adventure novels")
    
    chatbot = gr.Chatbot()
    submit = gr.Button("📚 Get Recommendations")
    
    def chat_response(query, books_api_key, gen_api_key):
        retrieved_books = search_books(query, books_api_key)  # Ensure this function exists
        return generate_book_recommendation_response(query, retrieved_books, gen_api_key)
    
    submit.click(chat_response, inputs=[query, books_api_key, gen_api_key], outputs=chatbot)

demo.launch(share=True)
