# book-recommendation-chatbot
📖 AI Book Recommendation Chatbot (Powered by Google Gemini)

🚀 Overview

This AI-powered chatbot provides personalized book recommendations based on user input. It uses:

FAISS vector search for local book suggestions

Google Books API for real-time book retrieval

Google Gemini API for AI-generated explanations

Metadata filtering (genre & author) to refine results

Gradio for an interactive user interface

🔧 How It Works

Enter a book description (e.g., "I love space adventure novels").

(Optional) Filter by genre or author.

The chatbot searches a local book database (FAISS).

If not enough results, it fetches books from Google Books API.

The chatbot uses Google Gemini API to generate a detailed recommendation response.

Streaming response displays results dynamically.

🌍 Live Demo

Try it out here: https://huggingface.co/spaces/Pat32rick/book-recommendation-chatbot

🛠 Technologies Used

FAISS – Fast Approximate Nearest Neighbor Search for book recommendations

Sentence Transformers – Converts book descriptions into embeddings

Google Books API – Fetches real-time book data

Google Gemini API – Generates explanations for recommendations

Gradio – Provides an interactive chatbot UI

🔑 API Key Setup (For Live Search & AI Generation)

To use real-time book search and AI-generated explanations, you need a Google API Key:

Go to Google AI Studio

Click "Create API Key"

Copy your API Key

Enter it when prompted in the chatbot UI

💰 Cost Estimation

FAISS-based local search – Free

Google Books API calls – ~$0.02 per request

Google Gemini API calls – ~$0.005 per 1,000 tokens

Estimated <$0.50 for testing

📂 Project Structure

📂 book-recommendation-chatbot
│── app.py               # Main chatbot script
│── requirements.txt     # Dependencies (FAISS, Gradio, Google APIs, etc.)
│── README.md            # Project documentation

✅ Advanced LLM Techniques Implemented

1️⃣ Hybrid Search – Combines FAISS (local search) with Google Books API2️⃣ Metadata Filtering – Filters books by genre & author3️⃣ Live Search Results – Uses Google Books API for real-time recommendations4️⃣ Streaming Responses – Uses Google Gemini API to stream responses token-by-token5️⃣ Context Caching – Stores recent queries to improve response efficiency

🤖 Deployment on Hugging Face Spaces

Create a Hugging Face Space (Gradio SDK)

Upload app.py & requirements.txt

Restart the space to deploy the chatbot
