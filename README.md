# book-recommendation-chatbot
📖 AI Book Recommendation Chatbot

🚀 Overview

This AI-powered chatbot provides personalized book recommendations based on user input. It uses:

FAISS vector search for local book suggestions

Google Books API for real-time book retrieval

Metadata filtering (genre & author) to refine results

Gradio for an interactive user interface

🔧 How It Works

Enter a book description (e.g., "I love space adventure novels").

(Optional) Filter by genre or author.

The chatbot searches a local book database (FAISS).

If not enough results, it fetches books from Google Books API.

Book recommendations are displayed with descriptions and metadata.

🌍 Live Demo

Try it out here: [YOUR_HUGGINGFACE_SPACE_LINK]

🛠 Technologies Used

FAISS – Fast Approximate Nearest Neighbor Search for book recommendations

Sentence Transformers – Converts book descriptions into embeddings

Google Books API – Fetches real-time book data

Gradio – Provides an interactive chatbot UI

🔑 API Key Setup (For Live Search)

To use real-time book search, you need a Google Books API Key:

Go to Google Cloud Console

Enable Google Books API

Generate an API Key

Enter it when prompted in the chatbot UI

💰 Cost Estimation

FAISS-based local search – Free

Google Books API calls – ~$0.02 per request

Estimated <$0.50 for testing

📂 Project Structure

📂 book-recommendation-chatbot
│── app.py               # Main chatbot script
│── requirements.txt     # Dependencies (FAISS, Gradio, etc.)
│── README.md            # Project documentation

🔧 Installation & Dependencies

This project requires the following dependencies, which are listed in requirements.txt:

gradio
faiss-cpu
sentence-transformers
numpy
requests

To install them manually, run:

pip install -r requirements.txt

🤖 Deployment on Hugging Face Spaces

Create a Hugging Face Space (Gradio SDK)

Upload app.py & requirements.txt

Restart the space to deploy the chatbot
