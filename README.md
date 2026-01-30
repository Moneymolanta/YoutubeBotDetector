Other Contributers include Git Users: 
dsamia25 - Aaditya
seunadekunle - Seun Adekunle
Priyanshu-Shekhar - Priyanshu Shekhar
agam-sidhu - Agam Sidhu

CSCI 566: Deep Learning and Its Applications | University of Southern California
A full-stack application that leverages advanced Deep Learning architectures (Hybrid Transformers, Bi-LSTM, CNN) to detect sophisticated bot comments on YouTube in real-time. Unlike traditional filters, this system utilizes a Hybrid Architecture that analyzes both semantic text embeddings and numerical metadata (timestamps, like ratios) to achieve high-precision detection.

Real-Time Analysis: Fetches live comments via YouTube Data API v3.

Hybrid Classification: Fuses text embeddings (DistilBERT) with metadata (Like Count, Reply Count, Publication Time) to detect "smart" bots that mimic human speech.

Ensemble Support: Users can toggle between CNN, RNN, and Transformer models to compare inference results.

Smart Caching: Implements server-side caching (CACHE_TTL) to optimize API quota usage and reduce latency.

Interactive Visualization: Generates dynamic Matplotlib charts (Confidence Distribution & Confusion Matrices) displayed on a Next.js frontend.

Deep Learning (Backend)
Framework: PyTorch, Hugging Face Transformers

Models:

DistilBERT (Custom Hybrid): Modified classification head to accept concatenated metadata vectors.

Bi-LSTM: Captures sequential dependencies in long comments.

CNN (1D): Efficient feature extraction for rapid inference.

API: Flask (Python) with CORS support.

Web Interface (Frontend)
Framework: Next.js (React)

Language: TypeScript

Styling: Tailwind CSS + Shadcn UI

Visualization: Matplotlib (Server-side rendering)

### Demo


<video src="https://github.com/user-attachments/assets/856fcb5d-0e53-4231-b5dc-75ae3e4e47f0" controls></video>
