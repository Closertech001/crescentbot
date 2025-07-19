# 🎓 Crescent University Chatbot

An intelligent chatbot built to answer student and visitor questions about Crescent University, Abeokuta — including departments, admissions, courses, and fees.

---

## 🚀 Features

- Natural language Q&A
- GPT fallback when no exact match is found
- Synonym and abbreviation support
- Semantic search using SentenceTransformer + FAISS
- Clean, responsive UI with Streamlit
- Tracks user queries in logs
- Easy deployment on Streamlit Cloud

---

## 🧠 How It Works

1. Loads a curated Q&A dataset from `data/crescent_qa.json`
2. Embeds the questions using `all-MiniLM-L6-v2`
3. Matches incoming queries semantically
4. If no good match is found, uses OpenAI's GPT to generate a fallback answer

---

## 📦 Installation

```bash
git clone https://github.com/your-username/crescent-chatbot.git
cd crescent-chatbot
pip install -r requirements.txt
