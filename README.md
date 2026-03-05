# 🧠 CSM Brain RAG

AI assistant designed for Customer Success Managers (CSMs).  
This tool uses **Retrieval Augmented Generation (RAG)** to answer questions using a local knowledge base of Customer Success resources.

It helps CSMs quickly generate:

• churn risk analysis  
• onboarding checklists  
• customer success playbooks  
• executive-level insights  
• QBR preparation guidance  

---

# 🚀 Features

• AI-powered Customer Success assistant  
• Retrieval Augmented Generation (RAG) using local documents  
• Clean Gradio web interface  
• Structured answer formats:
  - Playbook
  - Checklist
  - Executive summary
  - Bullet summary
• Adjustable document retrieval (Top-K sources)  
• Source citations for transparency  

---

# 🧠 Example Use Cases

Customer Success teams can ask questions like:

• "What churn risks should I watch for in a SaaS account?"  
• "Create an onboarding checklist for a new SaaS customer."  
• "How should I run a QBR with a CIO?"  
• "What signals indicate declining product adoption?"  

The assistant retrieves relevant knowledge from uploaded documents and generates structured responses.

---

# 🏗 Architecture

This project implements a lightweight **RAG pipeline**.

User Question  
↓  
Embedding Generation  
↓  
Similarity Search  
↓  
Top-K Document Retrieval  
↓  
LLM Response Generation  
↓  
Structured Answer + Sources

Core components:

• **OpenAI embeddings** for document search  
• **Local vector similarity retrieval**  
• **Gradio UI** for interaction  
• **Chunked document knowledge base**

---

# 📂 Project Structure

csm-brain-rag
│
├── app_gradio.py # Gradio UI
├── ingest_pdfs.py # Embedding + document ingestion
├── README.md
├── .gitignore
│
├── data/
│ └── .gitkeep

# ⚙️ Setup Instructions

Follow these steps to run the project locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/csm-brain-rag.git
cd csm-brain-rag
```

---

## 2. Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install gradio openai
```

---

## 4. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can get a key from:

https://platform.openai.com/api-keys

---

## 5. Ingest Documents into the Knowledge Base

This step converts documents into embeddings for retrieval.

```bash
python ingest_pdfs.py
```

---

## 6. Launch the Application

```bash
python app_gradio.py
```

---

## 7. Open the App

Open your browser and go to:

```
http://127.0.0.1:7860
```

You can now start asking questions using the **CSM Brain interface**.

---

# Example Questions

Try prompts like:

• What churn risks should I watch for in a SaaS account?  
• Create a SaaS onboarding checklist.  
• What signals indicate declining product adoption?  
• How should I prepare for a QBR with executive stakeholders?
