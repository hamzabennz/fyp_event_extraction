# Forensic Event Extraction

A digital forensics pipeline that uses LLMs (Google Gemini) and Stanford's LLooM framework to automatically extract events from raw text evidence and visualize them in an interactive mindmap.

## Setup

1. **Environment Variables:**
   Copy `env.example` to `.env` and add your Google Gemini API key:

   ```bash
   cp env.example .env
   ```

2. **Install Dependencies:**
   Requires **Python 3.10+** and **Node.js 18+**.

   ```bash
   # Python dependencies
   pip install -r requirements.txt
   pip install -r backend_service/requirements.txt

   # Node dependencies (for LLooM clustering)
   cd lloom
   npm install
   cd ..
   ```

## Run the Application

Start the backend server and web UI:

```bash
uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --reload --port 8000
```

Navigate to `http://localhost:8000` in your web browser:

1. Upload your `.txt` evidence files.
2. Wait for the extraction to complete, then click the **Review** link to approve or discard events.
3. Submit your review. The system will finish processing and generate a downloadable `evidence_mindmap.html`.
