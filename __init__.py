# core/__init__.py


Quick Start

1. Install UV

pip install uv

Restart PowerShell after installation.

---

2. Open Project

Open the project folder in VS Code or your preferred IDE.

---

3. Install Live Server

Install the Live Server extension from the VS Code Extensions Marketplace.

---

4. Install Dependencies

Open a terminal in the project root directory and run:

uv pip install -r requirements.txt

Wait for all dependencies to finish installing.

---

5. Start Backend

uv run python run.py

Keep this terminal running.

---

6. Start Frontend

Locate:

index.html

Right-click on the file and select:

Open with Live Server

The chatbot UI will open in your browser.

---

First Run

The first run may take significant time because the application may:

- Download the LLM model locally
- Crawl the TomTom documentation website
- Scrape documentation pages
- Process PDF files
- Generate embeddings
- Build the vector database

This process can take several minutes depending on hardware and internet speed.

To begin initialization, send any message in the chat interface after opening the UI.

---

Notes

- Do not close the terminal running "run.py".
- First startup is slower than subsequent startups.
- Model download occurs only if the model is not already present.
- Performance depends heavily on available CPU/GPU resources.
- Running on a GPU-enabled machine is recommended for best results.

---

Current Limitation

The chatbot is fully functional, but response generation speed is currently limited by local inference performance. Deploying on a machine with GPU acceleration is expected to significantly improve response times.