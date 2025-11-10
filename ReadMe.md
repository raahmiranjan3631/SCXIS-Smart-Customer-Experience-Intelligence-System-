# 🤖 SCXIS – Smart Customer Experience Intelligence System

## 🧩 Overview
**SCXIS (Smart Customer Experience Intelligence System)** is an AI-powered customer support and analytics platform that goes beyond simple chatbot interaction.  
It intelligently analyzes **customer emotions, intent, and satisfaction levels** from text conversations and provides **real-time insights** through an integrated analytics dashboard.

This system enables businesses to **understand their customers better**, identify pain points early, and improve overall experience through data-driven decisions.

---

## ✨ Key Features
- 💬 **Live Chat System** – Real-time user interaction through an intuitive Streamlit interface.  
- 🧠 **Emotion & Intent Analysis** – Detects customer intent (complaint, feedback, query) and emotions (joy, anger, sadness).  
- 📊 **Sentiment Analysis** – Evaluates the tone of each conversation as positive, negative, or neutral.  
- 💡 **Customer Satisfaction (CSAT) Tracking** – Allows users to rate their experience, providing quantifiable feedback.  
- 📈 **Insights Dashboard** – Visual analytics for sentiment trends, intent distribution, emotion ratios, and CSAT trends.  
- 📜 **Chat History Export** – Download complete chat logs as CSV for further analysis.  
- 🔒 **Lightweight & Secure** – Uses CPU-friendly transformer models for fast inference and privacy-safe processing.

---

## 🧠 AI Models Used
| Task | Model | Base Architecture | Source |
|------|--------|------------------|---------|
| **Sentiment Analysis** | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | Hugging Face |
| **Intent Detection** | `valhalla/distilbart-mnli-12-1` | DistilBART | Hugging Face |
| **Emotion Detection** | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa | Hugging Face |

All models are pre-trained and CPU-optimized for deployment on platforms like Streamlit Cloud or Render.

---

## ⚙️ Tech Stack
- **Frontend & Deployment:** Streamlit  
- **AI/ML Framework:** Hugging Face Transformers  
- **Language:** Python (v3.10+)  
- **Visualization:** Pandas, Streamlit Charts  
- **Environment:** CPU-only (no GPU required)

---

## 🧾 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/SCXIS.git
cd SCXIS
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

### 4️⃣ Access the App

Open your browser and navigate to:

```
http://localhost:8501
```

---

## 🧩 Folder Structure

```
SCXIS/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for Streamlit Cloud
├── README.md              # Documentation file
└── data/ (optional)       # For storing exported chat data
```

---

## 📊 System Workflow

1. **User Message Input** → User types a query or feedback.
2. **NLP Pipeline** → Message analyzed for sentiment, intent, and emotion.
3. **AI Response Generation** → Bot replies contextually.
4. **CSAT Collection** → User provides satisfaction rating (1–5).
5. **Analytics Dashboard** → Insights visualized (trends, patterns, and CSAT).

---

## 🧭 Project Roadmap (Phase-Wise)

| Phase                       | Description                                   |
| --------------------------- | --------------------------------------------- |
| **1. Ideation & Planning**  | Define problem statement and AI scope.        |
| **2. Core Development**     | Build chat interface and query handling.      |
| **3. AI Integration**       | Add emotion, intent, and satisfaction models. |
| **4. Advanced Features**    | Add trend analysis and insights dashboard.    |
| **5. Testing & Deployment** | Refine UI, test models, and deploy app.       |

---

## 🧑‍💻 Contributors

* **[Your Name]** – Developer & Project Lead
* **[Teammate Name]** – Co-developer / Model Integrator

Organized under **Product Space AI Agent Hackathon 2025** 🧠

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and build upon SCXIS with proper attribution.

---

## 💬 Contact

For queries or collaboration, contact:
📧 [rashmiranjansutar8@gmail.com](mailto:rashmiranjansutar8@gmail.com)
🔗 [LinkedIn Profile](https://www.linkedin.com/in/rashmi-ranjan-sutar-1a66232a4)

---
