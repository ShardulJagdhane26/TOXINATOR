# TOXINATOR ⚡

Check if comments are rude, harmful, or hateful in seconds.

TOXINATOR uses a BiLSTM deep learning model and Gemini commentary to analyze how toxic a comment is. It returns overall toxicity, per‑label probabilities, and a short AI "hot take" explaining the tone.

---

## 🚀 Live Demo

[https://toxinator.vercel.app](https://toxinator.vercel.app)

---

## 🧠 What It Does

- Multi‑label toxicity classification on 6 categories:
  - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Overall toxicity score + per‑label probabilities
- AI commentary (Gemini) that explains the tone in a short, witty paragraph

### Model Highlights

- **Architecture:** Embedding → BiLSTM → Dense layers (multi‑label sigmoid)
- **Parameters:** ~6.49M
- **Precision on held‑out data:** ~92.8%

---

## Project Structure

```
TOXINATOR/
├─ backend/
│  ├─ app.py            # Flask API + TensorFlow model + Gemini proxy
│  ├─ toxicity.h5       # Trained BiLSTM toxicity model (Keras H5)
│  ├─ vocab.json        # Vocabulary exported from TextVectorization
│  ├─ requirements.txt  # Python runtime + TensorFlow
│  └─ .env.example      # Environment variable template
├─ frontend/
│  └─ index.html        # Single‑page cartoon UI (Tailwind + vanilla JS)
└─ notebook/
   └─ ...               # Training / exploration notebooks
```

---

## 🔧 Backend (Flask + TensorFlow)

### 1. Environment Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run API Locally

```bash
cd backend
python app.py
```

By default it starts on `http://localhost:5000`.

### API Endpoints

- `GET /api/health` – service + model status
- `POST /api/classify` – classify one comment

### Request Body

```json
{
  "comment": "You freaking suck! I am going to hit you."
}
```

### Sample Response

```json
{
  "success": true,
  "comment": "You freaking suck! I am going to hit you.",
  "toxic_score": 0.73,
  "is_toxic": true,
  "percentage": 73.0,
  "message": "Toxic! ⚠️",
  "detailed_scores": [0.81, 0.12, 0.64, 0.03, 0.77, 0.09],
  "categories": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
  "analysis_text": "Short Gemini commentary goes here..."
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | Embedding → BiLSTM → Dense Layers |
| Total Parameters | ~6.49M |
| Precision | ~92.8% |
| Output Type | Multi-label sigmoid |

---

## 📝 Important Notes

- The toxicity model is trained on the **Jigsaw toxic comment dataset**; like all ML models, it can make mistakes and may not generalize to every domain.
- **Please do not rely on it as a single source of truth for moderation decisions in sensitive or high‑stakes environments.**
- Model metrics can be updated if you decide to retrain or optimize the model later.

---