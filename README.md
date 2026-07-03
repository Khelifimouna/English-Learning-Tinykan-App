# Multi-Agent KAN System for Adaptive English Learning

Three communicating agents deliver adaptive, explainable English-language
practice aligned to the CEFR scale:

- **Evaluator** — TinyKAN-Distilled (32,080 parameters), a hybrid
  Kolmogorov–Arnold Network distilled from three Sentence-BERT teacher
  encoders, paired with a perturbation-based XAI module.
- **Teacher** — Gemini 2.5 Flash, generating CEFR-aligned exercises and
  structured bilingual feedback.
- **Planner** — a persistent learner-profile tracker that adapts exercise
  difficulty from rolling performance statistics.

## Project structure

```
.
├── main.py            # Gradio interface + agent orchestration
├── evaluator.py        # Evaluator agent (TinyKAN + XAI)
├── teacher.py          # Teacher agent (Gemini 2.5 Flash)
├── profiler.py          # Planner agent (learner profile & adaptation)
├── requirements.txt     # Python dependencies
├── render.yaml           # Render.com deployment blueprint
└── tinykan_cosine.pth    # Trained model weights (add this yourself)
```

## Running locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python main.py
```

The app opens at `http://localhost:7860`.

## Deploying

See the project documentation for the full Render.com deployment guide.
In short: push this repository to GitHub, create a Render Blueprint from
it, and set the `GEMINI_API_KEY` secret in the Render dashboard.

## Required secret

| Name | Where to get it |
|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com |

## Note on model weights

`tinykan_cosine.pth` is not included in this listing (binary file). Place
it at the repository root before deploying — `evaluator.py` loads it via
a path relative to the project root.
