# ============================================================
# EVALUATOR — TinyKAN + XAI
# ============================================================
import torch, torch.nn as nn, numpy as np
from kan import KAN
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List

class TinyKANDistilled(nn.Module):
    def __init__(self, input_dim=9, grid=3):
        super().__init__()
        self.kan_main = KAN(width=[[input_dim,0],[16,0],[8,0],[1,0]], grid=grid)
        self.linear_branch = nn.Sequential(nn.Linear(input_dim,8), nn.GELU(), nn.Linear(8,1))
        self.gate = nn.Sequential(nn.Linear(input_dim,8), nn.ReLU(), nn.Linear(8,2), nn.Softmax(dim=1))
        self.skip = nn.Linear(input_dim, 1)
    def forward(self, x):
        k = self.kan_main(x); l = self.linear_branch(x); g = self.gate(x)
        return (g * torch.cat([k,l],dim=1)).sum(dim=1,keepdim=True) + 0.05*self.skip(x)
    def get_gate_weights(self, x):
        with torch.no_grad(): return self.gate(x).cpu().numpy()

@dataclass
class EvalResult:
    score: float; score_pct: float; cefr: str; confidence: float
    gate_kan: float; gate_linear: float; features: np.ndarray
    importance: np.ndarray; feedback: str; xai_explanation: str

class KANEvaluator:
    FEATURE_NAMES = ['cos_MiniLM','cos_MPNet','cos_RoBERTa',
                     '|MiniLM-MPNet|','|MiniLM-RoBERTa|','|MPNet-RoBERTa|',
                     'mean(cos)','max(cos)','min(cos)']
    CEFR_LEVELS = [(0.90,'C2'),(0.80,'C1'),(0.70,'B2'),(0.55,'B1'),(0.40,'A2'),(0.0,'A1')]
    ENCODERS = {'MiniLM':'sentence-transformers/all-MiniLM-L6-v2',
                'MPNet':'sentence-transformers/all-mpnet-base-v2',
                'RoBERTa':'sentence-transformers/stsb-roberta-base'}

    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TinyKANDistilled(9,3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        print(f"[Evaluator] TinyKAN loaded ✅ ({sum(p.numel() for p in self.model.parameters()):,} params)")
        self.encoders = {n: SentenceTransformer(p, device=self.device) for n,p in self.ENCODERS.items()}
        print("[Evaluator] Encoders loaded ✅")

    def _features(self, s1, s2):
        cos = []
        for enc in self.encoders.values():
            e1 = enc.encode([s1], convert_to_numpy=True)
            e2 = enc.encode([s2], convert_to_numpy=True)
            e1 /= np.maximum(np.linalg.norm(e1,axis=1,keepdims=True),1e-8)
            e2 /= np.maximum(np.linalg.norm(e2,axis=1,keepdims=True),1e-8)
            cos.append(float((e1*e2).sum()))
        c1,c2,c3 = cos
        return np.array([c1,c2,c3,abs(c1-c2),abs(c1-c3),abs(c2-c3),(c1+c2+c3)/3,max(cos),min(cos)],dtype=np.float32)

    def _predict(self, feat):
        x = torch.tensor(feat,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): return float(torch.sigmoid(self.model(x)).cpu().numpy()[0][0])

    def _importance(self, feat):
        base = self._predict(feat)
        imp = np.array([abs(base - self._predict(np.where(np.arange(9)==i, 0, feat))) for i in range(9)])
        return imp / imp.sum() if imp.sum() > 0 else imp

    def _cefr(self, score):
        for t,l in self.CEFR_LEVELS:
            if score >= t: return l
        return 'A1'

    def evaluate(self, reference, learner):
        feat = self._features(reference, learner)
        score = self._predict(feat)
        cefr = self._cefr(score)
        imp = self._importance(feat)
        x = torch.tensor(feat,dtype=torch.float32).unsqueeze(0).to(self.device)
        gates = self.model.get_gate_weights(x)[0]
        top_idx = np.argmax(imp)
        top_feat = self.FEATURE_NAMES[top_idx]

        if score >= 0.90: fb = f"Excellent — Score {score*100:.1f}% | Level {cefr}"
        elif score >= 0.75: fb = f"Good — Score {score*100:.1f}% | Level {cefr}"
        elif score >= 0.55: fb = f"Satisfactory — Score {score*100:.1f}% | Level {cefr}"
        elif score >= 0.40: fb = f"Needs improvement — Score {score*100:.1f}% | Level {cefr}"
        else: fb = f"Insufficient — Score {score*100:.1f}% | Level {cefr}"

        xai = f"Most influential feature: '{top_feat}' ({imp[top_idx]*100:.1f}% importance)"
        if top_idx < 3: xai += f" — Model relies on {['MiniLM','MPNet','RoBERTa'][top_idx]} cosine similarity."
        elif top_idx < 6: xai += " — Encoder divergence detected (semantic difficulty signal)."
        else: xai += f" — {['Mean','Maximum','Minimum'][top_idx-6]} cosine is the decisive statistic."

        return EvalResult(score=score, score_pct=score*100, cefr=cefr,
                         confidence=min(min([abs(score-t) for t,_ in self.CEFR_LEVELS])*2,1.0),
                         gate_kan=float(gates[0]), gate_linear=float(gates[1]),
                         features=feat, importance=imp, feedback=fb, xai_explanation=xai)
