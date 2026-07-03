# ============================================================
# PROFILER — Learner memory and adaptation
# ============================================================
import json, numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from datetime import datetime

@dataclass
class Session:
    timestamp: str; exercise_type: str; topic: str
    cefr_level: str; score: float; learner_answer: str

@dataclass
class Profile:
    name: str; current_cefr: str = 'B1'
    sessions: List[Session] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    total_exercises: int = 0; avg_score: float = 0.0

class LearnerProfiler:
    CEFR = ['A1','A2','B1','B2','C1','C2']
    PROMOTE = 0.80; DEMOTE = 0.50; MIN_SESSIONS = 3

    def __init__(self, path='/tmp/profiles.json'):
        self.path = path
        self.profiles: Dict[str, Profile] = {}
        self._load()

    def get(self, name, initial_cefr='B1'):
        if name not in self.profiles:
            self.profiles[name] = Profile(name=name, current_cefr=initial_cefr)
        return self.profiles[name]

    def update(self, name, score, exercise_type, cefr_level, learner_answer):
        p = self.get(name)
        # Always use the selected CEFR level
        p.current_cefr = cefr_level
        p.sessions.append(Session(
            timestamp=datetime.now().isoformat(),
            exercise_type=exercise_type, topic='General',
            cefr_level=cefr_level, score=score,
            learner_answer=learner_answer))
        p.score_history.append(score)
        p.total_exercises += 1
        p.avg_score = float(np.mean(p.score_history))
        self._save()
        return p

    def report(self, name):
        p = self.get(name)
        if not p.score_history:
            return {'message': 'No sessions yet'}
        recent = p.score_history[-5:]
        prog = 0.0
        if len(p.score_history) >= 2:
            h = len(p.score_history)//2
            prog = (np.mean(p.score_history[h:]) - np.mean(p.score_history[:h])) * 100
        return {
            'name': p.name, 'cefr': p.current_cefr,
            'total': p.total_exercises,
            'avg': f"{p.avg_score*100:.1f}%",
            'best': f"{max(p.score_history)*100:.1f}%",
            'progression': f"{prog:+.1f}%",
            'recent': [f"{s*100:.1f}%" for s in recent]
        }

    def _save(self):
        data = {}
        for name, p in self.profiles.items():
            data[name] = {
                'name': p.name, 'current_cefr': p.current_cefr,
                'score_history': p.score_history,
                'total_exercises': p.total_exercises,
                'avg_score': p.avg_score,
                'sessions': [asdict(s) for s in p.sessions[-50:]]
            }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            for name, d in data.items():
                sessions = [Session(**s) for s in d.pop('sessions', [])]
                p = Profile(**{k: v for k, v in d.items()})
                p.sessions = sessions
                self.profiles[name] = p
            print(f"[Profiler] {len(self.profiles)} profiles loaded ✅")
        except:
            print("[Profiler] Fresh start")
