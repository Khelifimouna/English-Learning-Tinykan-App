# ============================================================
# TEACHER — Gemini adaptive exercises
# ============================================================
import google.genai as genai
from dataclasses import dataclass
from typing import List
import json, re, os, random

@dataclass
class Exercise:
    type: str; instruction: str; reference: str
    cefr_level: str; topic: str; hints: List[str]

@dataclass
class Feedback:
    correction: str; explanation: str
    improvement: str; encouragement: str; next_focus: str

# Simple exercises by level (single sentences)
EXERCISES = {
    'A1': [
        {'type':'translation','instruction':'Traduisez: "Le chat dort sur le canapé."','reference':'The cat sleeps on the sofa.','hints':['cat','sleep','sofa']},
        {'type':'translation','instruction':'Traduisez: "Il fait beau aujourd\'hui."','reference':'The weather is nice today.','hints':['weather','nice','today']},
        {'type':'translation','instruction':'Traduisez: "J\'ai faim."','reference':'I am hungry.','hints':['hungry']},
    ],
    'A2': [
        {'type':'translation','instruction':'Traduisez: "J\'apprends l\'anglais depuis deux ans."','reference':'I have been learning English for two years.','hints':['have been learning','for two years']},
        {'type':'translation','instruction':'Traduisez: "Pouvez-vous m\'aider, s\'il vous plaît?"','reference':'Could you help me, please?','hints':['could','help','please']},
        {'type':'correction','instruction':'Corrigez: "She don\'t like to go at school."','reference':'She does not like to go to school.','hints':['does not','to school']},
    ],
    'B1': [
        {'type':'translation','instruction':'Traduisez: "Il est important de protéger l\'environnement."','reference':'It is important to protect the environment.','hints':['important','protect','environment']},
        {'type':'translation','instruction':'Traduisez: "La technologie transforme notre façon de communiquer."','reference':'Technology is transforming the way we communicate.','hints':['transforming','way','communicate']},
        {'type':'correction','instruction':'Corrigez: "He have been work here since three years."','reference':'He has been working here for three years.','hints':['has been working','for']},
    ],
    'B2': [
        {'type':'translation','instruction':'Traduisez: "Les énergies renouvelables représentent l\'avenir de notre planète."','reference':'Renewable energies represent the future of our planet.','hints':['renewable','represent','future']},
        {'type':'translation','instruction':'Traduisez: "La recherche scientifique a démontré l\'efficacité de cette méthode."','reference':'Scientific research has demonstrated the effectiveness of this method.','hints':['demonstrated','effectiveness']},
        {'type':'correction','instruction':'Corrigez: "If I would have more time, I will study more."','reference':'If I had more time, I would study more.','hints':['had','would study']},
    ],
    'C1': [
        {'type':'translation','instruction':'Traduisez: "Il est impératif que nous prenions des mesures immédiates face aux défis climatiques."','reference':'It is imperative that we take immediate action in the face of climate challenges.','hints':['imperative','immediate action','climate challenges']},
        {'type':'translation','instruction':'Traduisez: "La mondialisation a engendré des transformations socio-économiques sans précédent."','reference':'Globalization has brought about unprecedented socio-economic transformations.','hints':['brought about','unprecedented','socio-economic']},
    ],
    'C2': [
        {'type':'translation','instruction':'Traduisez: "L\'essor de l\'intelligence artificielle soulève des questions éthiques fondamentales quant à l\'avenir de l\'humanité."','reference':'The rise of artificial intelligence raises fundamental ethical questions regarding the future of humanity.','hints':['rise','raises','fundamental','ethical','regarding']},
        {'type':'translation','instruction':'Traduisez: "La préservation de la biodiversité constitue un enjeu civilisationnel majeur."','reference':'The preservation of biodiversity constitutes a major civilizational challenge.','hints':['preservation','constitutes','civilizational challenge']},
    ],
}

class GeminiTeacher:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-2.5-flash'
        print("[Teacher] Gemini loaded ✅")

    def get_exercise(self, cefr_level, exercise_type='auto'):
        pool = EXERCISES.get(cefr_level, EXERCISES['B1'])
        if exercise_type != 'auto':
            filtered = [e for e in pool if e['type'] == exercise_type]
            pool = filtered if filtered else pool
        ex = random.choice(pool)
        return Exercise(type=ex['type'], instruction=ex['instruction'],
                       reference=ex['reference'], cefr_level=cefr_level,
                       topic='General', hints=ex['hints'])

    def get_feedback(self, exercise, learner_answer, score, xai_explanation):
        prompt = f"""You are an academic English language instructor assessing a {exercise.cefr_level}-level learner.

Exercise: {exercise.instruction}
Reference answer: {exercise.reference}
Learner response: "{learner_answer}"
Semantic similarity score (TinyKAN): {score*100:.1f}%
XAI analysis: {xai_explanation}

Provide structured bilingual academic feedback. Return ONLY this JSON:
{{
  "correction": "Detailed correction in French (2-3 sentences max)",
  "explanation": "Brief grammar/vocabulary rule in French (1-2 sentences)",
  "improvement": "One concrete strategy in French",
  "encouragement": "Short academic encouragement in French",
  "next_focus": "One linguistic point to focus on next in French"
}}"""
        try:
            response = self.client.models.generate_content(model=self.model_id, contents=prompt)
            text = response.text.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return Feedback(**data)
        except Exception as e:
            print(f"[Teacher] Error: {e}")
        return self._fallback(score)

    def _fallback(self, score):
        if score >= 0.75:
            return Feedback("Bonne réponse.","Structure correcte.","Enrichissez votre vocabulaire.","Continuez ainsi.","Vocabulaire avancé")
        return Feedback("Réponse insuffisante.","Vérifiez la structure.","Relisez la consigne.","Ne vous découragez pas.","Structure de base")
