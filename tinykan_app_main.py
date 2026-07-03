# ============================================================
# MAIN — Multi-Agent KAN System (Academic Interface)
# ============================================================
import os, json
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from evaluator import KANEvaluator
from teacher import GeminiTeacher
from profiler import LearnerProfiler

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'tinykan_cosine.pth')
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY', '')
PROFILE_PATH = os.path.join(BASE_DIR, 'profiles.json')

print("Initializing agents...")
evaluator = KANEvaluator(MODEL_PATH)
teacher   = GeminiTeacher(GEMINI_KEY) if GEMINI_KEY else None
profiler  = LearnerProfiler(PROFILE_PATH)
print("✅ All agents initialized")

current_exercise = {}

ACADEMIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

body, .gradio-container {
    font-family: 'Source Sans 3', 'Georgia', serif !important;
    background: #f8f7f4 !important;
}

h1, h2, h3 {
    font-family: 'Crimson Text', 'Times New Roman', serif !important;
    color: #1a1a2e !important;
}

.gr-button-primary {
    background: #1a1a2e !important;
    border: none !important;
    font-family: 'Source Sans 3', sans-serif !important;
    letter-spacing: 0.5px !important;
}

.gr-button-secondary {
    background: transparent !important;
    border: 1px solid #1a1a2e !important;
    color: #1a1a2e !important;
}

table {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 13px !important;
}
"""

def plot_xai(result):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#fafaf8')
    fig.suptitle(
        f'Explainability Dashboard — TinyKAN-Distilled  |  Score: {result.score_pct:.1f}%  |  CEFR: {result.cefr}',
        fontsize=12, fontweight='bold', color='#1a1a2e', y=1.02, fontfamily='serif')

    cefr_colors = {'C2':'#1a5276','C1':'#1f618d','B2':'#2874a6',
                   'B1':'#7d6608','A2':'#922b21','A1':'#7b241c'}
    color = cefr_colors.get(result.cefr, '#555')
    feat_names = ['cos_MiniLM','cos_MPNet','cos_RoBERTa',
                  '|M−MP|','|M−R|','|MP−R|','mean','max','min']

    # Plot 1: Feature Importance
    ax1 = axes[0]
    feat_colors = ['#1a5276' if i<3 else '#7d6608' if i<6 else '#922b21' for i in range(9)]
    bars = ax1.barh(feat_names, result.importance*100, color=feat_colors,
                    edgecolor='white', linewidth=0.8, height=0.6)
    ax1.set_xlabel('Importance (%)', fontsize=10, color='#333')
    ax1.set_title('Feature Importance (XAI)', fontsize=11, fontweight='bold',
                  color='#1a1a2e', pad=10)
    ax1.axvline(x=100/9, color='#999', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.tick_params(labelsize=8, colors='#555')
    for spine in ['top','right']: ax1.spines[spine].set_visible(False)
    for spine in ['left','bottom']: ax1.spines[spine].set_color('#ccc')
    for bar, val in zip(bars, result.importance):
        ax1.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                 f'{val*100:.1f}%', va='center', fontsize=7.5, color='#444')
    patches = [mpatches.Patch(color='#1a5276',label='Raw cosine'),
               mpatches.Patch(color='#7d6608',label='Differences'),
               mpatches.Patch(color='#922b21',label='Statistics')]
    ax1.legend(handles=patches, fontsize=7.5, framealpha=0.8, edgecolor='#ddd')
    ax1.set_facecolor('#fafaf8')

    # Plot 2: Gate weights
    ax2 = axes[1]
    wedges, _, autotexts = ax2.pie(
        [result.gate_kan*100, result.gate_linear*100],
        labels=['KAN Branch\n(non-linear)', 'Linear Branch\n(bypass)'],
        colors=['#1a5276','#7d6608'],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize':9,'color':'#333'},
        wedgeprops={'edgecolor':'white','linewidth':2})
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
    ax2.set_title('Hybrid Architecture\nKAN vs Linear Gate', fontsize=11,
                  fontweight='bold', color='#1a1a2e', pad=10)

    # Plot 3: CEFR positioning
    ax3 = axes[2]
    levels = ['A1','A2','B1','B2','C1','C2']
    thresholds = [0, 0.40, 0.55, 0.70, 0.80, 0.90, 1.0]
    for i, (level, lc) in enumerate(zip(levels, [cefr_colors[l] for l in levels])):
        ax3.barh(0, thresholds[i+1]-thresholds[i], left=thresholds[i],
                 height=0.35, color=lc, alpha=0.85, edgecolor='white')
        ax3.text((thresholds[i]+thresholds[i+1])/2, 0, level,
                 ha='center', va='center', fontsize=9.5,
                 fontweight='bold', color='white')
    ax3.axvline(x=result.score, color='#1a1a2e', linewidth=2.5, zorder=5)
    ax3.plot(result.score, 0, 'v', color='#1a1a2e', markersize=11, zorder=6)
    ax3.text(result.score, 0.25, f'{result.score_pct:.1f}%\n{result.cefr}',
             ha='center', fontsize=10, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.5))
    ax3.set_xlim(0,1); ax3.set_ylim(-0.5,0.6)
    ax3.set_xlabel('Semantic Similarity Score', fontsize=10, color='#333')
    ax3.set_title('CEFR Level Positioning', fontsize=11,
                  fontweight='bold', color='#1a1a2e', pad=10)
    ax3.set_yticks([]); ax3.set_facecolor('#fafaf8')
    for spine in ['left','top','right']: ax3.spines[spine].set_visible(False)
    ax3.spines['bottom'].set_color('#ccc')
    ax3.tick_params(colors='#555', labelsize=8)

    plt.tight_layout()
    path = '/tmp/xai_plot.png'
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#fafaf8')
    plt.close()
    return path

def start_session(name, cefr, ex_type):
    global current_exercise
    if not name.strip():
        return "Please enter your name.", "", "", "", None

    profile = profiler.get(name.strip(), cefr)
    # Always use selected CEFR
    profile.current_cefr = cefr

    chosen_type = None if ex_type == 'auto' else ex_type
    exercise = teacher.get_exercise(cefr, chosen_type or 'auto') if teacher else None

    if not exercise:
        from teacher import Exercise
        exercise = Exercise('translation',
            'Translate: "It is important to protect the environment."',
            'It is important to protect the environment.',
            cefr, 'General', ['important','protect','environment'])

    current_exercise = {'exercise': exercise, 'name': name.strip(), 'cefr': cefr}

    profile_md = (f"**{name}** | Level: **{cefr}** | "
                 f"Sessions: {profile.total_exercises} | "
                 f"Avg. score: {profile.avg_score*100:.1f}%")

    exercise_md = (f"### {exercise.type.replace('_',' ').title()} — Level {cefr}\n\n"
                  f"{exercise.instruction}")

    hints_md = f"*Hints: {', '.join(exercise.hints)}*" if exercise.hints else ""

    return profile_md, exercise_md, hints_md, "", None

def evaluate_answer(learner_answer):
    global current_exercise
    if not current_exercise:
        return "Please start a session first.", "", "", None
    if not learner_answer.strip():
        return "Please enter your answer.", "", "", None

    exercise = current_exercise['exercise']
    name = current_exercise['name']
    cefr = current_exercise['cefr']

    result = evaluator.evaluate(exercise.reference, learner_answer)

    if teacher:
        feedback = teacher.get_feedback(exercise, learner_answer, result.score, result.xai_explanation)
        feedback_md = f"""
**Correction:** {feedback.correction}

**Explanation:** {feedback.explanation}

**Suggestion:** {feedback.improvement}

**Encouragement:** {feedback.encouragement}

**Next focus:** {feedback.next_focus}
"""
    else:
        feedback_md = result.feedback

    profile = profiler.update(name, result.score, exercise.type, cefr, learner_answer)

    score_md = f"""
| Metric | Value |
|--------|-------|
| Semantic score | **{result.score_pct:.1f}%** |
| CEFR level | **{result.cefr}** |
| Confidence | {result.confidence*100:.1f}% |
| KAN branch | {result.gate_kan*100:.1f}% |
| Linear branch | {result.gate_linear*100:.1f}% |

*{result.feedback}*

*{result.xai_explanation}*
"""
    profile_md = (f"**{name}** | Level: **{profile.current_cefr}** | "
                 f"Sessions: {profile.total_exercises} | "
                 f"Avg. score: {profile.avg_score*100:.1f}%")

    return score_md, feedback_md, profile_md, plot_xai(result)

def show_report(name):
    if not name.strip(): return "Please enter your name."
    r = profiler.report(name.strip())
    if 'message' in r: return r['message']
    return f"""
## Progress Report — {r['name']}

| Indicator | Value |
|-----------|-------|
| Current CEFR | **{r['cefr']}** |
| Sessions completed | {r['total']} |
| Average score | **{r['avg']}** |
| Best score | {r['best']} |
| Progression | {r['progression']} |

**Recent scores:** {' → '.join(r['recent'])}
"""

# ── Gradio Interface ──
with gr.Blocks(css=ACADEMIC_CSS, title="Multi-Agent KAN System") as app:

    gr.HTML("""
    <div style="background:#1a1a2e;padding:28px 36px;border-radius:8px;margin-bottom:20px;">
      <h1 style="color:white;font-family:'Georgia',serif;font-size:22px;margin:0 0 6px 0;
                 font-weight:600;letter-spacing:0.3px;">
        Multi-Agent KAN System for Adaptive English Learning
      </h1>
      <p style="color:#a8b4c8;font-size:13px;margin:0 0 12px 0;font-style:italic;">
        Ensemble Knowledge Distillation into Kolmogorov–Arnold Networks (TinyKAN-Distilled)
        · Gemini 2.5 Flash · Adaptive Planner
      </p>
      <p style="color:#7a8a9e;font-size:11px;margin:0;font-style:italic;">
        Thesis: Hybrid KAN-Ensemble Architecture with Knowledge Distillation
        for Optimization and Explainability of Large Language Models
        · University of Kairouan, Tunisia · 2026
      </p>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.1);color:#c8d4e4;padding:3px 10px;
                     border-radius:4px;font-size:11px;font-family:monospace;">
          TinyKAN-Distilled · 32,080 params · r = 0.8791
        </span>
        <span style="background:rgba(255,255,255,0.1);color:#c8d4e4;padding:3px 10px;
                     border-radius:4px;font-size:11px;font-family:monospace;">
          9D cosine-similarity pipeline
        </span>
        <span style="background:rgba(255,255,255,0.1);color:#c8d4e4;padding:3px 10px;
                     border-radius:4px;font-size:11px;font-family:monospace;">
          XAI · CEFR A1–C2
        </span>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Learner Profile")
            name_input = gr.Textbox(label="Full name", placeholder="e.g. Mouna Khlifi")
            cefr_input = gr.Dropdown(
                choices=['A1','A2','B1','B2','C1','C2'],
                value='B1', label="CEFR Proficiency Level")
            type_input = gr.Dropdown(
                choices=[
                    ('Translation (FR → EN)', 'translation'),
                    ('Error Correction', 'correction'),
                    ('Fill in the Blank', 'fill_blank'),
                    ('Automatic (Planner)', 'auto'),
                ],
                value='translation', label="Exercise Type")
            btn_start  = gr.Button("Start Session", variant="primary")
            btn_report = gr.Button("View Progress Report", variant="secondary")
            profile_md = gr.Markdown("*Start a session to view your profile.*")

        with gr.Column(scale=2):
            gr.Markdown("#### Exercise")
            exercise_md = gr.Markdown("*Your exercise will appear here after clicking Start Session.*")
            hints_md    = gr.Markdown("")
            planner_md  = gr.Markdown("")
            answer_box  = gr.Textbox(label="Your answer in English",
                                     placeholder="Write your answer here...", lines=3)
            with gr.Row():
                btn_eval = gr.Button("Submit Answer", variant="primary")
                btn_next = gr.Button("Next Exercise", variant="secondary")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Semantic Evaluation — TinyKAN-Distilled")
            score_md    = gr.Markdown("*Submit an answer to view results.*")
            gr.Markdown("#### Pedagogical Feedback — Gemini 2.5 Flash")
            feedback_md = gr.Markdown("*Feedback will appear here.*")

        with gr.Column(scale=1):
            gr.Markdown("#### Explainability Dashboard (XAI)")
            xai_plot = gr.Image(label="XAI Visualization", type="filepath")

    report_md = gr.Markdown("")

    gr.HTML("""
    <div style="margin-top:20px;padding:16px 20px;background:#f0ede8;border-radius:6px;
                border-left:3px solid #1a1a2e;">
      <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:12px;color:#444;">
        <span><strong style="color:#1a1a2e;">Agent 1 — Evaluator:</strong>
          TinyKAN-Distilled · 9D cosine pipeline · XAI by perturbation</span>
        <span><strong style="color:#1a1a2e;">Agent 2 — Teacher:</strong>
          Gemini 2.5 Flash · Adaptive CEFR exercises</span>
        <span><strong style="color:#1a1a2e;">Agent 3 — Planner:</strong>
          Learner profile · Automatic curriculum adaptation</span>
      </div>
    </div>
    """)

    btn_start.click(fn=start_session,
                    inputs=[name_input, cefr_input, type_input],
                    outputs=[profile_md, exercise_md, hints_md, planner_md, xai_plot])

    btn_eval.click(fn=evaluate_answer,
                   inputs=[answer_box],
                   outputs=[score_md, feedback_md, profile_md, xai_plot])

    btn_next.click(fn=lambda n,c,t: start_session(
                       current_exercise.get('name',n), c, t),
                   inputs=[name_input, cefr_input, type_input],
                   outputs=[profile_md, exercise_md, hints_md, planner_md, xai_plot])

    btn_report.click(fn=show_report,
                     inputs=[name_input],
                     outputs=[report_md])

if __name__ == '__main__':
    app.launch(
        server_name='0.0.0.0',
        server_port=int(os.environ.get('PORT', 7860)),
        pwa=True
    )
