import os
import gradio as gr
from openai import OpenAI
import ask_brain


QUESTION_GUIDANCE_MD = """
### 🔎 Question Guidance (optional)

**Playbook**
- “Give me a step-by-step playbook for ___ including risks + metrics.”

**Checklist**
- “Create an onboarding checklist for ___.”

**Executive**
- “How should I run a QBR with ___?”

**Email Draft**
- “Write a customer email for ___ with 2 follow-ups.”
"""

ANSWER_TEMPLATES = {
    "Playbook (recommended)": """You MUST respond using this structure:

## Direct Answer
- bullets

## CSM Playbook
1.
2.
3.
4.
5.

## Risks to Watch
- bullets

## Metrics
- bullets
""",

    "Bullets Only": """You MUST respond using:

## Answer
- concise bullet points
""",

    "Executive Summary": """You MUST respond using:

## Executive Summary
- insights

## Recommended Actions
- bullets
"""
}

EXAMPLES = [
    "What are the top churn risk factors in SaaS customer success?",
    "Create an onboarding checklist for a SaaS customer",
    "How do I run a QBR with an executive sponsor?",
    "Create a renewal recovery playbook"
]


CUSTOM_CSS = """
.gradio-container { max-width:1200px !important; }

/* cards */
.card{
  border-radius:16px;
  border:1px solid rgba(0,0,0,.08);
  box-shadow:0 8px 20px rgba(0,0,0,.05);
  padding:14px;
  background:white;
}

/* spacing between cards */
.stack{
 display:flex;
 flex-direction:column;
 gap:14px;
}

/* headings */
.section-title{
 font-weight:800;
 font-size:13px;
 margin-bottom:6px;
}

/* remove dropdown spacing */
.gradio-dropdown{ margin-bottom:0 !important; }

/* FORCE purple gradient buttons */
button.primary{
 background: linear-gradient(135deg,#4f46e5,#6366f1,#8b5cf6) !important;
 color:white !important;
 border:none !important;
 font-weight:800 !important;
 height:46px !important;
 box-shadow:0 8px 20px rgba(79,70,229,.25) !important;
}

button.primary:hover{
 background: linear-gradient(135deg,#4338ca,#6366f1,#7c3aed) !important;
}
"""


def format_sources(contexts):

    if not contexts:
        return "_No sources returned._"

    blocks=[]

    for i,c in enumerate(contexts,1):

        src=c.get("source","unknown")
        snippet=c.get("text","").replace("\n"," ")

        if len(snippet)>200:
            snippet=snippet[:200]+"..."

        blocks.append(f"**{i}. {src}**\n\n> {snippet}")

    return "\n\n".join(blocks)


def run_brain(question,answer_format,top_k):

    if not question:
        return "Please enter a question.",""

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set",""

    template=ANSWER_TEMPLATES.get(answer_format,"")

    prompt=f"""{template}

CRITICAL:
Answer the user's question directly first.

Question:
{question}
"""

    client=OpenAI()

    chunks=ask_brain.load_chunks()
    cache=ask_brain.load_embed_cache()

    contexts=ask_brain.retrieve_top_k(
        client=client,
        query=question,
        chunks=chunks,
        cache=cache,
        k=int(top_k)
    )

    answer=ask_brain.answer_with_citations(
        client=client,
        query=prompt,
        contexts=contexts
    )

    return answer,format_sources(contexts)


with gr.Blocks(css=CUSTOM_CSS,theme=gr.themes.Glass()) as demo:

    if os.path.exists("banner1.png"):
        gr.Image("banner1.png",show_label=False,height=170)

    gr.Markdown("# 🧠 CSM Brain")
    gr.Markdown("AI assistant for your Customer Success knowledge base")

    with gr.Row():

        with gr.Column(scale=5,elem_classes=["stack"]):

            with gr.Group(elem_classes=["card"]):

                gr.HTML('<div class="section-title">Question</div>')

                question=gr.Textbox(
                    placeholder="Example: What churn risks should I watch for?",
                    lines=4,
                    show_label=False
                )

                ask_btn=gr.Button(
                    "Ask the CSM Brain",
                    variant="primary"
                )

            with gr.Group(elem_classes=["card"]):

                gr.HTML('<div class="section-title">Answer format</div>')

                answer_format=gr.Dropdown(
                    choices=list(ANSWER_TEMPLATES.keys()),
                    value="Playbook (recommended)",
                    show_label=False
                )

            with gr.Group(elem_classes=["card"]):

                gr.HTML('<div class="section-title">Example question</div>')

                with gr.Row():

                    example=gr.Dropdown(
                        choices=EXAMPLES,
                        value=EXAMPLES[0],
                        show_label=False
                    )

                    load_btn=gr.Button(
                        "Load",
                        variant="primary"
                    )

            with gr.Group(elem_classes=["card"]):

                gr.Markdown(QUESTION_GUIDANCE_MD)

            with gr.Group(elem_classes=["card"]):

                with gr.Accordion("Advanced settings",open=False):

                    top_k=gr.Slider(
                        1,10,value=5,step=1,
                        label="Top K sources"
                    )

        with gr.Column(scale=7,elem_classes=["stack"]):

            with gr.Group(elem_classes=["card"]):

                gr.HTML('<div class="section-title">Answer</div>')

                answer_out=gr.Markdown("Your answer will appear here.")

            with gr.Group(elem_classes=["card"]):

                gr.HTML('<div class="section-title">Sources</div>')

                sources_out=gr.Markdown("Sources will appear here.")


    load_btn.click(fn=lambda x:x,inputs=example,outputs=question)

    ask_btn.click(
        fn=run_brain,
        inputs=[question,answer_format,top_k],
        outputs=[answer_out,sources_out]
    )

demo.launch()
