import torch
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ======================================================
# CONFIG
# ======================================================

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "./trained-lora"

MAX_NEW_TOKENS = 380
TEMPERATURE = 0.55
TOP_P = 0.9

# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(title="AI Recommendation Generator", version="FINAL-2.0")

# ======================================================
# LOAD TOKENIZER
# ======================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# LOAD BASE MODEL (4-bit)
# ======================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# ======================================================
# ATTACH LoRA (BASE + LoRA TOGETHER)
# ======================================================

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ======================================================
# REQUEST SCHEMA
# ======================================================

class SupportingReasonInput(BaseModel):
    evidenceCount: int

class RecommendationInput(BaseModel):
    supportingReasons: List[SupportingReasonInput]

class GenerateRequest(BaseModel):
    context: dict
    recommendations: List[RecommendationInput]

# ======================================================
# UTILITIES
# ======================================================

def trim_chars(text: str, max_chars: int) -> str:
    return text[:max_chars].rsplit(" ", 1)[0].strip()

def trim_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])

def build_context(context: dict) -> str:
    fields = [
        "Language",
        "Heading",
        "CustomerNeed",
        "Impact",
        "ServicePerformed",
        "ServiceSummaryTitle",
        "ServiceSummaryContent",
    ]
    return "\n".join(
        f"{k}: {context[k].strip()}"
        for k in fields
        if context.get(k) and context[k].strip()
    )

def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.replace(prompt, "").strip()

# ======================================================
# PROMPTS
# ======================================================

def recommendation_prompt(ctx: str) -> str:
    return f"""
Write ONE high-level technical recommendation.

Rules:
- ≤ 100 characters
- Abstract and semantic
- No dates, no locations
- No reports or inspections

Context:
{ctx}

Answer:
""".strip()

def result_prompt(ctx: str, rec: str) -> str:
    return f"""
Write ONE concise outcome sentence.

Rules:
- ≤ 120 characters
- Business or reliability impact
- No explanation

Recommendation:
{rec}

Context:
{ctx}

Answer:
""".strip()

def supporting_reason_prompt(ctx: str, rec: str) -> str:
    return f"""
Write ONE technical justification.

Rules:
- ≤ 100 characters
- Abstract
- No repetition
- No reports

Recommendation:
{rec}

Context:
{ctx}

Answer:
""".strip()

def evidence_title_prompt(reason: str) -> str:
    return f"""
Write ONE short technical evidence title.

Rules:
- ≤ 100 characters
- No punctuation
- No labels

Supporting Reason:
{reason}

Answer:
""".strip()

def evidence_content_prompt(ctx: str, reason: str) -> str:
    return f"""
Write a detailed technical explanation.

Rules:
- ~300 words
- Engineering focused
- No reports
- No conclusions

Supporting Reason:
{reason}

Context:
{ctx}

Answer:
""".strip()

# ======================================================
# API ENDPOINT
# ======================================================

@app.post("/generate")
def generate_document(req: GenerateRequest):

    ctx = build_context(req.context)
    solutions = []

    for rec in req.recommendations:

        recommendation = trim_chars(
            generate(recommendation_prompt(ctx)), 100
        )

        result = trim_chars(
            generate(result_prompt(ctx, recommendation)), 120
        )

        supporting_reasons_output = []

        for sr in rec.supportingReasons:

            reason = trim_chars(
                generate(supporting_reason_prompt(ctx, recommendation)), 100
            )

            evidence_list = []

            for _ in range(sr.evidenceCount):

                title = trim_chars(
                    generate(evidence_title_prompt(reason)), 100
                )

                content = trim_words(
                    generate(evidence_content_prompt(ctx, reason)), 300
                )

                evidence_list.append({
                    "EvidenceTitle": title,
                    "EvidenceContent": content
                })

            supporting_reasons_output.append({
                "SupportingReason": reason,
                "Evidence": evidence_list
            })

        solutions.append({
            "Recommendation": recommendation,
            "Result": result,
            "SupportingReasons": supporting_reasons_output
        })

    return {"Solutions": solutions}
