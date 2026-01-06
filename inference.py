# import torch
# import json
# import re
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
# from config import BASE_MODEL


# # ======================================================
# # Device
# # ======================================================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # ======================================================
# # Tokenizer (ALWAYS from base model)
# # ======================================================
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# tokenizer.pad_token = tokenizer.eos_token


# # ======================================================
# # Load base model (MATCH TRAINING)
# # ======================================================
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# # ======================================================
# # Attach LoRA adapter
# # ======================================================
# model = PeftModel.from_pretrained(base_model, "trained-lora")
# model.eval()


# # ======================================================
# # User Input
# # ======================================================
# def get_user_input():
#     return {
#         "Heading": input("Heading: "),
#         "CustomerNeed": input("Customer Need: "),
#         "Impact": input("Impact (optional): "),
#         "ServicePerformed": input("Service Performed: "),
#         "ServiceSummaryTitle": input("Service Summary Title: "),
#         "ServiceSummaryContent": input("Service Summary Content: "),
#         "Language": input("Language (e.g., en-US): ")
#     }


# print("\n=== DOCUMENT INPUT ===")
# user_input = get_user_input()

# print("\n=== OUTPUT CONTROL ===")
# NUM_RECS = int(input("Number of Recommendations: "))
# NUM_REASONS = int(input("Supporting Reasons per Recommendation: "))
# NUM_EVIDENCE = int(input("Evidence items per Supporting Reason: "))


# # ======================================================
# # Prompt (MATCHES TRAINING)
# # ======================================================
# prompt = f"""### Instruction:
# Generate a professional technical recommendation document.

# ### Constraints (MANDATORY):
# You MUST strictly follow these numeric requirements.
# Violating them is incorrect output.

# - NumRecommendations: {NUM_RECS}
# - NumSupportingReasons: {NUM_REASONS}
# - NumEvidence: {NUM_EVIDENCE}

# ### Input:
# {json.dumps(user_input, indent=2)}

# ### Output (JSON ONLY — no extra text):
# """

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


# # ======================================================
# # Generate (DETERMINISTIC)
# # ======================================================
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=900,
#         temperature=0.0,
#         do_sample=False,
#         repetition_penalty=1.15,
#         eos_token_id=tokenizer.eos_token_id
#     )

# decoded = tokenizer.decode(output[0], skip_special_tokens=True)


# # ======================================================
# # JSON Extraction (ROBUST)
# # ======================================================
# def extract_json(text):
#     match = re.search(r"\{[\s\S]*\}", text)
#     if not match:
#         return None
#     try:
#         return json.loads(match.group())
#     except json.JSONDecodeError:
#         return None


# result_json = extract_json(decoded)


# # ======================================================
# # Validation & Accuracy
# # ======================================================
# def validate_output(data):
#     issues = []

#     if data is None:
#         return 0, ["Invalid JSON"]

#     try:
#         solutions = data["Solutions"]
#         if len(solutions) != NUM_RECS:
#             issues.append("Recommendation count mismatch")

#         for sol in solutions:
#             reasons = sol.get("SupportingReasons", [])
#             if len(reasons) != NUM_REASONS:
#                 issues.append("SupportingReason count mismatch")

#             for r in reasons:
#                 evidence = r.get("Evidence", [])
#                 if len(evidence) != NUM_EVIDENCE:
#                     issues.append("Evidence count mismatch")

#     except Exception:
#         issues.append("Invalid structure")

#     score = max(100 - (len(issues) * 15), 0)
#     return score, list(set(issues))


# accuracy, issues = validate_output(result_json)


# # ======================================================
# # Output
# # ======================================================
# print("\n===== RAW MODEL OUTPUT =====\n")
# print(decoded)

# print("\n===== PARSED JSON =====\n")
# print(json.dumps(result_json, indent=2, ensure_ascii=False))

# print("\n===== VALIDATION REPORT =====")
# print(f"Accuracy Score: {accuracy}%")

# if issues:
#     print("Issues Detected:")
#     for i in issues:
#         print(f"- {i}")
# else:
#     print("✔ Output is VALID and STRUCTURALLY CORRECT")
  








import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import BASE_MODEL


# ======================================================
# Device
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# Tokenizer (ALWAYS from base model)
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


# ======================================================
# Load base model (MATCH TRAINING)
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
# Attach LoRA adapter
# ======================================================
model = PeftModel.from_pretrained(base_model, "trained-lora")
model.eval()


# ======================================================
# User Input
# ======================================================
def get_user_input():
    return {
        "Heading": input("Heading: "),
        "CustomerNeed": input("Customer Need: "),
        "Impact": input("Impact (optional): "),
        "ServicePerformed": input("Service Performed: "),
        "ServiceSummaryTitle": input("Service Summary Title: "),
        "ServiceSummaryContent": input("Service Summary Content: "),
        "Language": input("Language (e.g., en-US): ")
    }


print("\n=== DOCUMENT INPUT ===")
user_input = get_user_input()

print("\n=== OUTPUT CONTROL ===")
NUM_RECS = int(input("Number of Recommendations: "))
NUM_REASONS = int(input("Supporting Reasons per Recommendation: "))
NUM_EVIDENCE = int(input("Evidence items per Supporting Reason: "))


# ======================================================
# Prompt (MATCHES TRAINING)
# ======================================================
prompt = f"""### Instruction:
Generate a professional technical recommendation document.

### Constraints (MANDATORY):
You MUST strictly follow these numeric requirements.
Violating them is incorrect output.

- NumRecommendations: {NUM_RECS}
- NumSupportingReasons: {NUM_REASONS}
- NumEvidence: {NUM_EVIDENCE}

### Input:
{json.dumps(user_input, indent=2)}

### Output (JSON ONLY — no extra text):
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


# ======================================================
# Generate (DETERMINISTIC & SAFE)
# ======================================================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)

# ======================================================
# HARD STOP + CLEANUP
# ======================================================
if "<END_JSON>" in decoded:
    decoded = decoded.split("<END_JSON>")[0]

# 🔹 REMOVE EXTRA NEWLINES / SPACES (FIX FOR BLANK LINES)
decoded = decoded.strip()


# ======================================================
# Extract ONLY the OUTPUT JSON (after marker)
# ======================================================
def extract_output_json(text):
    marker = "### Output (JSON ONLY — no extra text):"
    idx = text.find(marker)

    if idx == -1:
        return None

    substring = text[idx + len(marker):]

    stack = []
    start = None

    for i, ch in enumerate(substring):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:
                        return json.loads(substring[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


# ======================================================
# Enforce STRICT numeric constraints
# ======================================================
def enforce_strict_counts(data, num_recs, num_reasons, num_evidence):
    if not data or "Solutions" not in data:
        return data

    data["Solutions"] = data["Solutions"][:num_recs]

    for sol in data["Solutions"]:
        reasons = sol.get("SupportingReasons", [])
        sol["SupportingReasons"] = reasons[:num_reasons]

        for r in sol["SupportingReasons"]:
            evidence = r.get("Evidence", [])
            r["Evidence"] = evidence[:num_evidence]

    return data


raw_json = extract_output_json(decoded)

result_json = enforce_strict_counts(
    raw_json,
    NUM_RECS,
    NUM_REASONS,
    NUM_EVIDENCE
)


# ======================================================
# Validation
# ======================================================
def validate_output(data):
    issues = []

    if data is None:
        return 0, ["Invalid JSON"]

    try:
        solutions = data.get("Solutions", [])
        if len(solutions) != NUM_RECS:
            issues.append("Recommendation count mismatch")

        for sol in solutions:
            reasons = sol.get("SupportingReasons", [])
            if len(reasons) != NUM_REASONS:
                issues.append("SupportingReason count mismatch")

            for r in reasons:
                evidence = r.get("Evidence", [])
                if len(evidence) != NUM_EVIDENCE:
                    issues.append("Evidence count mismatch")

    except Exception:
        issues.append("Invalid structure")

    score = max(100 - (len(set(issues)) * 15), 0)
    return score, list(set(issues))


accuracy, issues = validate_output(result_json)


# ======================================================
# Output
# ======================================================
print("\n===== PARSED & ENFORCED JSON =====\n")
print(json.dumps(result_json, indent=2, ensure_ascii=False))

print("\n===== VALIDATION REPORT =====")
print(f"Accuracy Score: {accuracy}%")

if issues:
    print("Issues Detected:")
    for i in issues:
        print(f"- {i}")
else:
    print("✔ Output is VALID and STRUCTURALLY CORRECT")

