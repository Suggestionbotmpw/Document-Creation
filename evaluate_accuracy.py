import json
import subprocess

TEST_FILE = "ok.txt"  # each line = one JSON input
TOTAL = 0
SCORE_SUM = 0

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        TOTAL += 1

        # run inference.py with input piped
        result = subprocess.run(
            ["python", "inference.py"],
            input=line,
            text=True,
            capture_output=True
        )

        # extract accuracy
        for l in result.stdout.splitlines():
            if "Accuracy Score:" in l:
                score = int(l.split(":")[1].replace("%", "").strip())
                SCORE_SUM += score
                break

print("\n===== FINAL ACCURACY =====")
print(f"Samples Tested: {TOTAL}")
print(f"Average Accuracy: {round(SCORE_SUM / TOTAL, 2)}%")
