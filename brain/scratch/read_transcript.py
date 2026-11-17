import os
import json

transcript_path = os.path.expanduser("~/.gemini/antigravity/brain/77e69092-eb15-4906-a8ca-3e6ae373558d/.system_generated/logs/transcript.jsonl")

if os.path.exists(transcript_path):
    print("Transcript exists!")
    with open(transcript_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                if obj.get("type") == "USER_INPUT":
                    content = obj.get("content", "")
                    first_line = content.strip().split("\n")[0]
                    print(f"Step {obj.get('step_index', idx)}: {first_line[:100]} (len={len(content)})")
            except Exception as e:
                pass
else:
    print("Transcript not found at:", transcript_path)
