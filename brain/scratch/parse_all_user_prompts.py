import os
import json

transcript_path = os.path.expanduser("~/.gemini/antigravity/brain/77e69092-eb15-4906-a8ca-3e6ae373558d/.system_generated/logs/transcript.jsonl")

with open(transcript_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            obj = json.loads(line)
            if obj.get("type") == "USER_INPUT":
                content = obj.get("content", "").replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                print(f"\n================ STEP {obj.get('step_index', idx)} ================")
                for l in lines[:6]:
                    print("  ", l[:120])
        except Exception as e:
            pass
