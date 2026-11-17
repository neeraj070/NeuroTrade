import os
import json

transcript_path = os.path.expanduser("~/.gemini/antigravity/brain/77e69092-eb15-4906-a8ca-3e6ae373558d/.system_generated/logs/transcript.jsonl")

steps_to_dump = [0, 134, 280, 350, 478, 600, 658, 685, 750]

with open(transcript_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            obj = json.loads(line)
            s_idx = obj.get("step_index", idx)
            if s_idx in steps_to_dump and obj.get("type") == "USER_INPUT":
                print(f"\n################################################################")
                print(f"USER REQUEST AT STEP {s_idx}")
                print(f"################################################################")
                content = obj.get("content", "").replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
                print(content[:1500])
                if len(content) > 1500:
                    print(f"... [truncated, total len={len(content)}]")
        except Exception as e:
            pass
