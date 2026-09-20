# check_meta.py
import json, glob

jsons = sorted(glob.glob("/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/*/*.json"))
if jsons:
    with open(jsons[0], "r") as f:
        print("线上第 0 帧 metadata:")
        print(json.dumps(json.load(f), indent=2))
else:
    print("未找到录制的 json 文件")