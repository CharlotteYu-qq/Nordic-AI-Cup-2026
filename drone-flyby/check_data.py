
import glob, json, cv2
from dtos import OBJECT_CLASSES

counts = {k: 0 for k in OBJECT_CLASSES}
areas = {k: [] for k in OBJECT_CLASSES}

for f in glob.glob("src/helsinki/annotations/*.json"):
    with open(f) as fp:
        data = json.load(fp)
        for obj in data:
            c = obj["object_id"]
            if c in counts:
                counts[c] += 1
                b = obj["bbox"]
                areas[c].append((b[2] - b[0]) * (b[3] - b[1]))

print(f"{'Class':<18} | {'Total Boxes':<12} | {'Avg 4K Area (px^2)':<18}")
print("-" * 52)
for k in OBJECT_CLASSES:
    avg_area = sum(areas[k]) / len(areas[k]) if areas[k] else 0
    print(f"{k:<18} | {counts[k]:<12} | {avg_area:<18.1f}")