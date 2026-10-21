
from pathlib import Path
import json
import pandas as pd
import html

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets/football/team_identification"
MANIFEST = DATA_DIR / "team_crop_manifest.csv"
OUTPUT_HTML = DATA_DIR / "label_team_crops.html"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST).fillna("")

    required = {
        "image_path",
        "frame_id",
        "canonical_id",
        "team_label",
        "notes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    records = []

    for row in df.to_dict(orient="records"):
        image_path = Path(str(row["image_path"]))
        crop_path = ROOT / image_path

        if not crop_path.exists():
            print(f"Warning: image not found: {crop_path}")
            continue

        records.append({
            "image_path": str(image_path).replace("\\", "/"),
            "image_src": "crops/" + crop_path.name,
            "frame_id": int(row["frame_id"]),
            "canonical_id": int(row["canonical_id"]),
            "team_label": str(row["team_label"]),
            "notes": str(row["notes"]),
            "quality": "good",
        })

    data_json = json.dumps(records).replace("</", "<\\/")

    page = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AthletiQ Team Crop Labeler</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f4f5f7; color: #18202b; }
header { position: sticky; top: 0; background: #fff; padding: 14px; z-index: 5;
         border: 1px solid #ddd; border-radius: 10px; margin-bottom: 18px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
button, select { padding: 9px 12px; cursor: pointer; border: 1px solid #bbb;
                 border-radius: 7px; background: white; }
button.active { background: #dceafe; border-color: #3875d7; }
#grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(190px,1fr)); gap: 12px; }
.card { background: white; border: 2px solid #ddd; border-radius: 9px; padding: 9px; }
.card img { width: 100%; height: 190px; object-fit: contain; background: #e9ecef; }
.meta { font-size: 12px; margin: 6px 0; color: #454d58; }
.labels { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
.labels button { padding: 7px 3px; font-size: 12px; }
.card[data-label="team_a"] { border-color: #2878cf; }
.card[data-label="team_b"] { border-color: #d84a4a; }
.card[data-label="referee"] { border-color: #c18a20; }
.card[data-label="unknown"] { border-color: #777; }
.card[data-quality="poor"] { background: #fff0f0; }
.small { font-size: 12px; color: #555; }
</style>
</head>
<body>
<header>
<h2>AthletiQ — Team Crop Labeler</h2>
<div id="stats">Loading images…</div>
<p class="small">Click a label for each image. Use Unknown when unsure.
Mark blurry or unusable crops as poor quality. Export your labels before closing.</p>
<div class="controls">
<select id="filter">
<option value="all">All images</option>
<option value="unlabeled">Unlabeled only</option>
<option value="team_a">Team A</option>
<option value="team_b">Team B</option>
<option value="referee">Referee</option>
<option value="unknown">Unknown</option>
<option value="poor">Poor quality</option>
</select>
<button onclick="exportCsv()">Export labeled CSV</button>
<button onclick="setVisibleUnknown()">Mark visible unlabeled as unknown</button>
</div>
<div class="controls">
<button onclick="showFilter('unlabeled')">Next: unlabeled</button>
<button onclick="showFilter('all')">Show all</button>
</div>
</header>
<div id="grid"></div>
<script>
const items = DATA_PLACEHOLDER;
const labels = ["team_a", "team_b", "referee", "unknown"];
const grid = document.getElementById("grid");
const filter = document.getElementById("filter");

function render() {
  grid.innerHTML = "";
  const wanted = filter.value;
  items.forEach((item, index) => {
    const label = item.team_label || "";
    const quality = item.quality || "good";
    if (wanted === "unlabeled" && label) return;
    if (wanted === "poor" && quality !== "poor") return;
    if (labels.includes(wanted) && label !== wanted) return;
    if (wanted === "all" || wanted === "unlabeled" ||
        wanted === "poor" || labels.includes(wanted)) {
      const card = document.createElement("div");
      card.className = "card";
      card.dataset.label = label;
      card.dataset.quality = quality;

      const img = document.createElement("img");
      img.src = item.image_src;
      img.loading = "lazy";
      img.alt = "Player crop";

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `ID ${item.canonical_id} · Frame ${item.frame_id} · ${label || "UNLABELED"}`;

      const buttons = document.createElement("div");
      buttons.className = "labels";
      labels.forEach(name => {
        const b = document.createElement("button");
        b.textContent = name;
        if (label === name) b.classList.add("active");
        b.onclick = () => {
          item.team_label = name;
          render();
        };
        buttons.appendChild(b);
      });

      const qualityButton = document.createElement("button");
      qualityButton.style.width = "100%";
      qualityButton.style.marginTop = "6px";
      qualityButton.textContent = quality === "poor"
        ? "Poor quality — click to restore"
        : "Mark crop poor quality";
      qualityButton.onclick = () => {
        item.quality = quality === "poor" ? "good" : "poor";
        render();
      };

      card.append(img, meta, buttons, qualityButton);
      grid.appendChild(card);
    }
  });

  const counts = Object.fromEntries(labels.map(
    name => [name, items.filter(x => x.team_label === name).length]
  ));
  const unlabeled = items.filter(x => !x.team_label).length;
  const poor = items.filter(x => x.quality === "poor").length;
  document.getElementById("stats").textContent =
    `${items.length} images · ${unlabeled} unlabeled · ` +
    `Team A: ${counts.team_a} · Team B: ${counts.team_b} · ` +
    `Referee: ${counts.referee} · Unknown: ${counts.unknown} · Poor: ${poor}`;
}

function showFilter(value) {
  filter.value = value;
  render();
}

filter.onchange = render;

function setVisibleUnknown() {
  if (!confirm("Mark every currently unlabeled image as unknown?")) return;
  items.forEach(item => {
    if (!item.team_label) item.team_label = "unknown";
  });
  render();
}

function exportCsv() {
  const columns = [
    "image_path", "frame_id", "canonical_id", "team_label", "notes", "quality"
  ];
  const quote = value => '"' + String(value ?? "").replace(/"/g, '""') + '"';
  const lines = [columns.join(",")];
  items.forEach(item => {
    lines.push(columns.map(key => quote(
      key === "quality" ? item.quality : item[key]
    )).join(","));
  });
  const blob = new Blob(["\ufeff" + lines.join("\r\n")],
                        {type: "text/csv;charset=utf-8;"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "team_crop_manifest_labeled.csv";
  a.click();
  URL.revokeObjectURL(url);
}

render();
</script>
</body>
</html>
""".replace("DATA_PLACEHOLDER", data_json)

    OUTPUT_HTML.write_text(page, encoding="utf-8")
    print(f"Images included: {len(records)}")
    print(f"Labeling tool created: {OUTPUT_HTML}")
    print("\nOpen this HTML file in your browser to label the crops.")


if __name__ == "__main__":
    main()
