import json
from pathlib import Path

import torch
from pdf2image import convert_from_path
from transformers import (
    DetrImageProcessor,
    TableTransformerForObjectDetection,
)

processor = DetrImageProcessor.from_pretrained("table-transformer-structure-recognition")
model = TableTransformerForObjectDetection.from_pretrained("table-transformer-structure-recognition")

pdf_path = Path("meus_pdfs/edital7.pdf")
out_path = Path("build/table_transformer")
out_path.mkdir(parents=True, exist_ok=True)

# Exemplo: só a primeira página; ajuste conforme precisar
images = convert_from_path(str(pdf_path), first_page=1, last_page=30)

for idx, image in enumerate(images, start=1):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.85,  # ajuste do score mínimo
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append(
            {
                "page": idx,
                "score": float(score),
                "label": model.config.id2label[int(label)],
                "box": {
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3]),
                },
            }
        )

    (out_path / f"{pdf_path.stem}_page{idx:02d}_tables.json").write_text(
        json.dumps(detections, indent=2),
        encoding="utf-8",
    )
    print(f"Página {idx}: {len(detections)} tabela(s) detectada(s)")
