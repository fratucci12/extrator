import argparse
import json
from pathlib import Path
from typing import Iterable

from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont


def draw_preview(pdf_path: Path, json_path: Path, out_dir: Path) -> None:
    detections = json.loads(json_path.read_text())
    if not detections:
        print(f"[SKIP] Nenhuma tabela em {json_path.name}")
        return

    page = detections[0]["page"]
    images = convert_from_path(str(pdf_path), first_page=page, last_page=page)
    image = images[0].convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        ymin = image.height - box["ymax"]
        ymax = image.height - box["ymin"]
        draw.rectangle(
            [(box["xmin"], ymin), (box["xmax"], ymax)],
            outline="red",
            width=3,
        )
        draw.text(
            (box["xmin"], ymin - 12),
            f"{det['label']} ({det['score']:.2f})",
            fill="yellow",
            font=font,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}_preview.png"
    image.save(out_path)
    print(f"[OK] {out_path}")


def json_files(directory: Path) -> Iterable[Path]:
    yield from sorted(
        p for p in directory.glob("*_tables.json") if p.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera imagens com boxes das tabelas detectadas pelo Table Transformer. "
            "Aceita PDF/JSON individuais ou diretórios."
        )
    )
    parser.add_argument("pdf", type=Path, help="Caminho do PDF ou diretório com PDFs")
    parser.add_argument("json", type=Path, help="JSON único ou diretório com JSONs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/table_transformer"),
        help="Diretório onde salvar os previews",
    )
    args = parser.parse_args()

    if args.pdf.is_dir() and args.json.is_dir():
        for json_path in json_files(args.json):
            pdf_stem = json_path.stem.split("_page")[0]
            pdf_path = args.pdf / f"{pdf_stem}.pdf"
            if not pdf_path.exists():
                print(f"[WARN] PDF não encontrado para {json_path.name}")
                continue
            draw_preview(pdf_path, json_path, args.out_dir)
    elif args.pdf.is_file() and args.json.is_file():
        draw_preview(args.pdf, args.json, args.out_dir)
    else:
        raise ValueError(
            "Use PDF/JSON individuais ou ambos como diretórios correspondentes."
        )


if __name__ == "__main__":
    main()
