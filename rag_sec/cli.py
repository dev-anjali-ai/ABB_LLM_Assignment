from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
import orjson

from .config import RAGConfig
from .pipeline import build_index
from .pipeline import answer_question
from .eval_questions import EVAL_QUESTIONS

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def ingest(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Directory containing PDFs"),
    index_dir: Path = typer.Option(Path("index"), "--index-dir", help="Output index directory"),
    embed_device: str = typer.Option("cuda", "--embed-device", help="cuda or cpu"),
    rerank_device: str = typer.Option("cuda", "--rerank-device", help="cuda or cpu"),
):
    cfg = RAGConfig(embed_device=embed_device, rerank_device=rerank_device)
    console.print(f"[bold]Building index[/bold] from {data_dir} -> {index_dir}")
    build_index(data_dir, index_dir, cfg)
    console.print("[green]Done.[/green]")

@app.command()
def eval(
    index_dir: Path = typer.Option(Path("index"), "--index-dir", help="Index directory"),
    out: Path = typer.Option(Path("outputs/predictions.json"), "--out", help="Output JSON path"),
):
    import os
    os.environ["RAG_INDEX_DIR"] = str(index_dir)

    results = []
    for q in EVAL_QUESTIONS:
        res = answer_question(q["question"])
        results.append({"question_id": q["question_id"], "answer": res["answer"], "sources": res["sources"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    console.print(f"[green]Wrote[/green] {out}")

if __name__ == "__main__":
    app()
