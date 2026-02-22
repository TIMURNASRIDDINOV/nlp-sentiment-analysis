import argparse
import logging
import sys
import uvicorn

from cli import run_repl, build_pipeline
from web import create_app
from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def parse_args(argv: list[str] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NLP Sentiment Analysis Tool")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Run mode: 'cli' (interactive) or 'web' (API/UI).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.model,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=settings.device,
        help="Device to use (-1 for CPU, 0+ for GPU).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help="Number of sentiment labels to return.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for CLI mode.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help="Host for web mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port for web mode.",
    )
    
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
    except SystemExit:
        return 2

    # Validation as expected by tests
    if args.top_k < 1:
        log.error("top_k must be at least 1")
        return 2
    if args.batch_size < 1:
        log.error("batch_size must be at least 1")
        return 2

    if args.mode == "cli":
        try:
            nlp = build_pipeline(args.model, args.device)
            run_repl(nlp, args.top_k, args.batch_size)
        except Exception as e:
            log.error(f"Failed to load model or run CLI: {e}")
            return 1
    else:
        try:
            app = create_app(model=args.model, device=args.device)
            uvicorn.run(app, host=args.host, port=args.port)
        except Exception as e:
            log.error(f"Failed to start web server: {e}")
            return 1
            
    return 0


if __name__ == "__main__":
    sys.exit(main())
