"""
Ablation demo: same output, different observation strategies.

Run: uv run python examples/ablation_demo.py
"""

from verifiers_interact import LineLimit, TokenBudget, Unconstrained
from verifiers_interact.folders import TruncateFolder, HeadTailFolder, StructureFolder


# Simulated REPL output: a model ran `cat utils.py` and got 80 lines back
SAMPLE_OUTPUT = '''import os
import sys
import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Application configuration."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    max_retries: int = 3
    timeout: float = 30.0

def load_config(path: str) -> Config:
    """Load configuration from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return Config(**data)

def validate_config(config: Config) -> list[str]:
    """Validate configuration values, return list of errors."""
    errors = []
    if config.port < 1 or config.port > 65535:
        errors.append(f"Invalid port: {config.port}")
    if config.timeout <= 0:
        errors.append(f"Invalid timeout: {config.timeout}")
    if config.max_retries < 0:
        errors.append(f"Invalid max_retries: {config.max_retries}")
    return errors

class Server:
    """HTTP server with retry logic."""

    def __init__(self, config: Config):
        self.config = config
        self.connections = []
        self._running = False

    def start(self):
        """Start the server."""
        errors = validate_config(self.config)
        if errors:
            raise ValueError(f"Bad config: {errors}")
        self._running = True
        print(f"Server running on {self.config.host}:{self.config.port}")

    def stop(self):
        """Gracefully stop the server."""
        self._running = False
        for conn in self.connections:
            conn.close()
        self.connections.clear()

    async def handle_request(self, request: dict) -> dict:
        """Handle an incoming request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                result = await self._process(request)
                return {"status": "ok", "data": result}
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return {"status": "error", "message": str(e)}

    async def _process(self, request: dict) -> Any:
        """Process a single request."""
        method = request.get("method", "GET")
        path = request.get("path", "/")
        return {"method": method, "path": path, "processed": True}

def create_app(config_path: str = "config.json") -> Server:
    """Factory function to create and configure a server."""
    config = load_config(config_path)
    return Server(config)

if __name__ == "__main__":
    server = create_app()
    server.start()
'''


def demo(name: str, constraint, output: str):
    """Run a constraint and show the result."""
    result = constraint.apply(output)
    lines = result.content.split("\n")
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Truncated: {result.was_truncated} | Metadata: {result.metadata}")
    print(f"{'='*70}")
    print(result.content)


if __name__ == "__main__":
    print("=" * 70)
    print("  ABLATION DEMO: Same 85-line file, different observation strategies")
    print("=" * 70)

    total_lines = len(SAMPLE_OUTPUT.strip().split("\n"))
    print(f"\n  Source: {total_lines} lines of Python (simulated `cat utils.py`)")
    print(f"  Budget: 15 lines")

    # 1. Baseline: no constraint
    demo(
        "UNCONSTRAINED (baseline — model sees everything)",
        Unconstrained(),
        SAMPLE_OUTPUT,
    )

    # 2. Naive truncation: first 15 lines
    demo(
        "LineLimit(15) + TruncateFolder — model sees first 15 lines only",
        LineLimit(15, folder=TruncateFolder()),
        SAMPLE_OUTPUT,
    )

    # 3. Head + tail: first 9 + last 6 lines
    demo(
        "LineLimit(15) + HeadTailFolder(0.6) — model sees start + end",
        LineLimit(15, folder=HeadTailFolder(0.6)),
        SAMPLE_OUTPUT,
    )

    # 4. Structural folding: function/class signatures
    demo(
        "LineLimit(15) + StructureFolder — model sees the CODE MAP",
        LineLimit(15, folder=StructureFolder()),
        SAMPLE_OUTPUT,
    )

    print(f"\n{'='*70}")
    print("  KEY INSIGHT: With StructureFolder, the model sees a table of")
    print("  contents it can navigate into — not a wall of truncated text.")
    print("  It learns to SEARCH, not SCROLL.")
    print(f"{'='*70}\n")
