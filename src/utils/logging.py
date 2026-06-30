from __future__ import annotations

import logging


def setup_logging(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name if name else "wc2026")
