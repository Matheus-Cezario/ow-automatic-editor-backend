from __future__ import annotations

import logging
import os
import sys


def setup_logging(service: str) -> logging.Logger:
    level = os.environ.get("OW_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s %(levelname)-5s [{service}] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("botocore").setLevel("WARNING")
    return logging.getLogger(service)
