import logging
import os
import sys


def get_logger(name: str):
    level = os.environ.get("LOGLEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("[%(thread)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def prettify_source(source):
    document = os.path.basename(source.get("document"))
    score = source.get("score")
    content_preview = source.get("content_preview")
    return f"• **{document}** with score ({round(score,2)}) \n\n **Preview:** \n {content_preview} \n"