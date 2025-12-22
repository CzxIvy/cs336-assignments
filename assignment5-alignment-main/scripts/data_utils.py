import json
import logging
import pathlib
from typing import List, Dict

logger = logging.getLogger(__name__)

def load_r1_zero_prompt() -> str:
    prompt_path = pathlib.Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()