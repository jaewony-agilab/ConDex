"""Judge backends. Each is a callable Packet -> Verdict.

  VLMJudge     Qwen3-VL-8B behind vLLM's OpenAI-compatible server; frozen, or a LoRA served with
               --enable-lora (pass the adapter name as `model`). Temperature 0, JSON-schema guided.
  ClaudeJudge  Claude Haiku 5.5 reference judge; structured output; a refusal counts as unclear.
The privileged oracle lives in condex.privileged.oracle.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from condex.cost import Ledger
from condex.judge.prompt import Packet
from condex.judge.schema import SCHEMA, Verdict, parse


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


class VLMJudge:
    def __init__(self, base_url: str, model: str = "Qwen/Qwen3-VL-8B-Instruct", name: str | None = None,
                 max_tokens: int = 256, timeout: float = 120.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model, self.max_tokens = model, max_tokens
        self.name = name or f"vlm:{model}"

    @staticmethod
    def messages(packet: Packet) -> list[dict]:
        content = []
        for it in packet.content:
            if it.kind == "text":
                content.append({"type": "text", "text": it.text})
            else:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(it.image)}"}})
        return [{"role": "system", "content": packet.system}, {"role": "user", "content": content}]

    def __call__(self, packet: Packet) -> Verdict:
        r = self.client.chat.completions.create(
            model=self.model, messages=self.messages(packet), temperature=0.0, max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True}},
        )
        return parse(r.choices[0].message.content or "", self.name)


class ClaudeJudge:
    def __init__(self, ledger: Ledger, model: str = "claude-haiku-5-5", effort: str = "low", max_tokens: int = 4096):
        import anthropic

        self.client = anthropic.Anthropic()
        self.ledger, self.model, self.effort, self.max_tokens = ledger, model, effort, max_tokens
        self.name = f"claude:{model}"

    @staticmethod
    def content(packet: Packet) -> list[dict]:
        out = []
        for it in packet.content:
            if it.kind == "text":
                out.append({"type": "text", "text": it.text})
            else:
                out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(it.image)}})
        return out

    def __call__(self, packet: Packet) -> Verdict:
        self.ledger.ensure()
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=packet.system,
            messages=[{"role": "user", "content": self.content(packet)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        self.ledger.charge("judge", self.model, r.usage.input_tokens, r.usage.output_tokens,
                           episode=packet.meta.get("episode"), boundary=packet.meta.get("boundary_index"))
        if r.stop_reason == "refusal":
            return Verdict("unclear", "yes", "judge refused", self.name, parse_ok=False)
        text = next((b.text for b in r.content if b.type == "text"), "")
        return parse(text, self.name)
