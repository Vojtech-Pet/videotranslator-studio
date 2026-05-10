from __future__ import annotations

from typing import Any

from agents.audio_agent import AudioAgent
from agents.asr_agent import ASRAgent
from agents.eval_agent import EvalAgent
from agents.memory_agent import MemoryAgent
from agents.text_agent import TextAgent


class Orchestrator:
    def __init__(self) -> None:
        self.text_agent = TextAgent()
        self.audio_agent = AudioAgent()
        self.eval_agent = EvalAgent()
        self.memory_agent = MemoryAgent()
        self.asr_agent = ASRAgent()

    def process_segment(self, seg: dict[str, Any]) -> dict[str, Any]:
        seg_copy = dict(seg)
        hint = self.memory_agent.get_hint()
        text_candidates = self.text_agent.generate_candidates(seg_copy, hint=hint)
        audio_candidates = self.audio_agent.prepare_audio_candidates(seg_copy, text_candidates)
        best = self.eval_agent.choose_best(audio_candidates)
        asr_text = self.asr_agent.transcribe(best["tts_ready_text"])

        asr_penalty = 0.0
        if any(bad in asr_text for bad in ("ajznul", "koalesk", "esikjuel", "esikjul")):
            asr_penalty += 0.3
        if best.get("prompt_name") == "locked_terms":
            expected_terms = [term for term in ("sql", "null", "is null", "is not null", "isnull", "coalesce") if term in best["tts_ready_text"].lower()]
            missing_terms = [term for term in expected_terms if term not in asr_text]
            if missing_terms:
                asr_penalty += 0.1 * len(missing_terms)

        adjusted_final_score = max(0.0, round(float(best["final_score"]) - asr_penalty, 3))

        seg_copy["tts_input"] = best["tts_ready_text"]
        seg_copy["selected_prompt"] = best["prompt_name"]
        seg_copy["cps"] = best["cps"]
        seg_copy["duration_score"] = best["duration_score"]
        seg_copy["term_score"] = best["term_score"]
        seg_copy["spoken_score"] = best["spoken_score"]
        seg_copy["final_score"] = adjusted_final_score
        seg_copy["agent_selected_source"] = best["source"]
        seg_copy["text_candidate_count"] = len(text_candidates)
        seg_copy["audio_candidate_count"] = len(audio_candidates)
        seg_copy["memory_hint"] = hint
        seg_copy["asr_backcheck_text"] = asr_text
        seg_copy["asr_penalty"] = round(asr_penalty, 3)

        seg_copy["accepted"] = bool(adjusted_final_score >= 0.82)
        if not seg_copy["accepted"]:
            seg_copy["retry_mode"] = "locked_terms"
            self.memory_agent.record_failure(seg_copy)
        else:
            seg_copy["retry_mode"] = None
            self.memory_agent.record_success(seg_copy)

        return seg_copy
