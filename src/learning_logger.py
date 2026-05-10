from __future__ import annotations

from src.project_memory import ProjectMemory


def log_segment_outcome(memory: ProjectMemory, seg: dict) -> None:
    memory.add_segment_history(
        {
            "seg": seg.get("seg"),
            "source_text": seg.get("source_text"),
            "tts_input": seg.get("tts_input"),
            "final_score": seg.get("final_score"),
            "retry_mode": seg.get("retry_mode"),
            "accepted": seg.get("accepted"),
            "term_score": seg.get("term_score"),
            "terminology_consistency_score": seg.get("terminology_consistency_score"),
            "asr_match_score": seg.get("asr_match_score"),
            "duration_score": seg.get("duration_score"),
            "duration_status": seg.get("duration_status"),
            "coherence_score": seg.get("coherence_score"),
            "spoken_score": seg.get("spoken_score"),
            "persona_name": seg.get("persona_name"),
            "scene_type": seg.get("scene_type"),
            "scene_base_style": seg.get("scene_base_style"),
            "prompt_name": seg.get("prompt_name"),
            "force_locked_terms": seg.get("force_locked_terms"),
        }
    )
