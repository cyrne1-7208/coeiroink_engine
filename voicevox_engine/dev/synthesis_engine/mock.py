"""旧importパスを維持するための互換エイリアス。"""

from ...synthesis_engine.coeiroink_adapter import CoeiroinkVoicevoxAdapter

MockSynthesisEngine = CoeiroinkVoicevoxAdapter

__all__ = ["MockSynthesisEngine"]
