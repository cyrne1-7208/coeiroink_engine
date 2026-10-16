"""旧importパス用の互換エイリアス。実体は通常合成に使うCOEIROINKアダプターであり、モックではない。"""

from ...synthesis_engine.coeiroink_adapter import CoeiroinkVoicevoxAdapter

MockSynthesisEngine = CoeiroinkVoicevoxAdapter

__all__ = ["MockSynthesisEngine"]
