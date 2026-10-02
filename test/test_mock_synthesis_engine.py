import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from coeirocore.devices import DeviceBackend

from voicevox_engine.dev.core.mock import supported_devices as core_supported_devices
from voicevox_engine.dev.synthesis_engine import MockSynthesisEngine
from voicevox_engine.kana_parser import create_kana
from voicevox_engine.model import AccentPhrase, AudioQuery, Mora, SupportedDevicesInfo
from voicevox_engine.synthesis_engine.make_synthesis_engines import (
    make_synthesis_engines,
    resolve_device,
)


class TestMockSynthesisEngine(TestCase):
    def setUp(self):
        super().setUp()

        self.accent_phrases_hello_hiho = [
            AccentPhrase(
                moras=[
                    Mora(
                        text="コ",
                        consonant="k",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ン",
                        consonant=None,
                        consonant_length=None,
                        vowel="N",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ニ",
                        consonant="n",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="チ",
                        consonant="ch",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ワ",
                        consonant="w",
                        consonant_length=0.0,
                        vowel="a",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=5,
                pause_mora=Mora(
                    text="、",
                    consonant=None,
                    consonant_length=None,
                    vowel="pau",
                    vowel_length=0.0,
                    pitch=0.0,
                ),
            ),
            AccentPhrase(
                moras=[
                    Mora(
                        text="ヒ",
                        consonant="h",
                        consonant_length=0.0,
                        vowel="i",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ホ",
                        consonant="h",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="デ",
                        consonant="d",
                        consonant_length=0.0,
                        vowel="e",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ス",
                        consonant="s",
                        consonant_length=0.0,
                        vowel="U",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=1,
                pause_mora=None,
            ),
        ]
        self.audio_manager = Mock()
        self.audio_manager.synthesis.return_value = np.zeros(32, dtype=np.float32)
        self.engine = MockSynthesisEngine(
            speakers="",
            supported_devices="",
            audio_manager=self.audio_manager,
        )

    def test_replace_phoneme_length(self):
        self.assertEqual(
            self.engine.replace_phoneme_length(
                accent_phrases=self.accent_phrases_hello_hiho,
                speaker_id=0,
            ),
            self.accent_phrases_hello_hiho,
        )

    def test_replace_mora_pitch(self):
        self.assertEqual(
            self.engine.replace_mora_pitch(
                accent_phrases=self.accent_phrases_hello_hiho,
                speaker_id=0,
            ),
            self.accent_phrases_hello_hiho,
        )

    def test_synthesis(self):
        wave = self.engine.synthesis(
            AudioQuery(
                accent_phrases=self.accent_phrases_hello_hiho,
                speedScale=1,
                pitchScale=0,
                intonationScale=1,
                volumeScale=1,
                prePhonemeLength=0.1,
                postPhonemeLength=0.1,
                outputSamplingRate=24000,
                outputStereo=False,
                kana=create_kana(self.accent_phrases_hello_hiho),
            ),
            speaker_id=0,
        )
        self.audio_manager.synthesis.assert_called_once()
        self.assertEqual(wave.shape, (32,))

    def test_device_is_passed_to_core_audio_manager(self):
        with patch(
            "voicevox_engine.dev.synthesis_engine.mock.AudioManager"
        ) as audio_manager_class:
            MockSynthesisEngine(
                speakers="",
                device="opencl",
                device_index=2,
                opencl_platform_index=1,
            )

        audio_manager_class.assert_called_once_with(
            fs=44100,
            device="opencl",
            device_index=2,
            opencl_platform_index=1,
            use_gpu=None,
            speaker_info_dir=Path("speaker_info"),
            cpu_num_threads=None,
        )

    def test_make_synthesis_engines_passes_complete_device_selection(self):
        engine = Mock()
        with (
            patch("voicevox_engine.dev.core.metas", return_value="[]"),
            patch(
                "voicevox_engine.dev.core.supported_devices",
                return_value='{"cpu": true}',
            ),
            patch(
                "voicevox_engine.dev.synthesis_engine.MockSynthesisEngine",
                return_value=engine,
            ) as engine_class,
        ):
            result = make_synthesis_engines(
                voicelib_dirs=[],
                runtime_dirs=[],
                speaker_info_dir=Path("speaker_info"),
                device="opencl",
                device_index=2,
                opencl_platform_index=1,
            )

        self.assertEqual(result, {"0.1.0": engine})
        engine_class.assert_called_once_with(
            speakers="[]",
            supported_devices='{"cpu": true}',
            speaker_info_dir=Path("speaker_info").resolve(),
            cpu_num_threads=0,
            device="opencl",
            device_index=2,
            opencl_platform_index=1,
        )

    def test_resolve_device_keeps_new_devices_and_maps_legacy_gpu_flag(self):
        self.assertEqual(resolve_device("directml"), "directml")
        self.assertEqual(resolve_device("opencl"), "opencl")
        self.assertEqual(resolve_device(use_gpu=True), "cuda")
        self.assertEqual(resolve_device(use_gpu=False), "cpu")

    def test_resolve_device_rejects_conflicting_arguments(self):
        with self.assertRaisesRegex(ValueError, "同時に指定"):
            resolve_device("cpu", use_gpu=True)

    def test_supported_devices_delegates_to_core_capability(self):
        capabilities = {
            DeviceBackend.CPU: SimpleNamespace(available=True),
            DeviceBackend.CUDA: SimpleNamespace(available=False),
            DeviceBackend.DIRECTML: SimpleNamespace(available=True),
            DeviceBackend.OPENCL: SimpleNamespace(available=True),
        }

        with patch(
            "coeirocore.devices.get_supported_device_capabilities",
            return_value=capabilities,
        ):
            device_info = json.loads(core_supported_devices())
            self.assertEqual(
                device_info,
                {"cpu": True, "cuda": False, "dml": True, "opencl": True},
            )
            self.assertEqual(
                SupportedDevicesInfo(**device_info).model_dump(),
                {"cpu": True, "cuda": False, "dml": True},
            )

    def test_supported_devices_does_not_hide_unexpected_probe_errors(self):
        with (
            patch(
                "coeirocore.devices.get_supported_device_capabilities",
                side_effect=RuntimeError("probe failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "probe failed"),
        ):
            core_supported_devices()
