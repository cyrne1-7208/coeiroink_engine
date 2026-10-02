"""エンジンマニフェストの読み込み処理。"""

import json
from base64 import b64encode
from pathlib import Path
from typing import Any

from .engine_manifest import EngineManifest, LicenseInfo, UpdateInfo


class EngineManifestLoader:
    def __init__(self, manifest_path: Path, root_dir: Path):
        self.manifest_path = manifest_path
        self.root_dir = root_dir
        # マニフェストとライセンス・アイコン資産はプロセス稼働中に不変なので、機能確認のたびに再読込・再エンコードしない。
        self._raw_manifest: dict[str, Any] | None = None
        self._manifest: EngineManifest | None = None

    def _load_json(self) -> dict[str, Any]:
        if self._raw_manifest is None:
            self._raw_manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        return self._raw_manifest

    @property
    def downloadable_libraries_path(self):
        return self._load_json().get("downloadable_libraries_path")

    @property
    def downloadable_libraries_url(self):
        return self._load_json().get("downloadable_libraries_url")

    def load_manifest(self) -> EngineManifest:
        if self._manifest is not None:
            return self._manifest

        manifest = self._load_json()

        self._manifest = EngineManifest(
            manifest_version=manifest["manifest_version"],
            name=manifest["name"],
            brand_name=manifest["brand_name"],
            uuid=manifest["uuid"],
            version=manifest["version"],
            url=manifest["url"],
            default_sampling_rate=manifest["default_sampling_rate"],
            frame_rate=manifest.get("frame_rate", 93.75),
            icon=b64encode((self.root_dir / manifest["icon"]).read_bytes()).decode(
                "utf-8"
            ),
            terms_of_service=(self.root_dir / manifest["terms_of_service"]).read_text(
                "utf-8"
            ),
            update_infos=[
                UpdateInfo(**update_info)
                for update_info in json.loads(
                    (self.root_dir / manifest["update_infos"]).read_text("utf-8")
                )
            ],
            dependency_licenses=[
                LicenseInfo(**license_info)
                for license_info in json.loads(
                    (self.root_dir / manifest["dependency_licenses"]).read_text("utf-8")
                )
            ],
            supported_vvlib_manifest_version=None,
            supported_features={
                key: item["value"]
                for key, item in manifest["supported_features"].items()
            },
        )
        # 初回構築時は、呼出元へプロセス共通キャッシュそのものを渡さない。
        return self._manifest.model_copy(deep=True)
