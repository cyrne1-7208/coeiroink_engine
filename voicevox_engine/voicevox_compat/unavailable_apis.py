"""COEIROINKが提供しないVOICEVOX APIへ明示的な応答を返す。"""

from fastapi import APIRouter, HTTPException


def add_unavailable_routes(router: APIRouter) -> None:
    """歌唱と音声ライブラリ管理の互換パスを501応答として登録する。"""

    @router.get("/singers", include_in_schema=False)
    @router.get("/singer_info", include_in_schema=False)
    @router.post("/sing_frame_audio_query", include_in_schema=False)
    @router.post("/sing_frame_f0", include_in_schema=False)
    @router.post("/sing_frame_volume", include_in_schema=False)
    @router.post("/frame_synthesis", include_in_schema=False)
    def unsupported_singing_api() -> None:
        raise HTTPException(
            status_code=501,
            detail="COEIROINKは歌唱機能を提供していません。",
        )

    @router.get("/downloadable_libraries", include_in_schema=False)
    @router.get("/installed_libraries", include_in_schema=False)
    def unsupported_library_query() -> None:
        raise HTTPException(
            status_code=501,
            detail="COEIROINKはVOICEVOX音声ライブラリ管理機能を提供していません。",
        )

    @router.post("/install_library/{library_uuid}", include_in_schema=False)
    @router.post("/uninstall_library/{library_uuid}", include_in_schema=False)
    def unsupported_library_mutation(library_uuid: str) -> None:
        raise HTTPException(
            status_code=501,
            detail="COEIROINKはVOICEVOX音声ライブラリ管理機能を提供していません。",
        )
