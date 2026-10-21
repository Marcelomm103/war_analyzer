from importlib.util import find_spec
from typing import Final

ANALYZER_MODULES: Final = (
    "anchor_coordinates",
    "pixel_coordinates",
    "pixel_hsv",
    "war_analyzer",
    "war_analyzer_gemini",
    "war_analyzer_resumed",
)


def test_legacy_modules_remain_installable() -> None:
    for module_name in ANALYZER_MODULES:
        assert find_spec(module_name) is not None
