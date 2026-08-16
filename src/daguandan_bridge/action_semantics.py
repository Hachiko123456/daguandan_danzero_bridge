from __future__ import annotations


_TYPE_ALIASES = {
    "pass": "PASS",
    "single": "SINGLE",
    "pair": "PAIR",
    "trips": "TRIPLE",
    "triple": "TRIPLE",
    "threewithtwo": "FULL",
    "full": "FULL",
    "fullhouse": "FULL",
    "straight": "STRAIGHT",
    "threepair": "PLATE",
    "plate": "PLATE",
    "twotrips": "TUBE",
    "tube": "TUBE",
    "bomb": "BOMB",
    "straightflush": "SFLUSH",
    "sflush": "SFLUSH",
    "rocket": "ROCKET",
    "不出": "PASS",
    "单张": "SINGLE",
    "对子": "PAIR",
    "三张": "TRIPLE",
    "三带二": "FULL",
    "顺子": "STRAIGHT",
    "三连对": "PLATE",
    "钢板": "TUBE",
    "炸弹": "BOMB",
    "同花顺": "SFLUSH",
    "四王": "ROCKET",
}

_PROJECT_TYPE_NAMES = {
    "PASS": "PASS",
    "SINGLE": "Single",
    "PAIR": "Pair",
    "TRIPLE": "Trips",
    "FULL": "ThreeWithTwo",
    "STRAIGHT": "Straight",
    "PLATE": "ThreePair",
    "TUBE": "TwoTrips",
    "BOMB": "Bomb",
    "SFLUSH": "StraightFlush",
    "ROCKET": "Rocket",
}


def canonical_fabledan_type(value: object) -> str | None:
    """Return the FableDan type name for project, FableDan, or Chinese labels."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = (
        raw.replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .casefold()
    )
    return _TYPE_ALIASES.get(normalized)


def project_play_type(value: object) -> str:
    """Convert a FableDan type to the action name used by DanZero rules."""

    canonical = canonical_fabledan_type(value)
    if canonical is None:
        return str(value)
    return _PROJECT_TYPE_NAMES[canonical]


__all__ = ["canonical_fabledan_type", "project_play_type"]
