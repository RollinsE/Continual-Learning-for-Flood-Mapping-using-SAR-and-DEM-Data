from floods.training_data_audit import _ids_from_stem, _overlap_map, TileRecord


def _row(split: str, tile: str) -> TileRecord:
    event, area, scene = _ids_from_stem(tile)
    return TileRecord(split, tile, event, area, scene, 100, 10, 0, 0.1)


def test_identifier_levels_are_extracted_from_tile_name():
    assert _ids_from_stem("EMSR342-5-3_0_1317") == ("EMSR342", "EMSR342-5", "EMSR342-5-3")


def test_scene_overlap_is_detected_without_exact_tile_overlap():
    rows = [_row("train", "EMSR342-5-3_0_0"), _row("val", "EMSR342-5-3_512_0")]
    assert _overlap_map(rows, "scene_id") == {"EMSR342-5-3": ["train", "val"]}
    assert _overlap_map(rows, "tile") == {}
