"""Geographic task definitions for continual learning.

MMFlood events are tagged with a country (and usually a lat/lon) in
``activations.json``. A region task trains on every event whose country
belongs to the region, which gives a much stronger domain shift (terrain,
land cover, climate, acquisition geometry) than splitting by year.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REGIONS = ("europe", "asia", "africa", "americas", "oceania")

_COUNTRIES: Dict[str, Tuple[str, ...]] = {
    "europe": (
        "albania", "austria", "belarus", "belgium", "bosnia and herzegovina", "bosnia-herzegovina", "bosnia", "bulgaria",
        "croatia", "cyprus", "czech republic", "czechia", "denmark", "estonia", "finland", "france", "germany",
        "greece", "hungary", "iceland", "ireland", "italy", "kosovo", "latvia", "lithuania", "luxembourg", "malta",
        "moldova", "republic of moldova", "montenegro", "netherlands", "north macedonia", "macedonia", "norway",
        "poland", "portugal", "romania", "russia", "russian federation", "serbia", "slovakia", "slovenia", "spain",
        "sweden", "switzerland", "ukraine", "united kingdom", "uk", "great britain", "england", "scotland", "wales",
    ),
    "asia": (
        "afghanistan", "armenia", "azerbaijan", "bangladesh", "bhutan", "cambodia", "china", "georgia", "india",
        "indonesia", "iran", "islamic republic of iran", "iraq", "israel", "japan", "jordan", "kazakhstan",
        "kyrgyzstan", "laos", "lao pdr", "lebanon", "malaysia", "mongolia", "myanmar", "burma", "nepal",
        "north korea", "pakistan", "philippines", "saudi arabia", "south korea", "korea", "sri lanka", "syria",
        "taiwan", "tajikistan", "thailand", "timor-leste", "east timor", "turkey", "turkiye", "turkmenistan",
        "united arab emirates", "uzbekistan", "vietnam", "viet nam", "yemen", "oman", "kuwait", "qatar", "bahrain",
    ),
    "africa": (
        "algeria", "angola", "benin", "botswana", "burkina faso", "burundi", "cameroon", "cape verde", "chad",
        "comoros", "congo", "democratic republic of the congo", "dr congo", "drc", "cote d'ivoire", "ivory coast",
        "djibouti", "egypt", "eritrea", "eswatini", "swaziland", "ethiopia", "gabon", "gambia", "ghana", "guinea",
        "kenya", "lesotho", "liberia", "libya", "madagascar", "malawi", "mali", "mauritania", "mauritius",
        "morocco", "mozambique", "namibia", "niger", "nigeria", "rwanda", "senegal", "sierra leone", "somalia",
        "south africa", "south sudan", "sudan", "tanzania", "togo", "tunisia", "uganda", "zambia", "zimbabwe",
    ),
    "americas": (
        "argentina", "bahamas", "belize", "bolivia", "brazil", "canada", "chile", "colombia", "costa rica", "cuba",
        "dominica", "dominican republic", "ecuador", "el salvador", "guatemala", "guyana", "haiti", "honduras",
        "jamaica", "mexico", "nicaragua", "panama", "paraguay", "peru", "puerto rico", "suriname",
        "trinidad and tobago", "united states", "united states of america", "usa", "us", "uruguay", "venezuela",
    ),
    "oceania": (
        "australia", "fiji", "new zealand", "papua new guinea", "samoa", "solomon islands", "tonga", "vanuatu",
    ),
}
COUNTRY_TO_REGION: Dict[str, str] = {country: region for region, names in _COUNTRIES.items() for country in names}


def _normalise_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.replace("_", " ").strip().lower())


def region_from_lat_lon(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Coarse continent from coordinates; used only when the country is missing or unknown."""
    if lat is None or lon is None:
        return None
    lat, lon = float(lat), float(lon)
    if lon < -25:
        return "americas"
    if lat < -10 and lon > 110:
        return "oceania"
    if lat > 35 and lon < 45:
        return "europe"
    if lat < 37 and lon < 52 and not (lat > 12 and lon > 34):
        return "africa"
    return "asia"


def region_for_event(details: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Return (region, country label) for one activations.json entry."""
    raw_country = details.get("country") or details.get("countries") or ""
    if isinstance(raw_country, (list, tuple)):
        raw_country = raw_country[0] if raw_country else ""
    country = _normalise_name(raw_country)
    for candidate in [country] + [part.strip() for part in re.split(r"[,;/]| and ", country) if part.strip()]:
        if candidate in COUNTRY_TO_REGION:
            return COUNTRY_TO_REGION[candidate], country
    lat = details.get("lat", details.get("latitude"))
    lon = details.get("lon", details.get("lng", details.get("longitude")))
    try:
        return region_from_lat_lon(lat, lon), country or "unknown"
    except (TypeError, ValueError):
        return None, country or "unknown"


def load_event_regions(activations_json_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ({event: region}, {event: country}) from MMFlood activations metadata."""
    path = Path(activations_json_path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    regions: Dict[str, str] = {}
    countries: Dict[str, str] = {}
    for event_id, details in data.items():
        region, country = region_for_event(details or {})
        key = str(event_id).upper()
        countries[key] = country
        if region:
            regions[key] = region
    if not regions:
        raise ValueError(f"No event regions could be derived from {path}: entries need a 'country' or 'lat'/'lon' field.")
    return regions, countries


def parse_region_tasks(values: Sequence[str] | None) -> List[Tuple[str, Tuple[str, ...]]]:
    """Parse task specs such as ``europe asia americas+africa+oceania`` into (name, regions)."""
    if not values:
        raise ValueError("At least one region task is required, e.g. --task-regions europe asia americas+africa+oceania")
    tasks: List[Tuple[str, Tuple[str, ...]]] = []
    used: set = set()
    for raw in values:
        parts = tuple(p.strip().lower() for p in re.split(r"[+,]", str(raw)) if p.strip())
        unknown = [p for p in parts if p not in REGIONS]
        if not parts or unknown:
            raise ValueError(f"Invalid region task '{raw}'. Use regions from {', '.join(REGIONS)}, joined with '+'.")
        overlap = used.intersection(parts)
        if overlap:
            raise ValueError(f"Region(s) {sorted(overlap)} appear in more than one CL task.")
        used.update(parts)
        tasks.append(("+".join(parts), parts))
    return tasks
