"""SWDE Dataset Loader and Preprocessor.

Parses the SWDE benchmark dataset and prepares it for evaluation with
the Cog+/VGS pipeline, matching the LiveWeb-IE paper's protocol (Appendix B.2):
- Sample 100 pages per website
- Remove sites with rendering issues (CollegeToolkit, FanHouse)
- Generate natural language queries from attribute names
- Group pages by (website, attribute) for wrapper generalization testing

SWDE structure:
- 8 verticals: auto, book, camera, job, movie, nbaplayer, restaurant, university
- 80 websites (10 per vertical)
- 124,291 HTML pages
- GT format: vertical-site-attribute.txt with page_id, count, values
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# SWDE attribute schema
VERTICAL_ATTRS = {
    "auto": ["model", "price", "engine", "fuel_economy"],
    "book": ["title", "author", "isbn_13", "publisher", "publication_date"],
    "camera": ["model", "price", "manufacturer"],
    "job": ["title", "company", "location", "date_posted"],
    "movie": ["title", "director", "genre", "mpaa_rating"],
    "nbaplayer": ["name", "team", "height", "weight"],
    "restaurant": ["name", "address", "phone", "cuisine"],
    "university": ["name", "phone", "website", "type"],
}

# Sites excluded per paper (rendering issues)
EXCLUDED_SITES = {
    "university": ["collegetoolkit"],
    "nbaplayer": ["fanhouse"],
}

# Natural language query templates for each vertical+attribute
QUERY_TEMPLATES = {
    "auto": {
        "model": "Extract the car model name",
        "price": "Extract the price of the car",
        "engine": "Extract the engine specifications",
        "fuel_economy": "Extract the fuel economy ratings",
    },
    "book": {
        "title": "Extract the book title",
        "author": "Extract the author name(s)",
        "isbn_13": "Extract the ISBN-13 number",
        "publisher": "Extract the publisher name",
        "publication_date": "Extract the publication date",
    },
    "camera": {
        "model": "Extract the camera model name",
        "price": "Extract the price of the camera",
        "manufacturer": "Extract the manufacturer/brand name",
    },
    "job": {
        "title": "Extract the job title",
        "company": "Extract the company name",
        "location": "Extract the job location",
        "date_posted": "Extract the date the job was posted",
    },
    "movie": {
        "title": "Extract the movie title",
        "director": "Extract the director name(s)",
        "genre": "Extract the movie genre(s)",
        "mpaa_rating": "Extract the MPAA rating",
    },
    "nbaplayer": {
        "name": "Extract the player's name",
        "team": "Extract the team name",
        "height": "Extract the player's height",
        "weight": "Extract the player's weight",
    },
    "restaurant": {
        "name": "Extract the restaurant name",
        "address": "Extract the restaurant address",
        "phone": "Extract the phone number",
        "cuisine": "Extract the cuisine type",
    },
    "university": {
        "name": "Extract the university name",
        "phone": "Extract the phone number",
        "website": "Extract the university website URL",
        "type": "Extract the type of institution",
    },
}


def parse_gt_file(filepath: str) -> dict:
    """Parse a SWDE ground truth file.
    
    Format:
        Line 1: vertical<tab>site<tab>attribute
        Line 2: total_pages<tab>pages_with_values<tab>total_values<tab>unique_values
        Line 3+: page_id<tab>count<tab>value1<tab>value2...
    
    Returns: {page_id: [values]}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        return {}
    
    gt = {}
    for line in lines[2:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        page_id = parts[0].strip()
        count = int(parts[1].strip())
        values = [v.strip() for v in parts[2:] if v.strip() and v.strip() != "<NULL>"]
        if values:
            gt[page_id] = values
    
    return gt


def get_html_path(data_root: str, vertical: str, site: str, page_id: str) -> Optional[str]:
    """Get the HTML file path for a given page.
    
    Site directories are named like: auto-aol(2000), movie-imdb(2000)
    """
    vertical_dir = Path(data_root) / f"{vertical}_extracted" / vertical
    
    if not vertical_dir.exists():
        return None
    
    # Find the site directory (e.g., auto-aol(2000))
    for site_dir in vertical_dir.iterdir():
        if not site_dir.is_dir():
            continue
        # Match site name (e.g., "auto-aol" matches "auto-aol(2000)")
        site_name = site_dir.name.split("(")[0]
        full_site = f"{vertical}-{site}"
        if site_name == full_site:
            html_path = site_dir / f"{page_id}.htm"
            if html_path.exists():
                return str(html_path)
    
    return None


def extract_url_from_html(html_path: str) -> Optional[str]:
    """Extract the original URL from the <base href> tag."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(5000)  # Only read first 5KB
        match = re.search(r'<base\s+href="([^"]+)"', content, re.IGNORECASE)
        if match:
            return match.group(1)
    except:
        pass
    return None


def is_site_excluded(vertical: str, site: str) -> bool:
    """Check if a site is excluded per paper protocol."""
    excluded = EXCLUDED_SITES.get(vertical, [])
    return site in excluded


def load_swde_dataset(
    data_root: str,
    sample_per_site: int = 100,
    verticals: Optional[list] = None,
) -> list[dict]:
    """Load SWDE dataset and return grouped samples.
    
    Each group represents one (website, attribute) pair with:
    - sample_id: swde_{vertical}_{site}_{attr}
    - vertical: domain name
    - site: website name
    - attribute: attribute to extract
    - query: natural language query
    - urls: list of (page_url, html_path) tuples
    - gt: {page_id: {attribute: [values]}}
    
    Args:
        data_root: Path to SWDE sourceCode directory
        sample_per_site: Number of pages to sample per website (paper uses 100)
        verticals: List of verticals to load (None = all)
    
    Returns:
        List of group dictionaries
    """
    gt_root = Path(data_root) / "groundtruth_extracted" / "groundtruth"
    
    if verticals is None:
        verticals = list(VERTICAL_ATTRS.keys())
    
    groups = []
    stats = {"total_sites": 0, "total_groups": 0, "total_pages": 0, "excluded": 0}
    
    for vertical in verticals:
        attrs = VERTICAL_ATTRS[vertical]
        
        # Find all sites for this vertical
        sites = set()
        vertical_gt_dir = gt_root / vertical
        if not vertical_gt_dir.exists():
            print(f"  Warning: GT dir not found for {vertical}")
            continue
        
        for gt_file in vertical_gt_dir.glob("*.txt"):
            # Parse filename: vertical-site-attribute.txt
            stem = gt_file.stem
            parts = stem.split("-")
            if len(parts) >= 3:
                site = "-".join(parts[1:-1])  # Site name may contain hyphens
                sites.add(site)
        
        for site in sorted(sites):
            if is_site_excluded(vertical, site):
                stats["excluded"] += 1
                continue
            
            stats["total_sites"] += 1
            
            # Collect all page_ids for this site (across all attributes)
            site_pages = {}  # {page_id: {attr: [values]}}
            
            for attr in attrs:
                gt_file = vertical_gt_dir / f"{vertical}-{site}-{attr}.txt"
                if not gt_file.exists():
                    continue
                
                gt = parse_gt_file(str(gt_file))
                for page_id, values in gt.items():
                    if page_id not in site_pages:
                        site_pages[page_id] = {}
                    site_pages[page_id][attr] = values
            
            if not site_pages:
                continue
            
            # Sample pages
            all_page_ids = sorted(site_pages.keys())
            if len(all_page_ids) > sample_per_site:
                # Deterministic sampling (first N pages)
                sampled_ids = all_page_ids[:sample_per_site]
            else:
                sampled_ids = all_page_ids
            
            # Build URL list and verify HTML exists
            page_entries = []
            for page_id in sampled_ids:
                html_path = get_html_path(data_root, vertical, site, page_id)
                if html_path is None:
                    continue
                
                url = extract_url_from_html(html_path)
                if url is None:
                    url = f"file://{html_path}"
                
                page_entries.append({
                    "page_id": page_id,
                    "url": url,
                    "html_path": html_path,
                    "gt": site_pages[page_id],
                })
            
            if not page_entries:
                continue
            
            # Create one group per attribute
            for attr in attrs:
                # Filter to pages that have this attribute
                attr_pages = [p for p in page_entries if attr in p["gt"]]
                if not attr_pages:
                    continue
                
                query = QUERY_TEMPLATES.get(vertical, {}).get(attr, f"Extract the {attr}")
                
                group = {
                    "sample_id": f"swde_{vertical}_{site}_{attr}",
                    "vertical": vertical,
                    "site": site,
                    "attribute": attr,
                    "query": query,
                    "urls": [p["url"] for p in attr_pages],
                    "html_paths": [p["html_path"] for p in attr_pages],
                    "gt": {p["page_id"]: {attr: p["gt"][attr]} for p in attr_pages},
                }
                groups.append(group)
                stats["total_groups"] += 1
                stats["total_pages"] += len(attr_pages)
    
    print(f"\n=== SWDE Dataset Stats ===")
    print(f"Sites loaded: {stats['total_sites']}")
    print(f"Sites excluded: {stats['excluded']}")
    print(f"Groups (site×attr): {stats['total_groups']}")
    print(f"Total pages: {stats['total_pages']}")
    
    return groups


def save_swde_manifest(groups: list, output_path: str):
    """Save a manifest file for the SWDE evaluation run."""
    manifest = {
        "dataset": "SWDE",
        "num_groups": len(groups),
        "groups": [
            {
                "sample_id": g["sample_id"],
                "vertical": g["vertical"],
                "site": g["site"],
                "attribute": g["attribute"],
                "query": g["query"],
                "num_pages": len(g["urls"]),
            }
            for g in groups
        ]
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {output_path}")


if __name__ == "__main__":
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "data" / "swde" / "swde" / "sourceCode" / "sourceCode"
    )
    
    print(f"Loading SWDE from: {data_root}")
    groups = load_swde_dataset(data_root, sample_per_site=100)
    
    # Show sample groups
    print(f"\n=== Sample Groups ===")
    for g in groups[:5]:
        print(f"  {g['sample_id']}: {g['query']} ({len(g['urls'])} pages)")
        print(f"    Site: {g['vertical']}/{g['site']}")
        print(f"    First URL: {g['urls'][0][:80]}...")
    
    # Save manifest
    output_dir = Path(__file__).parent.parent / "output" / "swde"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_swde_manifest(groups, str(output_dir / "swde_manifest.json"))
