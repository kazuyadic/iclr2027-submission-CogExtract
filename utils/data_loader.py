"""Load LiveWeb-IE dataset from the official JSON format."""
from __future__ import annotations

import json
from pathlib import Path


class DataLoader:
    """Parse dataset.json and group.jsonl files."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.dataset_path = data_root / "LiveWeb_IE" / "dataset.json"

    def load_dataset(self) -> dict:
        """Load dataset.json which is a dict keyed by website name."""
        with open(self.dataset_path, encoding="utf-8") as fh:
            return json.load(fh)

    def load_group_urls(self, group_url_path: str, group_id: str = "") -> list[str]:
        """Load URLs from a group.jsonl file.

        Each line is JSON like: {"g_000": {"group_url": ["https://...", ...]}}
        We match by *group_id* if provided, otherwise return all URLs.
        """
        cleaned = group_url_path.lstrip("./")
        full_path = self.data_root / "LiveWeb_IE" / cleaned
        urls = []
        with open(full_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # obj is like {"g_000": {"group_url": [...]}}
                for gid, gdata in obj.items():
                    if group_id and gid != group_id:
                        continue
                    group_urls = gdata.get("group_url", [])
                    if isinstance(group_urls, list):
                        urls.extend(group_urls)
                    elif isinstance(group_urls, str):
                        urls.append(group_urls)
        return urls

    def flatten_dataset(self) -> list[dict]:
        """Flatten the website-keyed dict into a flat list of entries."""
        raw = self.load_dataset()
        flat = []
        for website_key, entries in raw.items():
            flat.extend(entries)
        return flat

    # Websites blocked by Alibaba intranet DNS or unreachable
    EXCLUDED_WEBSITES = [
        "marinespecies", "thesportsdb", "themealdb",
        "thecocktaildb", "scrapethissite", "dp",  # 阿里域名拦截
        "huggingface",  # 网络不可达
    ]

    def build_samples(
        self,
        limit: int | None = None,
        urls_per_entry: int | None = None,
        exclude_websites: list[str] | None = None,
        task_types: list[str] | None = None,
    ) -> list[dict]:
        """Build a flat list of (url, query, metadata) samples for evaluation.

        Args:
            limit: Max total samples to return.
            urls_per_entry: Max URLs to pick per entry (None = all URLs in group).
            exclude_websites: Website keys to skip. Defaults to EXCLUDED_WEBSITES.
        """
        if exclude_websites is None:
            exclude_websites = self.EXCLUDED_WEBSITES
        excluded_set = set(exclude_websites)

        entries = self.flatten_dataset()
        type_filter = set(task_types) if task_types else None
        samples = []
        for entry in entries:
            website_key = entry.get("website", "")
            if any(ex in website_key.lower() for ex in excluded_set):
                continue
            if type_filter and entry.get("type", "") not in type_filter:
                continue

            group_urls = self.load_group_urls(entry["group_url"], entry.get("group_id", ""))
            # label is a dict: {attr_name: path}
            label = entry.get("label", {})
            label_paths = list(label.values()) if isinstance(label, dict) else label

            picked_urls = group_urls[:urls_per_entry] if urls_per_entry else group_urls
            for url in picked_urls:
                samples.append({
                    "sample_id": entry.get("sample_id", entry.get("query_id")),
                    "url": url,
                    "query": entry["query"],
                    "attributes": entry.get("attribute", []),
                    "task_type": entry.get("type", ""),
                    "label": label,
                    "label_paths": label_paths,
                    "website": entry.get("website", ""),
                })
            if limit and len(samples) >= limit:
                break
        return samples[:limit] if limit else samples

    def build_grouped_samples(
        self,
        limit: int | None = None,
        urls_per_entry: int | None = None,
        exclude_websites: list[str] | None = None,
        task_types: list[str] | None = None,
    ) -> list[dict]:
        """Build grouped samples following the paper's evaluation protocol.

        Paper (ICLR 2026, §5.1): "a set of XPaths is generated from the first
        web page in a group and then applied to all other pages within that group."

        Returns a list of group dicts, each containing:
            sample_id, query, attributes, task_type, label, label_paths, website,
            urls: list[str]  (first URL is used for XPath generation)
        """
        if exclude_websites is None:
            exclude_websites = self.EXCLUDED_WEBSITES
        excluded_set = set(exclude_websites)

        entries = self.flatten_dataset()
        type_filter = set(task_types) if task_types else None
        groups = []
        total_urls = 0
        for entry in entries:
            website_key = entry.get("website", "")
            if any(ex in website_key.lower() for ex in excluded_set):
                continue
            if type_filter and entry.get("type", "") not in type_filter:
                continue

            group_urls = self.load_group_urls(entry["group_url"], entry.get("group_id", ""))
            if not group_urls:
                continue
            picked_urls = group_urls[:urls_per_entry] if urls_per_entry else group_urls

            label = entry.get("label", {})
            label_paths = list(label.values()) if isinstance(label, dict) else label

            groups.append({
                "sample_id": entry.get("sample_id", entry.get("query_id")),
                "query": entry["query"],
                "attributes": entry.get("attribute", []),
                "task_type": entry.get("type", ""),
                "label": label,
                "label_paths": label_paths,
                "website": entry.get("website", ""),
                "urls": picked_urls,
            })
            total_urls += len(picked_urls)
            if limit and total_urls >= limit:
                # Trim last group's URLs if over limit
                excess = total_urls - limit
                if excess > 0:
                    groups[-1]["urls"] = groups[-1]["urls"][:-excess]
                break
        return groups
