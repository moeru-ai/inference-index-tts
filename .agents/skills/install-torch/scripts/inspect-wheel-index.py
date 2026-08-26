#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

PACKAGES = ("torch", "torchvision", "torchaudio")
MAX_INDEX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class WheelCandidate:
    filename: str
    public_version: str
    local_version: str | None
    python_tag: str
    abi_tag: str
    platform_tag: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href is not None:
            self.links.append(href)


def python_tag(version: str) -> str:
    if re.fullmatch(r"3\.\d+", version) is None:
        raise ValueError("--python-version must look like 3.13")
    return f"cp{version.replace('.', '')}"


def platform_matches(tag: str, platform_name: str) -> bool:
    tags = tag.split(".")
    if platform_name == "linux-64":
        return any(
            re.fullmatch(r"(?:linux_x86_64|manylinux(?:1|2010|2014|_\d+_\d+)_x86_64)", item) is not None
            for item in tags
        )
    if platform_name == "win-64":
        return "win_amd64" in tags
    if platform_name == "osx-arm64":
        return any(re.fullmatch(r"macosx_\d+_\d+_(?:arm64|universal2)", item) is not None for item in tags)
    raise ValueError(f"unsupported platform: {platform_name}")


def split_version(version: str) -> tuple[str, str | None]:
    public, separator, local = version.partition("+")
    if not public or (separator and not local):
        raise ValueError(f"invalid version: {version}")
    normalized_local = re.sub(r"[-_.]+", ".", local.lower()) if separator else None
    return public, normalized_local


def official_channel(index_url: str) -> str | None:
    channel = unquote(urlsplit(index_url).path).rstrip("/").rsplit("/", maxsplit=1)[-1].lower()
    if re.fullmatch(r"(?:cpu|cu\d+|rocm\d+(?:[-_.]\d+)*|xpu(?:\d+(?:[-_.]\d+)*)?)", channel) is None:
        return None
    return re.sub(r"[-_.]+", ".", channel)


def requested_local_version(versions: dict[str, str]) -> str | None:
    local_versions = {local for version in versions.values() if (local := split_version(version)[1]) is not None}
    if len(local_versions) > 1:
        raise ValueError("conflicting local version labels across the PyTorch trio")
    return next(iter(local_versions), None)


def effective_target_local(versions: dict[str, str], index_url: str) -> str | None:
    requested_local = requested_local_version(versions)
    index_channel = official_channel(index_url)
    if requested_local is not None and index_channel is not None and requested_local != index_channel:
        raise ValueError(f"requested local version {requested_local} conflicts with index channel {index_channel}")
    return requested_local or index_channel


def version_matches(candidate: WheelCandidate, requested: str, target_local: str | None) -> bool:
    requested_public, requested_local = split_version(requested)
    if candidate.public_version != requested_public:
        return False
    if requested_local is not None and candidate.local_version != requested_local:
        return False
    return target_local is None or candidate.local_version is None or candidate.local_version == target_local


def python_abi_matches(wheel_python_tag: str, wheel_abi_tag: str, requested_python_tag: str) -> bool:
    compatible_abi_tags = {requested_python_tag, "abi3", "none"}
    return any(
        python_tag == requested_python_tag and abi_tag in compatible_abi_tags
        for python_tag in wheel_python_tag.split(".")
        for abi_tag in wheel_abi_tag.split(".")
    )


def without_build_tag(version: str) -> str:
    candidate_version, separator, build_tag = version.rpartition("-")
    if separator and re.fullmatch(r"\d[^-]*", build_tag) is not None:
        return candidate_version
    return version


def parse_wheel_candidate(href: str, package: str) -> WheelCandidate | None:
    filename = unquote(Path(urlsplit(href).path).name)
    if not filename.endswith(".whl"):
        return None
    try:
        distribution_version, wheel_python_tag, wheel_abi_tag, wheel_platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return None
    prefix = f"{package.replace('-', '_')}-"
    if not distribution_version.startswith(prefix):
        return None
    candidate_version = without_build_tag(distribution_version[len(prefix) :])
    try:
        public_version, local_version = split_version(candidate_version)
    except ValueError:
        return None
    return WheelCandidate(
        filename=filename,
        public_version=public_version,
        local_version=local_version,
        python_tag=wheel_python_tag,
        abi_tag=wheel_abi_tag,
        platform_tag=wheel_platform_tag,
    )


def matching_wheels(
    html: str,
    package: str,
    requested_version: str,
    requested_python_tag: str,
    platform_name: str,
    target_local: str | None,
) -> list[WheelCandidate]:
    parser = LinkParser()
    parser.feed(html)
    matches: list[WheelCandidate] = []

    for href in parser.links:
        candidate = parse_wheel_candidate(href, package)
        if candidate is None:
            continue
        if not python_abi_matches(candidate.python_tag, candidate.abi_tag, requested_python_tag):
            continue
        if not version_matches(candidate, requested_version, target_local):
            continue
        if platform_matches(candidate.platform_tag, platform_name):
            matches.append(candidate)

    return sorted(set(matches), key=lambda candidate: candidate.filename)


def resolve_consistent_channels(
    matches: dict[str, list[WheelCandidate]],
) -> tuple[dict[str, list[WheelCandidate]], str | None, str | None]:
    if any(not package_matches for package_matches in matches.values()):
        return matches, None, None

    candidate_channels = {
        candidate.local_version
        for package_matches in matches.values()
        for candidate in package_matches
        if candidate.local_version is not None
    }
    common_channels = {
        channel
        for channel in candidate_channels
        if all(
            any(candidate.local_version in {None, channel} for candidate in package_matches)
            for package_matches in matches.values()
        )
    }
    all_packages_have_unlabelled = all(
        any(candidate.local_version is None for candidate in package_matches) for package_matches in matches.values()
    )
    if len(common_channels) > 1:
        channels = ", ".join(sorted(common_channels))
        raise ValueError(f"multiple compatible local version channels ({channels}); specify one explicit +local target")
    if common_channels:
        selected_channel = next(iter(common_channels))
        filtered_matches = {
            package: [
                candidate
                for candidate in package_matches
                if candidate.local_version is None or candidate.local_version == selected_channel
            ]
            for package, package_matches in matches.items()
        }
        return filtered_matches, selected_channel, None
    if all_packages_have_unlabelled:
        unlabelled_matches = {
            package: [candidate for candidate in package_matches if candidate.local_version is None]
            for package, package_matches in matches.items()
        }
        return unlabelled_matches, None, None
    return matches, None, "no common local version channel across the PyTorch trio candidates"


def decode_index(data: bytes) -> str:
    if len(data) > MAX_INDEX_BYTES:
        raise ValueError(f"index response exceeds {MAX_INDEX_BYTES} bytes")
    return data.decode("utf-8")


def read_index(index_url: str, package: str, html_dir: Path | None) -> str:
    if html_dir is not None:
        with (html_dir / f"{package}.html").open("rb") as index_file:
            return decode_index(index_file.read(MAX_INDEX_BYTES + 1))
    package_url = urljoin(f"{index_url.rstrip('/')}/", f"{package}/")
    request = Request(package_url, headers={"User-Agent": "pixi-install-torch-skill/1"})
    with urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        expected_bytes = int(content_length) if content_length is not None else None
        if expected_bytes is not None and expected_bytes > MAX_INDEX_BYTES:
            raise ValueError(f"index response exceeds {MAX_INDEX_BYTES} bytes")
        data = response.read(MAX_INDEX_BYTES + 1)
        if expected_bytes is not None and len(data) < expected_bytes:
            raise IncompleteRead(data, expected_bytes - len(data))
        return decode_index(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a PyTorch trio against wheel indexes.",
        epilog=(
            "Network reads use a 30-second socket inactivity timeout and a 16 MiB response limit; "
            "this is not a wall-clock deadline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--platform", required=True, choices=("linux-64", "win-64", "osx-arm64"))
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torchvision-version", required=True)
    parser.add_argument("--torchaudio-version", required=True)
    parser.add_argument("--html-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    versions = {
        "torch": args.torch_version,
        "torchvision": args.torchvision_version,
        "torchaudio": args.torchaudio_version,
    }
    try:
        requested_python_tag = python_tag(args.python_version)
        target_local = effective_target_local(versions, args.index_url)
        candidate_matches = {
            package: matching_wheels(
                read_index(args.index_url, package, args.html_dir),
                package,
                versions[package],
                requested_python_tag,
                args.platform,
                target_local,
            )
            for package in PACKAGES
        }
        selected_channel = target_local
        channel_issue = None
        if target_local is None:
            candidate_matches, selected_channel, channel_issue = resolve_consistent_channels(candidate_matches)
    except (HTTPException, OSError, UnicodeError, URLError, ValueError) as error:
        print(f"index inspection failed: {error}", file=sys.stderr)
        return 2

    matches = {
        package: [candidate.filename for candidate in package_matches]
        for package, package_matches in candidate_matches.items()
    }
    missing = sorted(package for package, wheels in matches.items() if not wheels)
    compatible = not missing and channel_issue is None
    payload = {
        "compatible": compatible,
        "index_url": args.index_url,
        "python_tag": requested_python_tag,
        "platform": args.platform,
        "versions": versions,
        "matches": matches,
        "missing": missing,
        "selected_channel": selected_channel,
        "channel_issue": channel_issue,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"compatible: {payload['compatible']}")
        if channel_issue is not None:
            print(f"channel incompatibility: {channel_issue}")
        for package in PACKAGES:
            print(f"{package}: {', '.join(matches[package]) or 'MISSING'}")
    return 0 if compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
