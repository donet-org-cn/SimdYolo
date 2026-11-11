"""Verify that public NuGet downloads contain the exact CI package contents."""

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def contents(package):
    with zipfile.ZipFile(package) as archive:
        return {entry.filename: hashlib.sha256(archive.read(entry)).digest()
                for entry in archive.infolist() if entry.filename != ".signature.p7s"}


def verify(directory, version):
    expected = {}
    for package_id in ("SimdYolo", "SimdYolo.Training"):
        package = directory / f"{package_id}.{version}.nupkg"
        lower_id = package_id.lower()
        url = f"https://api.nuget.org/v3-flatcontainer/{lower_id}/{version}/{package.name.lower()}"
        expected[url] = contents(package)

    deadline = time.monotonic() + 1200
    while expected:
        for url in list(expected):
            try:
                request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    package = response.read()
            except urllib.error.HTTPError as error:
                if error.code not in (404, 429, 500, 502, 503, 504):
                    raise
                continue
            except (urllib.error.URLError, TimeoutError):
                continue

            if contents(io.BytesIO(package)) != expected[url]:
                raise SystemExit(f"Public package contents differ from CI: {url}")
            del expected[url]
            print(f"Verified {url} (SHA-256 {hashlib.sha256(package).hexdigest()})", flush=True)
        if expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit("Timed out waiting for " + ", ".join(expected))
            time.sleep(min(20, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    arguments = parser.parse_args()
    verify(arguments.directory, arguments.version)
