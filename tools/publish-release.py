"""以 UTF-8 文件发布 GitHub Release；凭据只在内存中使用，已有不同附件不会被覆盖。"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


def credential(owner):
    """读取用户已配置的 Git 凭据；禁止交互和输出令牌。"""
    env = os.environ | {"GCM_INTERACTIVE": "never", "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(["git", "-c", "safe.directory=F:/simdyolo", "-c", "credential.interactive=never", "credential", "fill"],
                            input=f"protocol=https\nhost=github.com\nusername={owner}\n\n", capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError("无法读取已有 GitHub 凭据。")
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not values.get("password"):
        raise RuntimeError("已有凭据没有访问令牌。")
    return values.get("username", owner), values["password"]


def digest(path):
    """以有界内存计算附件 SHA256。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GitHub:
    """仅向 GitHub API/上传域名发送当前用户的内存凭据。"""
    def __init__(self, repository):
        self.repository = repository
        owner = repository.split("/")[0]
        username, token = credential(owner)
        self.authorization = "Basic " + base64.b64encode((username + ":" + token).encode()).decode()
        user = self.request("/user")
        if user["login"].lower() != owner.lower():
            raise RuntimeError("当前凭据不是目标仓库所有者，停止发布。")

    def request(self, path, method="GET", value=None, file=None):
        """发送结构化 UTF-8 JSON 或附件流，并读取原始 JSON 响应。"""
        url = path if path.startswith("https://") else "https://api.github.com" + path
        if urllib.parse.urlparse(url).hostname not in {"api.github.com", "uploads.github.com"}:
            raise ValueError("目标不是允许的 GitHub API 域名。")
        # 发布后的立即回读必须绕开中间代理缓存，避免读到创建草稿之前的列表。
        if method == "GET": url += ("&" if "?" in url else "?") + "_release_check=" + str(time.time_ns())
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "SimdYolo-release", "X-GitHub-Api-Version": "2022-11-28", "Cache-Control": "no-cache", "Authorization": self.authorization}
        data = None if value is None else json.dumps(value, ensure_ascii=False).encode("utf-8")
        if value is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if file is not None:
            headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(file.stat().st_size)})
            with file.open("rb") as stream:
                return self._send(url, method, headers, stream)
        return self._send(url, method, headers, data)

    @staticmethod
    def _send(url, method, headers, data):
        """隔离网络错误；只显示状态和服务端说明，不输出请求头。"""
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            message = json.loads(error.read().decode("utf-8")).get("message", "request failed")
            raise RuntimeError(f"GitHub HTTP {error.code}: {message}") from None

    def release(self, tag):
        """同时查找公开和草稿版本，避免在草稿已存在时重复创建。"""
        for page in range(1, 11):
            releases = self.request(f"/repos/{self.repository}/releases?per_page=100&page={page}")
            found = next((item for item in releases if item["tag_name"] == tag), None)
            if found is not None or len(releases) < 100:
                return found
        raise RuntimeError("Release 数量超出检查范围。")


def main():
    """先准备草稿和附件，最后显式发布；修改旧发布说明时保留已有附件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["inspect", "draft", "notes", "upload", "publish"])
    parser.add_argument("--repo", default="hnlyf168/SimdYolo")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--release-id", type=int, help="GitHub草稿可能使用临时tag，可用明确ID继续操作同一草稿")
    parser.add_argument("--asset", type=Path, action="append", default=[])
    args = parser.parse_args()
    github = GitHub(args.repo)
    state_path = Path(__file__).resolve().parent.parent / ".cache/release-state" / (args.repo.replace("/", "-") + "-" + args.tag + ".json")
    release = github.request(f"/repos/{args.repo}/releases/{args.release_id}") if args.release_id else github.release(args.tag)
    if release is None and state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved["repository"] == args.repo and saved["tag"] == args.tag:
            release = github.request(f"/repos/{args.repo}/releases/{saved['id']}")
    if release is not None and not release["draft"] and release["tag_name"] != args.tag:
        raise RuntimeError("指定ID属于其他已发布版本，拒绝更改其tag。")
    body = None
    if args.notes:
        body = args.notes.read_text(encoding="utf-8-sig")
        if "\ufffd" in body or "????" in body or not any("\u4e00" <= c <= "\u9fff" for c in body):
            raise ValueError("中文发布说明编码异常，拒绝上传。")
    if args.operation in {"draft", "notes"}:
        if body is None:
            parser.error("需要 --notes UTF-8 文件。")
        title = body.splitlines()[0].lstrip("# ")
        if release is None:
            if args.operation != "draft" or not args.target:
                parser.error("创建草稿需要 --target 提交 SHA。")
            release = github.request(f"/repos/{args.repo}/releases", "POST", {"tag_name": args.tag, "target_commitish": args.target, "name": title, "body": body, "draft": True, "prerelease": False})
        else:
            changes = {"name": title, "body": body, "tag_name": args.tag}
            if args.target and release["draft"]: changes["target_commitish"] = args.target
            release = github.request(f"/repos/{args.repo}/releases/{release['id']}", "PATCH", changes)
        checked = github.request(f"/repos/{args.repo}/releases/{release['id']}")
        if checked is None or checked["body"] != body:
            raise RuntimeError("GitHub 回读的中文内容与本地 UTF-8 文件不同。")
    if release is None:
        raise RuntimeError("目标 Release 不存在。")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"repository": args.repo, "tag": args.tag, "id": release["id"]}), encoding="utf-8")
    if args.operation == "upload":
        for asset in args.asset:
            if not asset.is_file():
                raise FileNotFoundError(asset)
            sha = digest(asset)
            existing = next((item for item in release["assets"] if item["name"] == asset.name), None)
            if existing:
                if existing.get("digest") != "sha256:" + sha:
                    raise RuntimeError("已有同名但内容不同的附件，未覆盖：" + asset.name)
                print(json.dumps({"alreadyUploaded": asset.name, "sha256": sha}), flush=True)
                continue
            url = release["upload_url"].split("{", 1)[0] + "?name=" + urllib.parse.quote(asset.name)
            print(json.dumps({"uploading": asset.name, "bytes": asset.stat().st_size}), flush=True)
            uploaded = github.request(url, "POST", file=asset)
            if uploaded.get("digest") != "sha256:" + sha or uploaded["size"] != asset.stat().st_size:
                raise RuntimeError("GitHub 附件摘要或长度不同。")
            print(json.dumps({"uploaded": asset.name, "sha256": sha, "url": uploaded["browser_download_url"]}), flush=True)
        release = github.request(f"/repos/{args.repo}/releases/{release['id']}")
    if args.operation == "publish":
        if not release["assets"]:
            raise RuntimeError("没有附件，未发布空 Release。")
        changes = {"tag_name": args.tag, "draft": False, "prerelease": False, "make_latest": "true"}
        if args.target and release["draft"]: changes["target_commitish"] = args.target
        release = github.request(f"/repos/{args.repo}/releases/{release['id']}", "PATCH", changes)
    print(json.dumps({"tag": args.tag, "id": release["id"], "draft": release["draft"], "url": release["html_url"], "assets": [item["name"] for item in release["assets"]]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
