"""Resolving a GitHub URL into a local checkout.

Accepts the forms people actually paste: full URLs, `owner/repo`, SSH remotes, branch
links, PR links, or a local path. PR links are recognised here rather than behind a flag
-- making the user describe a URL they already handed us would be a poor interface.

Clones are shallow and blobless: we only need the tree at one commit.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_GITHUB_URL = re.compile(
    r"""^(?:https?://)?(?:www\.)?(?P<host>github\.com|gitlab\.com)/
        (?P<owner>[^/\s]+)/(?P<name>[^/\s#?]+?)(?:\.git)?
        (?:/(?:tree|blob)/(?P<ref>[^/\s#?]+))?
        (?:/pull/(?P<pr>\d+))?
        /?(?:[#?].*)?$""",
    re.VERBOSE,
)
_SSH_URL = re.compile(r"^git@(?P<host>[^:]+):(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?$")
_SHORTHAND = re.compile(r"^(?P<owner>[A-Za-z0-9._-]+)/(?P<name>[A-Za-z0-9._-]+)$")


class RepoError(RuntimeError):
    """Anything that stops us getting a usable checkout."""


@dataclass(frozen=True)
class RepoRef:
    host: str
    owner: str
    name: str
    ref: str | None = None
    pr_number: int | None = None
    local_path: Path | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://{self.host}/{self.owner}/{self.name}.git"

    @property
    def is_pr(self) -> bool:
        return self.pr_number is not None

    @property
    def is_local(self) -> bool:
        return self.local_path is not None

    def describe(self) -> str:
        parts = [self.slug]
        if self.pr_number:
            parts.append(f"PR #{self.pr_number}")
        elif self.ref:
            parts.append(f"@{self.ref}")
        return " ".join(parts)

    @classmethod
    def parse(cls, target: str) -> RepoRef:
        target = target.strip()

        candidate = Path(target).expanduser()
        if candidate.exists() and candidate.is_dir():
            resolved = candidate.resolve()
            return cls(
                host="local",
                owner=resolved.parent.name or "local",
                name=resolved.name,
                local_path=resolved,
            )

        if m := _GITHUB_URL.match(target):
            pr = m.group("pr")
            return cls(
                host=m.group("host"),
                owner=m.group("owner"),
                name=m.group("name"),
                ref=m.group("ref"),
                pr_number=int(pr) if pr else None,
            )

        if m := _SSH_URL.match(target):
            return cls(host=m.group("host"), owner=m.group("owner"), name=m.group("name"))

        if m := _SHORTHAND.match(target):
            return cls(host="github.com", owner=m.group("owner"), name=m.group("name"))

        raise RepoError(
            f"could not interpret {target!r} as a repository. Expected a GitHub URL, "
            f"'owner/repo', or a path to a local clone."
        )


@dataclass
class Checkout:
    """A repository on disk at a known commit."""

    ref: RepoRef
    root: Path
    commit: str
    default_branch: str | None = None

    @property
    def short_commit(self) -> str:
        return self.commit[:8]


def _git(*args: str, cwd: Path | None = None, timeout: int = 300) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RepoError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RepoError(f"git {' '.join(args[:2])} timed out after {timeout}s") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RepoError(f"git {' '.join(args[:2])} failed: {detail[-1] if detail else '?'}")
    return result.stdout.strip()


def acquire(ref: RepoRef, work_dir: Path, *, refresh: bool = False) -> Checkout:
    """Get `ref` onto local disk and return a `Checkout`.

    A local path is used in place rather than copied -- reviewing a working tree you
    already have is the fastest way to iterate, and copying it would only invite
    confusion about which copy the line numbers refer to.
    """
    if ref.local_path is not None:
        root = ref.local_path
        if not (root / ".git").exists():
            # A plain directory is still reviewable -- an extracted archive, a vendored
            # copy, a working tree without history. Only the churn signal is lost, and
            # the risk scorer already handles that signal being absent.
            log.info("%s is not a git repository; churn signal unavailable", root)
            return Checkout(ref=ref, root=root, commit="0" * 40)
        return Checkout(
            ref=ref,
            root=root,
            commit=_git("rev-parse", "HEAD", cwd=root),
            default_branch=_current_branch(root),
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    root = work_dir / f"{ref.owner}__{ref.name}"

    if root.exists() and refresh:
        shutil.rmtree(root)

    if not root.exists():
        # blob:none keeps history metadata (needed for churn) but skips file contents
        # for commits we never check out.
        args = ["clone", "--filter=blob:none", "--depth", "50"]
        if ref.ref and not ref.is_pr:
            args += ["--branch", ref.ref]
        _git(*args, ref.clone_url, str(root))

    if ref.is_pr:
        # GitHub exposes PR heads as a ref; fetching it directly avoids needing the API.
        _git("fetch", "origin", f"pull/{ref.pr_number}/head:codesage-pr", cwd=root)
        _git("checkout", "codesage-pr", cwd=root)
    elif ref.ref:
        _git("checkout", ref.ref, cwd=root)

    return Checkout(
        ref=ref,
        root=root,
        commit=_git("rev-parse", "HEAD", cwd=root),
        default_branch=_current_branch(root),
    )


def _current_branch(root: Path) -> str | None:
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
        return None if branch == "HEAD" else branch
    except RepoError:
        return None
