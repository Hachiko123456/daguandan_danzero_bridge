from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_release.ps1"
PACKAGE_LAUNCHER = PROJECT_ROOT / "package_release.bat"
MARKER_NAME = ".daguandan-release-root"
MARKER_VALUE = "guandan.package-release-root/2"
METRICS_NAME = "build_metrics.json"
METRICS_VALUE = {
    "schema": "guandan.release-build-metrics/1",
    "status": "passed",
    "total_seconds": 0.01,
    "environment_cache_hit": False,
    "work_cache_hit": False,
    "stages": [],
}


pytestmark = pytest.mark.contract


_FUNCTION_LOADER = r"""
$ErrorActionPreference = "Stop"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:PACKAGE_SCRIPT,
    [ref] $tokens,
    [ref] $parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw (($parseErrors | ForEach-Object { $_.Message }) -join "`n")
}
$functionNodes = @(
    $ast.FindAll(
        { param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] },
        $true
    )
)
foreach ($node in $functionNodes) {
    . ([scriptblock]::Create($node.Extent.Text))
}
"""


def _powershell() -> str:
    if os.name != "nt":
        pytest.skip("Windows release path contracts require Windows PowerShell")
    configured = os.environ.get("DAGUANDAN_POWERSHELL")
    candidates = [configured] if configured else []
    candidates.extend(["powershell.exe", "pwsh"])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return str(Path(candidate))
    pytest.skip("A real PowerShell executable is required for packaging contract tests")


def _run_ps(ps: str, body: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PACKAGE_SCRIPT"] = str(PACKAGE_SCRIPT)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            ps,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            base64.b64encode((_FUNCTION_LOADER + "\n" + body).encode("utf-16-le")).decode("ascii"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )


def _assert_ps_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"PowerShell failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def _assert_ps_failure(result: subprocess.CompletedProcess[str], *patterns: str) -> None:
    assert result.returncode != 0, "PowerShell unexpectedly succeeded"
    output = f"{result.stdout}\n{result.stderr}"
    if patterns:
        assert any(re.search(pattern, output, re.IGNORECASE) for pattern in patterns), output


def _owned_release_root(root: Path, identity: str) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    (root / MARKER_NAME).write_text(MARKER_VALUE + "\n", encoding="utf-8")
    payload = root / "payload"
    payload.mkdir()
    (payload / f"{identity}.txt").write_text(identity, encoding="utf-8")
    (root / METRICS_NAME).write_text(
        json.dumps(METRICS_VALUE, indent=2) + "\n", encoding="utf-8"
    )
    return root


def _write_unmanaged(root: Path, name: str = "not-managed.txt") -> None:
    (root / name).write_text("must not be deleted", encoding="utf-8")


def _assert_owned_root(ps: str, root: Path) -> subprocess.CompletedProcess[str]:
    return _run_ps(
        ps,
        """
try {
    Assert-ManagedReleaseRoot -Root $env:ROOT
    Write-Output "owned-root-ok"
}
catch {
    Write-Error $_
    exit 17
}
""",
        extra_env={"ROOT": str(root)},
    )


def _command_facts(ps: str) -> list[dict[str, object]]:
    result = _run_ps(
        ps,
        r"""
$interesting = @(
    "Assert-ManagedReleaseRoot",
    "Publish-ManagedRelease",
    "Clear-ManagedReleaseRoot",
    "Compress-Archive",
    "Copy-Item",
    "Move-Item",
    "New-Item"
)
$commands = @(
    $ast.FindAll(
        { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
        $true
    ) |
    ForEach-Object {
        $name = $_.GetCommandName()
        if ($interesting -contains $name) {
            $owner = ''
            $parent = $_.Parent
            while ($null -ne $parent) {
                if ($parent -is [System.Management.Automation.Language.FunctionDefinitionAst]) { $owner = $parent.Name; break }
                $parent = $parent.Parent
            }
            [pscustomobject]@{
                name = $name
                owner = $owner
                text = $_.Extent.Text
                offset = $_.Extent.StartOffset
                line = $_.Extent.StartLineNumber
            }
        }
    }
)
$commands | ConvertTo-Json -Compress -Depth 8
""",
    )
    _assert_ps_success(result)
    if not result.stdout.strip():
        return []
    decoded = json.loads(result.stdout)
    return decoded if isinstance(decoded, list) else [decoded]


def _command_texts(facts: Iterable[dict[str, object]], name: str) -> list[str]:
    return [str(item["text"]) for item in facts if item["name"] == name]


def test_assert_managed_release_root_accepts_owned_allowlist_including_metrics(tmp_path: Path) -> None:
    ps = _powershell()
    root = _owned_release_root(tmp_path / "owned", "accepted")

    result = _assert_owned_root(ps, root)

    _assert_ps_success(result)
    assert "owned-root-ok" in result.stdout

    _write_unmanaged(root)
    result = _assert_owned_root(ps, root)
    _assert_ps_failure(result, "unmanaged", "refus", "allowlist")
    assert (root / "not-managed.txt").exists()


def test_assert_managed_release_root_rejects_reparse_point_tree(tmp_path: Path) -> None:
    ps = _powershell()
    root = _owned_release_root(tmp_path / "owned", "junction")
    target = tmp_path / "junction-target"
    target.mkdir()
    (target / "secret.txt").write_text("target", encoding="utf-8")
    junction = root / "payload" / "linked-target"

    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction), str(target)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"The Windows test environment cannot create a junction: {result.stdout} {result.stderr}")

    try:
        check = _assert_owned_root(ps, root)
        _assert_ps_failure(check, "reparse", "junction", "symlink", "link")
    finally:
        if junction.exists():
            junction.rmdir()


def test_publish_moves_current_to_previous_then_promotes_stage(tmp_path: Path) -> None:
    ps = _powershell()
    destination = tmp_path / "current"
    stage = tmp_path / "stage"
    _owned_release_root(destination, "old-current")
    _owned_release_root(stage, "new-stage")

    result = _run_ps(
        ps,
        """
try {
    Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION
    Write-Output "publish-ok"
}
catch {
    Write-Error $_
    exit 17
}
""",
        extra_env={"STAGE": str(stage), "DESTINATION": str(destination)},
    )

    _assert_ps_success(result)
    previous = destination.with_name(destination.name + ".previous")
    assert "publish-ok" in result.stdout
    assert (destination / "payload" / "new-stage.txt").read_text(encoding="utf-8") == "new-stage"
    assert (previous / "payload" / "old-current.txt").read_text(encoding="utf-8") == "old-current"
    assert not stage.exists()


def test_publish_failure_rolls_back_previous_and_retains_failed_stage(tmp_path: Path) -> None:
    ps = _powershell()
    destination = tmp_path / "current"
    stage = tmp_path / "stage"
    _owned_release_root(destination, "old-current")
    _owned_release_root(stage, "failed-stage")

    result = _run_ps(
        ps,
        r"""
function global:Move-Item {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string] $LiteralPath,
        [Parameter(Mandatory = $true)][string] $Destination,
        [switch] $Force,
        [switch] $PassThru
    )
    if (
        [System.IO.Path]::GetFullPath($LiteralPath).TrimEnd('\').Equals(
            [System.IO.Path]::GetFullPath($env:STAGE).TrimEnd('\'),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "injected stage promotion failure"
    }
    $arguments = @{
        LiteralPath = $LiteralPath
        Destination = $Destination
        Force = $Force
    }
    if ($PassThru) {
        $arguments.PassThru = $true
    }
    Microsoft.PowerShell.Management\Move-Item @arguments
}
try {
    Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION
    Write-Error "Publish unexpectedly succeeded"
    exit 19
}
catch {
    Write-Output $_.Exception.Message
    exit 17
}
""",
        extra_env={"STAGE": str(stage), "DESTINATION": str(destination)},
    )

    _assert_ps_failure(result, "injected", "promotion", "publish")
    assert (destination / "payload" / "old-current.txt").read_text(encoding="utf-8") == "old-current"
    assert (stage / "payload" / "failed-stage.txt").read_text(encoding="utf-8") == "failed-stage"
    assert not destination.with_name(destination.name + ".previous").exists()


def test_publish_rejects_unmanaged_current_without_mutation(tmp_path: Path) -> None:
    ps = _powershell()
    destination = tmp_path / "current"
    stage = tmp_path / "stage"
    _owned_release_root(destination, "current")
    _write_unmanaged(destination)
    _owned_release_root(stage, "stage")

    result = _run_ps(
        ps,
        """
try {
    Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION
    exit 19
}
catch {
    Write-Output $_.Exception.Message
    exit 17
}
""",
        extra_env={"STAGE": str(stage), "DESTINATION": str(destination)},
    )

    _assert_ps_failure(result, "unmanaged", "refus", "owned")
    assert (destination / "not-managed.txt").exists()
    assert (stage / "payload" / "stage.txt").exists()
    assert not destination.with_name(destination.name + ".previous").exists()


def test_publish_rejects_unmanaged_existing_previous_without_mutation(tmp_path: Path) -> None:
    ps = _powershell()
    destination = tmp_path / "current"
    stage = tmp_path / "stage"
    previous = destination.with_name(destination.name + ".previous")
    _owned_release_root(destination, "current")
    _owned_release_root(stage, "stage")
    _owned_release_root(previous, "previous")
    _write_unmanaged(previous, "previous-unmanaged.txt")

    result = _run_ps(
        ps,
        """
try {
    Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION
    exit 19
}
catch {
    Write-Output $_.Exception.Message
    exit 17
}
""",
        extra_env={"STAGE": str(stage), "DESTINATION": str(destination)},
    )

    _assert_ps_failure(result, "unmanaged", "refus", "owned")
    assert (destination / "payload" / "current.txt").exists()
    assert (previous / "previous-unmanaged.txt").exists()
    assert (stage / "payload" / "stage.txt").exists()


def test_clear_managed_release_root_can_clean_only_an_owned_stage_target(tmp_path: Path) -> None:
    ps = _powershell()
    stage = _owned_release_root(tmp_path / "stage", "cleanable")
    (stage / "build").mkdir()
    (stage / "build" / "intermediate.bin").write_bytes(b"intermediate")

    result = _run_ps(
        ps,
        """
try {
    Clear-ManagedReleaseRoot -Root $env:STAGE -Overwrite $true
    Write-Output "clear-ok"
}
catch {
    Write-Error $_
    exit 17
}
""",
        extra_env={"STAGE": str(stage)},
    )

    _assert_ps_success(result)
    assert "clear-ok" in result.stdout
    assert (stage / MARKER_NAME).exists()
    assert not (stage / "build").exists()
    assert not (stage / "payload").exists()


def test_same_and_nested_roots_are_rejected_by_real_powershell_function(tmp_path: Path) -> None:
    ps = _powershell()
    root = tmp_path / "root"
    nested = root / "nested"
    root.mkdir()
    nested.mkdir()

    for first, second in ((root, root), (root, nested), (nested, root)):
        result = _run_ps(
            ps,
            """
try {
    Assert-DisjointRoots -First $env:FIRST -Second $env:SECOND -Description "test roots"
    exit 19
}
catch {
    Write-Output $_.Exception.Message
    exit 17
}
""",
            extra_env={"FIRST": str(first), "SECOND": str(second)},
        )
        _assert_ps_failure(result, "disjoint", "same", "nested", "below")


def test_release_script_checks_absolute_filesystem_root_and_disjoint_external_roots() -> None:
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert "[System.IO.Path]::GetFullPath" in source
    assert re.search(r"\[(?:System\.)?IO\.Path\]::GetPathRoot", source)
    assert re.search(r"filesystem\s+root", source, re.IGNORECASE)
    assert "Assert-DisjointRoots" in source
    assert "ReleaseRoot and WheelhouseRoot" in source
    assert "ReleaseRoot and project root" in source


def test_release_lifecycle_verifies_fresh_stage_before_archive_and_publishes_after_archive() -> None:
    ps = _powershell()
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    facts = _command_facts(ps)

    assert re.search(r"(?i)(fresh|new|staging|stage).{0,160}(release|root|directory)", source, re.DOTALL)
    assert ".cache\\release" in source or ".cache/release" in source
    assert "release_build_cache.py" in source

    main_facts = [item for item in facts if not item["owner"]]
    archives = [item for item in main_facts if item["name"] == "Compress-Archive"]
    publishes = [item for item in main_facts if item["name"] == "Publish-ManagedRelease"]
    assert len(archives) == len(publishes) == 1
    assert int(archives[0]["offset"]) < int(publishes[0]["offset"])
    assert source.index("[Guid]::NewGuid()") < source.index('"--distpath", $distPath')
    assert source.index("@manifestVerifyArguments") < int(archives[0]["offset"])
    assert source.index("--checksum $archiveChecksumPath") < int(publishes[0]["offset"])
    assert source.rindex("@sourceIdentityVerifyArguments") < int(publishes[0]["offset"])
    assert source.index("-Status 'validated'") < int(publishes[0]["offset"]) < source.index("-Status 'passed'")
    assert "Write-BuildMetrics -Root $stagingReleaseRoot -Status 'failed'" in source
    # dist is always inside this run's stage, never relocated to a cache hit.
    assert source.count("$distPath =") == 1
    assert 'Resolve-ManagedChildPath -Root $stagingReleaseRoot -Child (Join-Path $stagingReleaseRoot "dist")' in source


def test_clear_calls_never_target_current_release_root() -> None:
    ps = _powershell()
    facts = _command_facts(ps)
    clear_calls = _command_texts(facts, "Clear-ManagedReleaseRoot")
    assert clear_calls, "The release script must retain Clear-ManagedReleaseRoot for disposable roots"
    assert not any(
        re.search(r"(?i)-Root\s+\$(?:current|releaseRoot|destination)\b", text)
        for text in clear_calls
    ), "The build must not clear current before publishing"
    assert any(re.search(r"(?i)\$(?:previous|stage|staging)", text) for text in clear_calls)


def test_fast_path_uses_release_cache_and_never_copies_old_dist_into_current() -> None:
    ps = _powershell()
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    facts = _command_facts(ps)
    copy_calls = _command_texts(facts, "Copy-Item")

    assert ".cache\\release" in source or ".cache/release" in source
    assert "release_build_cache.py" in source
    assert "environment_cache_hit" in source
    assert "work_cache_hit" in source
    assert not any(
        re.search(r"(?i)\$(?:finalReleaseRoot|previousRoot|previousPath|distPath)\b", text)
        for text in copy_calls
    ), "Build output must be published from a fresh stage, not copied into current"


@pytest.mark.parametrize("status", ["validated", "passed", "failed"])
def test_real_build_stage_functions_write_typed_metrics(tmp_path: Path, status: str) -> None:
    root = _owned_release_root(tmp_path / "stage", "metrics")
    result = _run_ps(_powershell(), r"""
$script:buildWatch = [Diagnostics.Stopwatch]::StartNew()
$script:buildStarted = [DateTime]::UtcNow.ToString('o')
$script:stageTimings = New-Object 'System.Collections.Generic.List[object]'
$script:stageWatch = $null
$script:environmentCacheHit = $true
$script:workCacheHit = $false
$script:cacheEnabled = $true
Start-BuildStage 1 'validate fixture'
Start-Sleep -Milliseconds 15
Start-BuildStage 2 'publish fixture'
Start-Sleep -Milliseconds 15
Complete-BuildStage
Complete-BuildStage
Write-BuildMetrics -Root $env:ROOT -Status $env:STATUS -Failure $(if ($env:STATUS -eq 'failed') {'injected error'} else {''})
""", extra_env={"ROOT": str(root), "STATUS": status})
    _assert_ps_success(result)
    metrics = json.loads((root / METRICS_NAME).read_text(encoding="utf-8-sig"))
    assert metrics["schema"] == "guandan.release-build-metrics/1"
    assert metrics["status"] == status
    assert isinstance(metrics["total_seconds"], (float, int))
    assert metrics["total_seconds"] >= 0
    assert metrics["environment_cache_hit"] is True
    assert metrics["work_cache_hit"] is False
    assert [stage["name"] for stage in metrics["stages"]] == ["validate fixture", "publish fixture"]
    assert all(isinstance(stage["seconds"], (float, int)) and stage["seconds"] >= 0 for stage in metrics["stages"])
    assert sum(stage["seconds"] for stage in metrics["stages"]) <= metrics["total_seconds"] + 0.01
    assert metrics["error"] == ("injected error" if status == "failed" else "")


def test_formal_build_path_keeps_dirty_tree_guard_and_dirty_mode_is_explicit() -> None:
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert "[switch] $AllowDirtyDevelopmentBuild" in source
    assert re.search(
        r"\$sourceTreeDirty\s+-and\s+-not\s+\$AllowDirtyDevelopmentBuild",
        source,
        re.IGNORECASE,
    )
    assert "Source tree is dirty" in source
    assert re.search(r"(?i)strict", source)


def test_package_release_bat_default_contract_has_one_entry_and_default_current_output() -> None:
    launcher = PACKAGE_LAUNCHER.read_text(encoding="utf-8")
    assert 'if "%~1"==""' in launcher
    assert r"release\current" in launcher
    assert "scripts\\package_release.ps1" in launcher
    assert "Usage: package_release.bat [RELEASE_ROOT [WHEELHOUSE_ROOT]]" in launcher
    assert len(list(PROJECT_ROOT.glob("package_release*.bat"))) == 1


def test_publish_rejects_filesystem_root_before_mutation(tmp_path: Path) -> None:
    stage = _owned_release_root(tmp_path / "stage", "root-guard")
    result = _run_ps(_powershell(), r"""
# Guard both mutation cmdlets; a regression must not touch the actual drive root.
function Move-Item { throw 'UNSAFE MUTATION REACHED' }
function Remove-Item { throw 'UNSAFE MUTATION REACHED' }
try { Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION; exit 19 }
catch { Write-Output $_.Exception.Message; exit 17 }
""", extra_env={"STAGE": str(stage), "DESTINATION": stage.anchor})
    assert result.returncode == 17, result.stdout + result.stderr
    assert "filesystem root" in result.stdout.lower()
    assert "UNSAFE MUTATION" not in result.stdout
    assert (stage / "payload" / "root-guard.txt").exists()



@pytest.mark.parametrize("relation", ["same", "stage-under-current", "current-under-stage", "normalized-same", "stage-is-previous"])
def test_publish_rejects_same_nested_and_aliased_roots_without_mutation(tmp_path: Path, relation: str) -> None:
    current = _owned_release_root(tmp_path / "current", "preserve")
    stage = current
    if relation == "stage-under-current":
        stage = _owned_release_root(current / "payload" / "stage", "stage")
    elif relation == "current-under-stage":
        stage = current
        current = _owned_release_root(stage / "payload" / "current", "nested")
    elif relation == "normalized-same":
        stage = current / "payload" / ".."
    elif relation == "stage-is-previous":
        stage = _owned_release_root(tmp_path / "current.previous", "stage")
    before = {str(f.relative_to(tmp_path)): f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}
    result = _run_ps(_powershell(), r"""
try { Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION; exit 19 }
catch { Write-Output $_.Exception.Message; exit 17 }
""", extra_env={"STAGE": str(stage), "DESTINATION": str(current)})
    assert result.returncode == 17, result.stdout + result.stderr
    assert "disjoint" in result.stdout.lower()
    assert before == {str(f.relative_to(tmp_path)): f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}


def test_publish_cleans_only_owned_previous_and_preserves_unrelated_sibling(tmp_path: Path) -> None:
    current = _owned_release_root(tmp_path / "current", "current")
    stage = _owned_release_root(tmp_path / "stage", "stage")
    previous = _owned_release_root(tmp_path / "current.previous", "stale")
    unrelated = _owned_release_root(tmp_path / "current.previous-other", "untouched")
    result = _run_ps(_powershell(), 'Publish-ManagedRelease -Stage $env:STAGE -Destination $env:DESTINATION',
                     extra_env={"STAGE": str(stage), "DESTINATION": str(current)})
    _assert_ps_success(result)
    assert (current / "payload" / "stage.txt").exists()
    assert (previous / "payload" / "current.txt").exists()
    assert not (previous / "payload" / "stale.txt").exists()
    assert (unrelated / "payload" / "untouched.txt").read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("cache_hit", [False, True])
def test_actual_environment_branch_installs_only_on_cache_miss(tmp_path: Path, cache_hit: bool) -> None:
    fake_python = tmp_path / "env" / "Scripts" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(b"fixture only; never executed")
    result = _run_ps(_powershell(), r"""
$branch = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Clauses[0].Item1.Extent.Text -eq '-not $script:environmentCacheHit' -and
    $node.Extent.Text -match 'venv'
}, $true))
if ($branch.Count -ne 1) { throw 'Expected one environment creation branch' }
$script:environmentCacheHit = $env:HIT -eq 'true'
$buildEnvPath = $env:ENVROOT
$bootstrapPython = 'fixture-bootstrap.exe'
$projectRoot = $env:ENVROOT
$wheelhouseRoot = $env:ENVROOT
$script:calls = New-Object 'System.Collections.Generic.List[string]'
function Invoke-CleanPython { $flat = @($args | ForEach-Object { if ($_ -is [array]) { $_ | ForEach-Object { [string]$_ } } else { [string]$_ } }); $script:calls.Add(($flat -join ' ')) }
. ([scriptblock]::Create($branch[0].Extent.Text))
Write-Output ('RESULT:' + (ConvertTo-Json -InputObject @($script:calls.ToArray()) -Compress))
""", extra_env={"HIT": str(cache_hit).lower(), "ENVROOT": str(fake_python.parent.parent)})
    _assert_ps_success(result)
    calls = json.loads(next(line[len("RESULT:"):] for line in result.stdout.splitlines() if line.startswith("RESULT:")))
    if cache_hit:
        assert calls == [], "A validated environment must not recreate venv or reinstall wheels"
    else:
        assert len(calls) == 2
        assert "venv" in calls[0] and "pip install" in calls[1]
        assert "--no-index" in calls[1] and "--require-hashes" in calls[1]


@pytest.mark.parametrize("relationship", ["equal", "wheelhouse-inside-final"])
def test_main_root_guard_checks_final_output_not_just_staging(tmp_path: Path, relationship: str) -> None:
    current = tmp_path / "current"
    wheelhouse = current if relationship == "equal" else current / "payload" / "wheelhouse"
    result = _run_ps(_powershell(), r"""
$finalReleaseRoot = $env:CURRENT
$stagingReleaseRoot = $env:STAGE
$releaseRoot = $stagingReleaseRoot
$wheelhouseRoot = $env:WHEELHOUSE
$projectRoot = $env:PROJECT
$guards = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.CommandAst] -and
    $node.GetCommandName() -eq 'Assert-DisjointRoots' -and
    $node.Extent.Text -match 'ReleaseRoot and WheelhouseRoot'
}, $true))
if ($guards.Count -eq 0) { throw 'Missing release/wheelhouse root guard' }
try {
    foreach ($guard in $guards) { . ([scriptblock]::Create($guard.Extent.Text)) }
    Write-Output 'UNSAFE: final output overlaps wheelhouse but passed the real guard'
    exit 19
} catch { Write-Output $_.Exception.Message; exit 17 }
""", extra_env={"CURRENT": str(current), "STAGE": str(tmp_path / ".staging" / "current-run"),
                "WHEELHOUSE": str(wheelhouse), "PROJECT": str(tmp_path / "source")})
    assert result.returncode == 17, result.stdout + result.stderr
    assert "disjoint" in result.stdout.lower()


def test_self_extracting_rlcard_data_is_materialized_before_module_collection():
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    prepare = source.index(' -c "import rlcard.envs"')
    assert source.index("--verify-installed") < prepare < source.index('"--collect-data", "rlcard"')
    assert "_input_datas" in source
