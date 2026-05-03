#!/usr/bin/env node
/*
 * Postinstall: download the matching llm-speed binary from the GitHub release
 * for this package version, then unzip into ./bin/.
 *
 * The wrapper is intentionally thin: the canonical implementation is the Python
 * package on PyPI. npm is just a distribution channel for users who already
 * `npm install -g` everything.
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");
const { execSync } = require("child_process");

const pkg = require("./package.json");
const VERSION = pkg.version;

function platformTag() {
  const p = process.platform; // 'linux' | 'darwin' | 'win32'
  if (p === "linux" || p === "darwin" || p === "win32") return p;
  throw new Error(`unsupported platform: ${p}`);
}

function archTag() {
  const a = process.arch; // 'x64' | 'arm64' | ...
  if (a === "x64" || a === "arm64") return a;
  throw new Error(`unsupported arch: ${a}`);
}

function assetName(version, platform, arch) {
  return `llm-speed-${version}-${platform}-${arch}.zip`;
}

function downloadUrl(version, platform, arch) {
  return (
    `https://github.com/meadow-kun/llm-speed/releases/download/` +
    `v${version}/${assetName(version, platform, arch)}`
  );
}

function download(url, destPath, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error("too many redirects"));
    const file = fs.createWriteStream(destPath);
    https
      .get(url, (res) => {
        if (
          (res.statusCode === 301 || res.statusCode === 302) &&
          res.headers.location
        ) {
          file.close();
          fs.unlinkSync(destPath);
          return resolve(download(res.headers.location, destPath, redirects + 1));
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(destPath);
          return reject(
            new Error(`download failed: HTTP ${res.statusCode} for ${url}`)
          );
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
      })
      .on("error", (err) => {
        file.close();
        if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
        reject(err);
      });
  });
}

function unzip(zipPath, destDir) {
  // Use unzip (POSIX) or PowerShell Expand-Archive (Windows).
  if (process.platform === "win32") {
    execSync(
      `powershell -NoProfile -Command "Expand-Archive -Force -Path '${zipPath}' -DestinationPath '${destDir}'"`,
      { stdio: "inherit" }
    );
  } else {
    execSync(`unzip -o '${zipPath}' -d '${destDir}'`, { stdio: "inherit" });
  }
}

async function main() {
  const platform = platformTag();
  const arch = archTag();
  const url = downloadUrl(VERSION, platform, arch);

  const binDir = path.join(__dirname, "bin");
  if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });

  const tmpZip = path.join(os.tmpdir(), assetName(VERSION, platform, arch));

  console.log(`[llm-speed] downloading ${url}`);
  await download(url, tmpZip);

  console.log(`[llm-speed] extracting to ${binDir}`);
  unzip(tmpZip, binDir);

  // Pick the binary name written into bin/ by the zip; chmod +x on POSIX.
  const candidates = ["llm-speed", "llm-speed.exe"];
  for (const name of candidates) {
    const p = path.join(binDir, name);
    if (fs.existsSync(p) && process.platform !== "win32") {
      fs.chmodSync(p, 0o755);
    }
  }

  try {
    fs.unlinkSync(tmpZip);
  } catch (_) {
    /* ignore */
  }

  console.log("[llm-speed] install complete");
}

main().catch((err) => {
  console.error("[llm-speed] install failed:", err.message);
  console.error(
    "[llm-speed] you can install via pipx instead:  pipx install llm-speed"
  );
  // Soft-fail: do not break the npm install of consumers' lockfiles. The
  // bin/llm-speed.js stub will print a clear error if the binary is missing.
  process.exit(0);
});
