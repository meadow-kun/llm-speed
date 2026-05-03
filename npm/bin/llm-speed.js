#!/usr/bin/env node
/*
 * Tiny stub: forward argv + stdio to the real llm-speed binary that
 * install.js dropped into this directory.
 */
"use strict";

const path = require("path");
const fs = require("fs");
const { spawnSync } = require("child_process");

const isWin = process.platform === "win32";
const binName = isWin ? "llm-speed.exe" : "llm-speed";
const binPath = path.join(__dirname, binName);

if (!fs.existsSync(binPath)) {
  console.error(
    `[llm-speed] binary not found at ${binPath}.\n` +
      `[llm-speed] postinstall may have failed; try: pipx install llm-speed`
  );
  process.exit(1);
}

const result = spawnSync(binPath, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error("[llm-speed] failed to launch:", result.error.message);
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);
