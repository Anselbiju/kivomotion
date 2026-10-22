import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const packageJson = JSON.parse(await readFile(
  new URL("../package.json", import.meta.url),
  "utf8",
));
const startScript = await readFile(
  new URL("../data_pipeline/start.py", import.meta.url),
  "utf8",
);

test("normal start command builds and serves the authenticated dashboard", () => {
  assert.equal(packageJson.scripts["build:auth"], "vite build --config vite.auth.config.js");
  assert.equal(packageJson.scripts.start, "python3 data_pipeline/start.py --port 4180");
  assert.match(startScript, /ensure_current\(project_root, data_dir\)/);
  assert.match(startScript, /\["npm", "run", "build:auth"\]/);
  assert.match(startScript, /auth_server\.py/);
});

test("authoring preview remains an explicit separate command", () => {
  assert.equal(packageJson.scripts.dev, "vite");
});
