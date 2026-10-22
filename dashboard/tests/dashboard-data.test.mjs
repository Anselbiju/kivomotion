import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const data = JSON.parse(
  await readFile(new URL("../src/data.json", import.meta.url), "utf8"),
);

test("dashboard snapshot contains every required role-level query", () => {
  const required = [
    "events",
    "sessions",
    "athletes",
    "acceleration_events",
    "acceleration_athletes",
    "athlete_comparison",
    "model_performance",
    "model_validation",
    "quality",
  ];
  for (const queryId of required) {
    assert.ok(data.queries[queryId], `missing query ${queryId}`);
    assert.ok(Array.isArray(data.queries[queryId].rows));
  }
});

test("quality totals reconcile with reviewed rows", () => {
  const quality = data.queries.quality.rows[0];
  assert.equal(data.queries.events.rows.length, quality.matchedJumps);
  assert.equal(data.queries.sessions.rows.length, quality.matchedSessions);
  assert.equal(data.queries.athletes.rows.length, quality.athletes);
  assert.equal(quality.athleteIdCompletenessPct, 100);
  assert.equal(quality.duplicateJumpIds, 0);
  assert.equal(quality.missingAccelerationCells, 0);
  assert.equal(quality.repeatedAthletes + quality.singleSessionAthletes, quality.athletes);
});

test("deployment status remains conservative without an external holdout", () => {
  const validation = data.queries.model_validation.rows[0];
  assert.equal(validation.externalHoldoutEvaluated, false);
  assert.equal(validation.deploymentReady, false);
});
