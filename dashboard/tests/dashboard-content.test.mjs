import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../src/content/dashboard/DashboardContent.jsx", import.meta.url),
  "utf8",
);

test("all three audience views remain registered", () => {
  for (const label of ["Athlete", "Coach", "Sport scientist"]) {
    assert.match(source, new RegExp(`label: "${label}"`));
  }
});

test("athlete sessions are restricted to the athlete page and cannot change identity", () => {
  assert.match(source, /athleteOnly \? roleTabs\.filter/);
  assert.match(source, /initialAthleteId=\{athleteOnly \? auth\.athleteId/);
  assert.match(source, /canSelectAthlete=\{!athleteOnly\}/);
});

test("athlete jump history opens a reviewed jump-detail dialog", () => {
  assert.match(source, /function JumpDetailDialog/);
  assert.match(source, /rowKey="jumpId"/);
  assert.match(source, /onRowSelect=\{\(selected\) => setSelectedJumpId\(selected\.jumpId\)\}/);
  assert.match(source, /title=\{`Jump \$\{jump\.jumpNumber\} details`\}/);
  assert.match(source, /Timing and agreement/);
  assert.match(source, /Acceleration magnitude/);
  assert.match(source, /Axis details/);
});

test("coach and scientist jump tables open the shared reviewed detail dialog", () => {
  assert.match(source, /id="coach-jump-history"/);
  assert.match(source, /title="Team jump history"/);
  assert.match(source, /id="scientist-event-table"[\s\S]*rowKey="jumpId"/);
  assert.match(source, /id="scientist-acceleration-explorer"[\s\S]*rowKey="jumpId"/);
  assert.match(source, /function CoachView[\s\S]*<JumpDetailDialog jump=\{selectedAccelerationJump\}/);
  assert.match(source, /function ScientistView[\s\S]*<JumpDetailDialog jump=\{selectedAccelerationJump\}/);
});

test("coach view includes a two-athlete performance comparison", () => {
  assert.match(source, /id="coach-athlete-comparison"/);
  assert.match(source, /queryId="athlete_comparison"/);
  assert.match(source, /label="Athlete A"/);
  assert.match(source, /label="Athlete B"/);
  assert.match(source, /className="km-comparison-kpi-grid"/);
  assert.match(source, /Height comparison/);
  assert.match(source, /Acceleration comparison/);
  assert.match(source, /Performance comparison/);
  assert.match(source, /Impulse comparison/);
  for (const category of ["Height", "Acceleration", "Performance", "Impulse"]) {
    assert.match(source, new RegExp(`group: "${category}"`));
  }
});

test("athlete label rename is available only to coach and scientist workspaces", () => {
  assert.match(source, /function AthleteRenameControl/);
  assert.match(source, /<CoachView queries=\{displayQueries\} renameControl=\{renameControl\}/);
  assert.match(source, /<ScientistView queries=\{displayQueries\} renameControl=\{renameControl\}/);
  assert.match(source, /<AthleteView queries=\{displayQueries\}[\s\S]*canSelectAthlete=\{!athleteOnly\} \/>/);
  assert.doesNotMatch(source, /<AthleteView[^>]*renameControl/);
  assert.match(source, /Stable athlete ID/);
  assert.match(source, /fetch\("\/api\/athletes\/rename"/);
  assert.match(source, /renamed_athlete column in athlete_id_map\.csv/);
  assert.match(source, /persistent=\{auth\.authenticated\}/);
  assert.match(source, /<form className="km-rename-form" onSubmit=/);
  assert.match(source, /type="submit" className="km-rename-save"/);
});

test("scientist workspace exposes event-yield and flight-time QA", () => {
  assert.match(source, /id="scientist-session-quality"/);
  assert.match(source, /queryId="session_quality"/);
  assert.match(source, /Force-event retention/);
  assert.match(source, /IMU-event retention/);
  assert.match(source, /Median flight-time difference/);
  assert.match(source, /Model artifact provenance/);
});

test("scientist workspace includes an interactive impulse prediction and model review", () => {
  assert.match(source, /function PredictionModelReview/);
  assert.match(source, /id="scientist-prediction-review"/);
  assert.match(source, /Actual vs predicted relative impulse/);
  assert.match(source, /label="Athlete" showLabel/);
  assert.match(source, /label="Session" showLabel/);
  assert.match(source, /aria-label="Chart series"/);
  assert.match(source, /View full jump details/);
  assert.match(source, /Candidate model comparison/);
  assert.match(source, /Relative MAE/);
  assert.match(source, /Relative RMSE/);
  assert.match(source, /Residual SD/);
  assert.match(source, /No confidence band is shown/);
  assert.match(source, /item\.feature_set === selectedFeatureSet/);
  assert.match(source, /\{candidateModels\.length\} candidates/);
  assert.match(source, /candidateModels\.map/);
  assert.doesNotMatch(source, /id="scientist-model-comparison"/);
  assert.doesNotMatch(source, /id="scientist-predictions"/);
});

test("scientist workspace includes the approved audited data-review workflow", () => {
  assert.match(source, /function ScientistDataReview/);
  assert.match(source, /id="scientist-data-review"/);
  assert.match(source, /Scientific data review/);
  assert.match(source, /Audited corrections/);
  assert.match(source, /Original values are read-only/);
  assert.match(source, /Proposed flight-time difference/);
  assert.match(source, /Save draft/);
  assert.match(source, /Submit correction/);
  assert.match(source, /flightTimeDiffMs\) > 100/);
  assert.match(source, /fetch\("\/api\/scientist-reviews"/);
  assert.match(source, /reviewPersistent=\{auth\.authenticated && auth\.role === "coach"\}/);
});

test("sortable row identifiers are unique", () => {
  const ids = [...source.matchAll(/row\("([^"]+)"/g)].map((match) => match[1]);
  const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
  assert.deepEqual([...new Set(duplicates)], []);
});
