import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  Button,
  DataComponent,
  DataTable,
  Dialog,
  Dropdown,
  EvidenceChart,
  Icon,
  MetricCard,
  SortableItem,
  SortableRegion,
  useDataApp,
  useDashboardTabs,
} from "../../data-app-public.jsx";

export const roleTabs = [
  { id: "athlete", label: "Athlete", filterIds: [], focusFields: ["athleteId"] },
  { id: "coach", label: "Coach", filterIds: [] },
  { id: "scientist", label: "Sport scientist", filterIds: [] },
];

function currentAuth() {
  const auth = globalThis.__KIVO_AUTH__;
  if (auth?.authenticated && ["coach", "athlete"].includes(auth.role)) return auth;
  return { authenticated: false, role: "coach", displayName: "Local preview" };
}

const column = (field, label, presentation) => ({ field, label, ...(presentation ? { presentation } : {}) });
const fmt = (value, digits = 1) => Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
const signed = (value, digits = 1) => Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(digits)}%` : "—";
const mean = (values) => {
  const valid = values.filter(Number.isFinite);
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
};
const sampleStandardDeviation = (values) => {
  const valid = values.filter(Number.isFinite);
  if (valid.length < 2) return null;
  const average = mean(valid);
  return Math.sqrt(valid.reduce((total, value) => total + ((value - average) ** 2), 0) / (valid.length - 1));
};
const row = (id, items, spacing = "content") => ({ id, items, spacing });

function Canvas({ id, label, rows, children }) {
  return <SortableRegion id={id} label={label} variant="canvas" columns={12} rows={rows}
    spacing="standard" authoredRevision={1}>{children}</SortableRegion>;
}

function Metric({ id, queryId, title, value, comparison, description, sourceRows, trendValues, span = 3, negative = false }) {
  return <SortableItem id={id} label={title} kind="metric" span={span} minSpan={2}>
    <MetricCard id={id} queryId={queryId} title={title} value={value} comparison={comparison}
      negative={negative} trendValues={trendValues} description={description}
      sourceRows={sourceRows} displayRows={sourceRows} />
  </SortableItem>;
}

function Chart({ id, queryId, title, rows, sourceRows = rows, spec, description, span = 6, height = 260 }) {
  return <SortableItem id={id} label={title} kind="chart" span={span} minSpan={4} className="km-plot-block">
    <EvidenceChart variant="card" id={id} queryId={queryId} title={title} rows={rows}
      sourceRows={sourceRows} spec={spec} height={height} description={description} />
  </SortableItem>;
}

function Table({
  id,
  queryId,
  title,
  rows,
  sourceRows,
  columns,
  description,
  span = 12,
  searchable = false,
  rowKey,
  selectedRowKey,
  onRowSelect,
  rowActionLabel,
}) {
  return <SortableItem id={id} label={title} kind="table" span={span} minSpan={5} className="km-content-sized">
    <DataComponent variant="card" id={id} queryId={queryId} title={title} kind="table"
      sourceRows={sourceRows} displayRows={rows} description={description}>
      <DataTable rows={rows} columns={columns} searchable={searchable} rowKey={rowKey}
        selectedRowKey={selectedRowKey} onRowSelect={onRowSelect} rowActionLabel={rowActionLabel} />
    </DataComponent>
  </SortableItem>;
}

function JumpDetailMetric({ label, value, note }) {
  return <div className="km-jump-detail-metric">
    <span>{label}</span>
    <strong>{value}</strong>
    {note ? <small>{note}</small> : null}
  </div>;
}

function JumpDetailDialog({ jump, event, onClose }) {
  if (!jump) return null;
  const metric = (value, unit, digits = 2) => Number.isFinite(value) ? `${Number(value).toFixed(digits)} ${unit}` : "—";
  const axisRows = [
    { axis: "X axis", mean: jump.xAccelerationMeanG, variability: jump.xAccelerationStdG, minimum: jump.xAccelerationMinG, maximum: jump.xAccelerationMaxG, range: jump.xAccelerationRangeG, rms: jump.xAccelerationRmsG },
    { axis: "Y axis", mean: jump.yAccelerationMeanG, variability: jump.yAccelerationStdG, minimum: jump.yAccelerationMinG, maximum: jump.yAccelerationMaxG, range: jump.yAccelerationRangeG, rms: jump.yAccelerationRmsG },
    { axis: "Z axis", mean: jump.zAccelerationMeanG, variability: jump.zAccelerationStdG, minimum: jump.zAccelerationMinG, maximum: jump.zAccelerationMaxG, range: jump.zAccelerationRangeG, rms: jump.zAccelerationRmsG },
  ];

  return <Dialog open expanded title={`Jump ${jump.jumpNumber} details`} onClose={onClose}
    className="km-jump-detail-dialog" initialFocusSelector=".km-jump-detail-summary">
    <div className="km-jump-detail" data-reviewed-rows>
      <div className="km-jump-detail-summary" tabIndex={-1}>
        <div>
          <span>Accepted matched jump</span>
          <strong>{jump.date} · {jump.sessionId}</strong>
        </div>
        <div>
          <span>Jump ID</span>
          <strong>{jump.jumpId}</strong>
        </div>
      </div>

      <section>
        <header><h3>Performance</h3><p>Force-plate outcomes for this matched effort.</p></header>
        <div className="km-jump-detail-grid">
          <JumpDetailMetric label="Impulse-derived height" value={metric(jump.jumpHeightCm, "cm", 1)} />
          <JumpDetailMetric label="Flight-time height" value={metric(event?.flightHeightCm, "cm", 1)} />
          <JumpDetailMetric label="Takeoff velocity" value={metric(jump.takeoffVelocityMs, "m/s")} />
          <JumpDetailMetric label="Relative impulse" value={metric(jump.impulseNsKg, "N·s/kg")} />
          <JumpDetailMetric label="Peak force" value={metric(event?.peakForceNkg, "N/kg")} note={metric(event?.peakForceN, "N", 0)} />
          <JumpDetailMetric label="Peak landing force" value={metric(event?.peakLandingForceN, "N", 0)} />
        </div>
      </section>

      <section>
        <header><h3>Timing and agreement</h3><p>Force and IMU event timing for the selected match.</p></header>
        <div className="km-jump-detail-grid">
          <JumpDetailMetric label="Force flight time" value={metric(event?.forceFlightTimeMs, "ms", 0)} />
          <JumpDetailMetric label="IMU flight time" value={metric(event?.imuFlightTimeMs, "ms", 0)} />
          <JumpDetailMetric label="Flight-time difference" value={metric(event?.flightTimeDiffMs, "ms", 0)} />
          <JumpDetailMetric label="Synchronization error" value={metric(jump.syncErrorMs, "ms", 1)} />
          <JumpDetailMetric label="Body mass" value={metric(event?.massKg, "kg", 1)} />
          <JumpDetailMetric label="Flight minimum" value={metric(jump.flightMinAccelerationG, "g", 3)} />
        </div>
      </section>

      <section>
        <header><h3>Acceleration magnitude</h3><p>Filtered IMU acceleration features calculated within this jump window.</p></header>
        <div className="km-jump-detail-grid">
          <JumpDetailMetric label="Peak" value={metric(jump.accelerationMagnitudeMaxG, "g")} />
          <JumpDetailMetric label="RMS" value={metric(jump.accelerationMagnitudeRmsG, "g")} />
          <JumpDetailMetric label="Mean" value={metric(jump.accelerationMagnitudeMeanG, "g")} />
          <JumpDetailMetric label="Variability" value={metric(jump.accelerationMagnitudeStdG, "g")} note="Within-jump standard deviation" />
          <JumpDetailMetric label="Range" value={metric(jump.accelerationMagnitudeRangeG, "g")} />
          <JumpDetailMetric label="Acceleration area" value={metric(jump.accelerationAreaGs, "g·s")} />
        </div>
      </section>

      <section>
        <header><h3>Axis details</h3><p>Filtered acceleration summaries for each sensor axis, in g.</p></header>
        <div className="km-jump-axis-table-wrap">
          <table className="km-jump-axis-table">
            <thead><tr><th>Axis</th><th>Mean</th><th>SD</th><th>Minimum</th><th>Maximum</th><th>Range</th><th>RMS</th></tr></thead>
            <tbody>{axisRows.map((axis) => <tr key={axis.axis}>
              <th scope="row">{axis.axis}</th>
              <td>{fmt(axis.mean, 3)}</td><td>{fmt(axis.variability, 3)}</td><td>{fmt(axis.minimum, 3)}</td>
              <td>{fmt(axis.maximum, 3)}</td><td>{fmt(axis.range, 3)}</td><td>{fmt(axis.rms, 3)}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </section>
    </div>
  </Dialog>;
}

function AthleteComparison({ id, athletes, leftId, rightId, onLeftChange, onRightChange }) {
  const left = athletes.find((item) => item.athleteId === leftId);
  const right = athletes.find((item) => item.athleteId === rightId);
  const comparisonRows = [
    { group: "Height", label: "Average jump", field: "avgHeightCm", unit: "cm", digits: 1 },
    { group: "Height", label: "Best jump", field: "bestHeightCm", unit: "cm", digits: 1 },
    { group: "Acceleration", label: "Average peak", field: "avgPeakAccelerationG", unit: "g", digits: 2 },
    { group: "Acceleration", label: "Maximum peak", field: "maxPeakAccelerationG", unit: "g", digits: 2 },
    { group: "Performance", label: "Jump variation", field: "consistencyCvPct", unit: "% CV", digits: 1, lowerIsBetter: true },
    { group: "Performance", label: "Session height change", field: "sessionChangePct", unit: "%", digits: 1, signedValue: true },
    { group: "Impulse", label: "Relative impulse", field: "avgImpulseNsKg", unit: "N·s/kg", digits: 2 },
  ];
  const comparisonCards = [
    { key: "height", title: "Height comparison", primary: comparisonRows[0], secondary: comparisonRows[1] },
    { key: "acceleration", title: "Acceleration comparison", primary: comparisonRows[2], secondary: comparisonRows[3] },
    { key: "performance", title: "Performance comparison", primary: comparisonRows[4], secondary: comparisonRows[5] },
    { key: "impulse", title: "Impulse comparison", primary: comparisonRows[6] },
  ];
  const formatValue = (athlete, metric) => {
    const value = athlete?.[metric.field];
    if (!Number.isFinite(value)) return "—";
    const formatted = metric.signedValue ? `${value > 0 ? "+" : ""}${value.toFixed(metric.digits)}` : value.toFixed(metric.digits);
    return `${formatted} ${metric.unit}`;
  };
  const difference = (metric) => {
    const leftValue = left?.[metric.field];
    const rightValue = right?.[metric.field];
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) return "—";
    const delta = leftValue - rightValue;
    return `${delta > 0 ? "+" : ""}${delta.toFixed(metric.digits)} ${metric.unit}`;
  };

  return <SortableItem id={id} label="Two-athlete comparison" kind="table" span={12} minSpan={8} className="km-content-sized">
    <DataComponent variant="card" id={id} queryId="athlete_comparison" title="Two-athlete comparison" kind="table"
      sourceRows={[left, right].filter(Boolean)} displayRows={comparisonRows.map((metric) => ({
        category: metric.group,
        metric: metric.label,
        athleteA: formatValue(left, metric),
        athleteB: formatValue(right, metric),
        difference: difference(metric),
      }))}
      description="Side-by-side descriptive comparison from matched force and IMU jumps. Acceleration and change direction are contextual measures, not readiness or fatigue diagnoses.">
      <div className="km-comparison">
        <div className="km-comparison-controls" aria-label="Choose athletes to compare">
          <div className="km-comparison-athlete">
            <span className="km-comparison-side">Athlete A</span>
            <span className="km-comparison-select">
              <Dropdown label="Athlete A" value={left?.athleteId ?? ""}
                choices={athletes.map((item) => item.athleteId)} onChange={onLeftChange}
                formatChoice={(value) => athletes.find((item) => item.athleteId === value)?.athleteLabel ?? value} />
            </span>
            <small>{left?.jumpCount ?? 0} jumps · {left?.sessionCount ?? 0} session{left?.sessionCount === 1 ? "" : "s"}</small>
          </div>
          <div className="km-comparison-versus" aria-hidden="true">VS</div>
          <div className="km-comparison-athlete">
            <span className="km-comparison-side">Athlete B</span>
            <span className="km-comparison-select">
              <Dropdown label="Athlete B" value={right?.athleteId ?? ""}
                choices={athletes.map((item) => item.athleteId)} onChange={onRightChange}
                formatChoice={(value) => athletes.find((item) => item.athleteId === value)?.athleteLabel ?? value} />
            </span>
            <small>{right?.jumpCount ?? 0} jumps · {right?.sessionCount ?? 0} session{right?.sessionCount === 1 ? "" : "s"}</small>
          </div>
        </div>
        <div className="km-comparison-kpi-grid" role="list" aria-label="Selected athlete KPI comparison">
          {comparisonCards.map((card) => <article className={`km-comparison-kpi km-comparison-kpi-${card.key}`} role="listitem" key={card.key}>
            <header>
              <span>{card.title}</span>
              <strong>{difference(card.primary)}</strong>
              <small>A − B</small>
            </header>
            <div className="km-comparison-kpi-measure">{card.primary.label}{card.primary.lowerIsBetter ? <em>Lower is steadier</em> : null}</div>
            <div className="km-comparison-kpi-values">
              <section>
                <small>A · {left?.athleteLabel ?? "Athlete A"}</small>
                <strong>{formatValue(left, card.primary)}</strong>
              </section>
              <span aria-hidden="true">vs</span>
              <section>
                <small>B · {right?.athleteLabel ?? "Athlete B"}</small>
                <strong>{formatValue(right, card.primary)}</strong>
              </section>
            </div>
            {card.secondary ? <div className="km-comparison-kpi-secondary">
              <span>{card.secondary.label}</span>
              <strong>{formatValue(left, card.secondary)}</strong>
              <span aria-hidden="true">/</span>
              <strong>{formatValue(right, card.secondary)}</strong>
              <small>A − B {difference(card.secondary)}</small>
            </div> : <div className="km-comparison-kpi-secondary">
              <span>Matched jump coverage</span>
              <strong>{left?.jumpCount ?? 0}</strong>
              <span aria-hidden="true">/</span>
              <strong>{right?.jumpCount ?? 0}</strong>
              <small>A / B jumps</small>
            </div>}
          </article>)}
        </div>
        <p className="km-comparison-note">A second session is required before session change can be compared. Review acceleration direction in the context of the movement task and testing conditions.</p>
      </div>
    </DataComponent>
  </SortableItem>;
}

function RoleContext({ title, meta, control }) {
  return <div className="km-role-context">
    <div className="km-role-heading">
      <h2>{title}</h2>
    </div>
    <div className="km-role-actions">
      {control}
      <span className="km-meta-chip">{meta}</span>
    </div>
  </div>;
}

function AthleteRenameControl({ athletes, aliases, onRename, persistent }) {
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [athleteId, setAthleteId] = useState(athletes[0]?.athleteId ?? "");
  const athlete = athletes.find((item) => item.athleteId === athleteId) ?? athletes[0];
  const generatedLabel = athlete ? `Athlete ${athlete.athleteId.split("-")[0].toUpperCase()}` : "";
  const currentLabel = athlete ? aliases[athlete.athleteId] || generatedLabel : "";
  const [draft, setDraft] = useState(currentLabel);
  useEffect(() => {
    if (!athletes.some((item) => item.athleteId === athleteId)) setAthleteId(athletes[0]?.athleteId ?? "");
  }, [athleteId, athletes]);
  useEffect(() => setDraft(currentLabel), [currentLabel, open]);
  const save = async () => {
    if (!athlete || !draft.trim()) return;
    setSaving(true);
    setError("");
    try {
      await onRename(athlete.athleteId, draft.trim());
      setOpen(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The athlete name could not be saved.");
    } finally {
      setSaving(false);
    }
  };
  const reset = async () => {
    if (!athlete) return;
    setSaving(true);
    setError("");
    try {
      await onRename(athlete.athleteId, "");
      setDraft(generatedLabel);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The generated label could not be restored.");
    } finally {
      setSaving(false);
    }
  };

  return <>
    <Button className="km-rename-button" onClick={() => setOpen(true)}>Rename athlete</Button>
    {open && <Dialog open title="Rename athlete ID label" onClose={() => setOpen(false)}
      className="km-rename-dialog" initialFocusSelector="#athlete-display-name">
      <form className="km-rename-form" onSubmit={(event) => { event.preventDefault(); void save(); }}
        aria-describedby={error ? "athlete-rename-error" : undefined}>
        <p>Set a readable dashboard label while preserving the athlete's stable unique ID for joins and access control.</p>
        <label>
          <span>Athlete ID</span>
          <Dropdown label="Athlete ID to rename" value={athlete?.athleteId ?? ""}
            choices={athletes.map((item) => item.athleteId)} onChange={setAthleteId}
            formatChoice={(value) => {
              const item = athletes.find((candidate) => candidate.athleteId === value);
              const label = aliases[value] || (item ? `Athlete ${item.athleteId.split("-")[0].toUpperCase()}` : value);
              return `${label} · ${value}`;
            }} />
        </label>
        <label htmlFor="athlete-display-name">
          <span>Display name</span>
          <input id="athlete-display-name" value={draft} maxLength={48}
            onChange={(event) => setDraft(event.target.value)} placeholder="Enter athlete display name" />
        </label>
        <div className="km-rename-identity">
          <span>Stable athlete ID</span>
          <code>{athlete?.athleteId ?? "—"}</code>
        </div>
        <p className="km-rename-note">{persistent
          ? "Saved names are written to SQLite and the renamed_athlete column in athlete_id_map.csv. The stable athlete ID and account mapping do not change."
          : "This authoring preview keeps the label only for the current browser session. Use the authenticated dashboard to save it to SQLite and athlete_id_map.csv."}</p>
        {error && <p id="athlete-rename-error" className="km-rename-error" role="alert">{error}</p>}
        <div className="km-rename-actions">
          <Button type="button" className="km-rename-reset" disabled={saving} onClick={reset}>Restore generated label</Button>
          <Button type="submit" className="km-rename-save" disabled={saving || !draft.trim()}>{saving ? "Saving…" : "Save display name"}</Button>
        </div>
      </form>
    </Dialog>}
  </>;
}

function DashboardGuide({ activeRole, quality, validation, athleteOnly = false }) {
  const [open, setOpen] = useState(false);
  return <>
    <Button className="km-guide-button" aria-label="Open dashboard guide" onClick={() => setOpen(true)}>
      <Icon name="info" size={16} />Dashboard guide
    </Button>
    {open && <Dialog open title="KivoMotion dashboard guide" onClose={() => setOpen(false)}
      className="km-guide-dialog" initialFocusSelector=".dialog-header">
      <div className="km-guide-intro">
        <p>Use the three role tabs to move between personal performance, team decisions, and scientific validation. Every metric is calculated from accepted force-plate and IMU matches.</p>
      </div>
      <div className="km-guide-grid" data-single={athleteOnly || undefined}>
        <section data-active={activeRole === "athlete" || undefined}>
          <p className="km-guide-kicker">Athlete</p>
          <h2>Personal performance</h2>
          <dl>
            <div><dt>Best and average jump</dt><dd>Impulse-derived jump height across the selected athlete's matched jumps.</dd></div>
            <div><dt>Relative impulse</dt><dd>Net impulse divided by body mass, expressed in N·s/kg.</dd></div>
            <div><dt>Consistency</dt><dd>Jump-height coefficient of variation. Lower percentages mean more repeatable efforts.</dd></div>
            <div><dt>Progress charts</dt><dd>Jump-by-jump and session-level height trends. A second session is needed for session change.</dd></div>
            <div><dt>Acceleration</dt><dd>Peak, RMS, variability, area, and first-to-latest session change from the filtered IMU signal, expressed in g or g·s.</dd></div>
          </dl>
        </section>
        {!athleteOnly && <section data-active={activeRole === "coach" || undefined}>
          <p className="km-guide-kicker">Coach</p>
          <h2>Team decision support</h2>
          <dl>
            <div><dt>Repeat coverage</dt><dd>{quality.repeatedAthletes ?? 0} of {quality.athletes ?? 0} athletes have at least two accepted sessions. Change metrics are preliminary while this coverage remains limited.</dd></div>
            <div><dt>Two-athlete comparison</dt><dd>Select two athletes to compare jump height, acceleration, repeatability, session change, and relative impulse side by side.</dd></div>
            <div><dt>Performance map</dt><dd>Average height compared with jump-height variation for each athlete.</dd></div>
            <div><dt>Follow-up queue</dt><dd>Flags one-session baselines, variation above 10% CV, or a session-average decrease greater than 5%.</dd></div>
            <div><dt>Roster</dt><dd>Searchable athlete comparison across height, impulse, consistency, and session change.</dd></div>
            <div><dt>Acceleration overview</dt><dd>Team-level peak and RMS trends plus each athlete's first-to-latest session change for contextual coaching review.</dd></div>
          </dl>
        </section>}
        {!athleteOnly && <section data-active={activeRole === "scientist" || undefined}>
          <p className="km-guide-kicker">Sport scientist</p>
          <h2>Methods and validation</h2>
          <dl>
            <div><dt>Grouped-CV R²</dt><dd>Research benchmark from five-fold validation grouped by athlete ID. It is not an external deployment test.</dd></div>
            <div><dt>RMSE and residuals</dt><dd>Prediction error for relative impulse; residuals are observed minus predicted values.</dd></div>
            <div><dt>Feature associations</dt><dd>Pearson correlations with relative impulse. Association does not establish causality.</dd></div>
            <div><dt>Synchronization QA</dt><dd>Absolute timing error between matched force and IMU takeoff events.</dd></div>
            <div><dt>Acceleration diagnostics</dt><dd>Full axis and magnitude summaries, distribution, session change, and correlations with force-plate relative impulse.</dd></div>
            <div><dt>Deployment gate</dt><dd>{validation.deploymentReady ? "Passed" : "Not passed"}. An untouched external athlete cohort is required before operational deployment.</dd></div>
          </dl>
        </section>}
      </div>
      <div className="km-guide-footer">
        <p><strong>Data coverage:</strong> accepted matched jumps recorded from {quality.dataStartDate ?? "—"} to {quality.dataEndDate ?? "—"}. Athlete labels are shortened forms of stable project UUIDs.</p>
        <p><strong>Longitudinal coverage:</strong> {quality.repeatedAthletes ?? 0} of {quality.athletes ?? 0} athletes have repeated sessions ({fmt(quality.repeatCoveragePct, 1)}%). First-to-latest change is shown only for that subset.</p>
        <p><strong>Interpretation:</strong> the dashboard supports review and follow-up; it does not diagnose injury, fatigue, readiness, or medical status.</p>
        <p><strong>Sources:</strong> project batch outputs are normalized into SQLite and a reviewed dashboard snapshot. Local development automatically refreshes both when source summaries change; production builds remain versioned snapshots. Use a card's source action for exact records and metric lineage.</p>
      </div>
    </Dialog>}
  </>;
}

function AthleteView({ queries, initialAthleteId, canSelectAthlete = true }) {
  const athletes = queries.athletes?.rows ?? [];
  const events = queries.events?.rows ?? [];
  const sessions = queries.sessions?.rows ?? [];
  const accelerationAthletes = queries.acceleration_athletes?.rows ?? [];
  const accelerationEvents = queries.acceleration_events?.rows ?? [];
  const accelerationSessions = queries.acceleration_sessions?.rows ?? [];
  const defaultAthlete = useMemo(() => athletes.toSorted((a, b) => b.sessionCount - a.sessionCount || a.athleteLabel.localeCompare(b.athleteLabel))[0], [athletes]);
  const [athleteId, setAthleteId] = useState(initialAthleteId || defaultAthlete?.athleteId || "");
  const [selectedJumpId, setSelectedJumpId] = useState(null);
  useEffect(() => {
    if (!athletes.some((athlete) => athlete.athleteId === athleteId)) setAthleteId(defaultAthlete?.athleteId || "");
  }, [athleteId, athletes, defaultAthlete?.athleteId]);
  useEffect(() => setSelectedJumpId(null), [athleteId]);
  const athlete = athletes.find((item) => item.athleteId === athleteId) ?? defaultAthlete;
  const athleteEvents = events.filter((item) => item.athleteId === athlete?.athleteId)
    .map((item, index) => ({ ...item, jumpSequence: index + 1 }));
  const athleteSessions = sessions.filter((item) => item.athleteId === athlete?.athleteId)
    .map((item, index) => ({ ...item, sessionLabel: `${item.date} · S${index + 1}` }));
  const athleteAcceleration = accelerationAthletes.find((item) => item.athleteId === athlete?.athleteId);
  const athleteAccelerationEvents = accelerationEvents.filter((item) => item.athleteId === athlete?.athleteId)
    .map((item, index) => ({ ...item, jumpSequence: index + 1 }));
  const athleteAccelerationSessions = accelerationSessions.filter((item) => item.athleteId === athlete?.athleteId)
    .map((item, index) => ({ ...item, sessionLabel: `${item.date} · S${index + 1}` }));
  const heightHistory = athleteEvents.map((item) => item.jumpHeightCm);
  const bestGap = athlete ? athlete.bestHeightCm - athlete.avgHeightCm : null;
  const steadiness = !athlete ? "—" : athlete.consistencyCvPct < 5 ? "Very steady" : athlete.consistencyCvPct < 10 ? "Moderate variation" : "High variation";
  const athleteTable = athleteAccelerationEvents.map((item) => ({
    jumpId: item.jumpId,
    date: item.date,
    session: item.sessionId,
    jump: item.jumpNumber,
    height: `${fmt(item.jumpHeightCm)} cm`,
    velocity: `${fmt(item.takeoffVelocityMs, 2)} m/s`,
    impulse: `${fmt(item.impulseNsKg, 2)} N·s/kg`,
    peakAcceleration: `${fmt(item.accelerationMagnitudeMaxG, 2)} g`,
    accelerationVariability: `${fmt(item.accelerationMagnitudeStdG, 2)} g`,
    accelerationArea: `${fmt(item.accelerationAreaGs, 2)} g·s`,
    sync: `${fmt(item.syncErrorMs)} ms`,
  }));
  const selectedAccelerationJump = athleteAccelerationEvents.find((item) => item.jumpId === selectedJumpId);
  const selectedForceJump = athleteEvents.find((item) => item.jumpId === selectedJumpId);

  const metricRows = [
    row("athlete-metrics", ["athlete-best-height", "athlete-average-height", "athlete-impulse", "athlete-consistency"], "none"),
    row("athlete-trends", ["athlete-height-trend", "athlete-session-trend"]),
    row("athlete-acceleration-metrics", ["athlete-acceleration-average-peak", "athlete-acceleration-max-peak", "athlete-acceleration-variability", "athlete-acceleration-change"], "none"),
    row("athlete-acceleration-trends", ["athlete-acceleration-trend", "athlete-acceleration-session-trend"]),
    row("athlete-history", ["athlete-jump-history"]),
  ];
  return <>
    <RoleContext title={athlete?.athleteLabel ?? "Athlete"}
      meta={`${athlete?.jumpCount ?? 0} matched jumps · ${athlete?.sessionCount ?? 0} session${athlete?.sessionCount === 1 ? "" : "s"}`}
      control={canSelectAthlete ? <span className="km-athlete-filter">
        <Dropdown label="Athlete ID" showLabel value={athlete?.athleteId ?? ""}
          choices={athletes.map((item) => item.athleteId)} onChange={setAthleteId}
          formatChoice={(value) => athletes.find((item) => item.athleteId === value)?.athleteLabel ?? value} />
      </span> : null} />
    <Canvas id="kivomotion-athlete-canvas" label="Athlete performance blocks" rows={metricRows}>
      <Metric id="athlete-best-height" queryId="events" title="Best jump" value={`${fmt(athlete?.bestHeightCm)} cm`}
        comparison={Number.isFinite(bestGap) ? `+${fmt(bestGap)} cm vs average` : null}
        trendValues={heightHistory} sourceRows={athleteEvents} description="Highest impulse-derived jump height in this athlete's matched jumps." />
      <Metric id="athlete-average-height" queryId="events" title="Average jump" value={`${fmt(athlete?.avgHeightCm)} cm`}
        trendValues={heightHistory} sourceRows={athleteEvents} description="Mean impulse-derived jump height across this athlete's matched jumps." />
      <Metric id="athlete-impulse" queryId="events" title="Relative impulse" value={`${fmt(athlete?.avgImpulseNsKg, 2)} N·s/kg`}
        sourceRows={athleteEvents} description="Average net impulse normalized by body mass, allowing fairer comparison across body sizes." />
      <Metric id="athlete-consistency" queryId="athletes" title="Consistency" value={`${fmt(athlete?.consistencyCvPct)}% CV`}
        comparison={steadiness} negative sourceRows={athlete ? [athlete] : []}
        description="Coefficient of variation in jump height. Lower values indicate more repeatable efforts." />
      <Chart id="athlete-height-trend" queryId="events" title="Jump-height progression" rows={athleteEvents} sourceRows={athleteEvents}
        spec={{ type: "line", x: "jumpSequence", y: "jumpHeightCm", startAtZero: false, valueDecimals: 1, colors: { jumpHeightCm: "var(--chart-1)" } }}
        description="Each point is one matched jump in recording order. Use it to see spread and direction, not to infer fatigue or readiness on its own." />
      <Chart id="athlete-session-trend" queryId="sessions" title="Session summary" rows={athleteSessions} sourceRows={athleteSessions}
        spec={{ type: "line", x: "sessionLabel", y: "avgHeightCm", fields: ["avgHeightCm", "bestHeightCm"], startAtZero: false,
          colors: { avgHeightCm: "var(--chart-3)", bestHeightCm: "var(--chart-4)" },
          legend: { labels: { avgHeightCm: "Average", bestHeightCm: "Best" } } }}
        description="Average and best jump height for each available session. A single session establishes a baseline; repeated sessions reveal direction." />
      <Metric id="athlete-acceleration-average-peak" queryId="acceleration_athletes" title="Average peak acceleration"
        value={`${fmt(athleteAcceleration?.avgPeakAccelerationG, 2)} g`} comparison="Across matched jumps"
        trendValues={athleteAccelerationEvents.map((item) => item.accelerationMagnitudeMaxG)}
        sourceRows={athleteAcceleration ? [athleteAcceleration] : []}
        description="Average of the maximum filtered IMU acceleration magnitude recorded in each matched jump." />
      <Metric id="athlete-acceleration-max-peak" queryId="acceleration_athletes" title="Highest acceleration"
        value={`${fmt(athleteAcceleration?.maxPeakAccelerationG, 2)} g`} comparison="Maximum observed"
        trendValues={athleteAccelerationEvents.map((item) => item.accelerationMagnitudeMaxG)}
        sourceRows={athleteAcceleration ? [athleteAcceleration] : []}
        description="Highest filtered acceleration-magnitude peak across this athlete's matched jumps." />
      <Metric id="athlete-acceleration-variability" queryId="acceleration_athletes" title="Acceleration variability"
        value={`${fmt(athleteAcceleration?.avgAccelerationVariabilityG, 2)} g`} comparison="Mean within-jump SD"
        trendValues={athleteAccelerationEvents.map((item) => item.accelerationMagnitudeStdG)}
        sourceRows={athleteAcceleration ? [athleteAcceleration] : []}
        description="Average within-jump standard deviation of acceleration magnitude. Interpret alongside the movement task and session context." />
      <Metric id="athlete-acceleration-change" queryId="acceleration_athletes" title="Acceleration change"
        value={signed(athleteAcceleration?.peakAccelerationSessionChangePct)}
        comparison={athleteAcceleration?.sessionCount > 1 ? "First to latest session" : "Needs a second session"}
        sourceRows={athleteAcceleration ? [athleteAcceleration] : []}
        description="Change in session-average peak acceleration from the first accepted session to the latest. It remains unavailable for single-session athletes." />
      <Chart id="athlete-acceleration-trend" queryId="acceleration_events" title="Acceleration by jump"
        rows={athleteAccelerationEvents} sourceRows={athleteAccelerationEvents}
        spec={{ type: "line", x: "jumpSequence", y: "accelerationMagnitudeMaxG",
          fields: ["accelerationMagnitudeMaxG", "accelerationMagnitudeRmsG"], startAtZero: false, valueDecimals: 2,
          colors: { accelerationMagnitudeMaxG: "var(--chart-2)", accelerationMagnitudeRmsG: "var(--chart-5)" },
          legend: { labels: { accelerationMagnitudeMaxG: "Peak", accelerationMagnitudeRmsG: "RMS" } } }}
        description="Peak and RMS filtered acceleration magnitude for every matched jump, expressed in g." />
      <Chart id="athlete-acceleration-session-trend" queryId="acceleration_sessions" title="Acceleration by session"
        rows={athleteAccelerationSessions} sourceRows={athleteAccelerationSessions}
        spec={{ type: "line", x: "sessionLabel", y: "avgPeakAccelerationG",
          fields: ["avgPeakAccelerationG", "avgAccelerationRmsG"], startAtZero: false, valueDecimals: 2,
          colors: { avgPeakAccelerationG: "var(--chart-2)", avgAccelerationRmsG: "var(--chart-5)" },
          legend: { labels: { avgPeakAccelerationG: "Average peak", avgAccelerationRmsG: "Average RMS" } } }}
        description="Session-average peak and RMS acceleration magnitude. Compare repeated sessions recorded under similar test conditions." />
      <Table id="athlete-jump-history" queryId="acceleration_events" title="Jump history" rows={athleteTable} sourceRows={athleteAccelerationEvents} searchable
        rowKey="jumpId" selectedRowKey={selectedJumpId} onRowSelect={(selected) => setSelectedJumpId(selected.jumpId)}
        rowActionLabel={(selected) => `View details for jump ${selected.jump} from ${selected.date}`}
        columns={[column("date", "Date"), column("session", "Session"), column("jump", "Jump"), column("height", "Height"), column("velocity", "Takeoff velocity"), column("impulse", "Relative impulse"), column("peakAcceleration", "Peak acceleration"), column("accelerationVariability", "Acceleration variability"), column("accelerationArea", "Acceleration area"), column("sync", "Sync error")]}
        description="Matched force-plate and IMU jumps with acceleration magnitude, variability, and area for this athlete. Select a row to inspect the full jump details." />
    </Canvas>
    <JumpDetailDialog jump={selectedAccelerationJump} event={selectedForceJump}
      onClose={() => setSelectedJumpId(null)} />
  </>;
}

function CoachView({ queries, renameControl }) {
  const athletes = queries.athletes?.rows ?? [];
  const comparisonAthletes = queries.athlete_comparison?.rows ?? [];
  const sessions = queries.sessions?.rows ?? [];
  const events = queries.events?.rows ?? [];
  const accelerationEvents = queries.acceleration_events?.rows ?? [];
  const daily = queries.daily?.rows ?? [];
  const accelerationAthletes = queries.acceleration_athletes?.rows ?? [];
  const accelerationDaily = queries.acceleration_daily?.rows ?? [];
  const rankedComparisonAthletes = useMemo(() => comparisonAthletes.toSorted((a, b) => b.sessionCount - a.sessionCount || a.athleteLabel.localeCompare(b.athleteLabel)), [comparisonAthletes]);
  const [comparisonLeftId, setComparisonLeftId] = useState(rankedComparisonAthletes[0]?.athleteId ?? "");
  const [comparisonRightId, setComparisonRightId] = useState(rankedComparisonAthletes[1]?.athleteId ?? rankedComparisonAthletes[0]?.athleteId ?? "");
  const [selectedJumpId, setSelectedJumpId] = useState(null);
  useEffect(() => {
    const available = new Set(rankedComparisonAthletes.map((item) => item.athleteId));
    if (!available.has(comparisonLeftId)) setComparisonLeftId(rankedComparisonAthletes[0]?.athleteId ?? "");
    if (!available.has(comparisonRightId) || comparisonRightId === comparisonLeftId) {
      setComparisonRightId(rankedComparisonAthletes.find((item) => item.athleteId !== comparisonLeftId)?.athleteId ?? "");
    }
  }, [comparisonLeftId, comparisonRightId, rankedComparisonAthletes]);
  const chooseLeftAthlete = (athleteId) => {
    if (athleteId === comparisonRightId) setComparisonRightId(comparisonLeftId);
    setComparisonLeftId(athleteId);
  };
  const chooseRightAthlete = (athleteId) => {
    if (athleteId === comparisonLeftId) setComparisonLeftId(comparisonRightId);
    setComparisonRightId(athleteId);
  };
  const accelerationById = new Map(accelerationAthletes.map((item) => [item.athleteId, item]));
  const forceEventByJumpId = new Map(events.map((item) => [item.jumpId, item]));
  const selectedAccelerationJump = accelerationEvents.find((item) => item.jumpId === selectedJumpId);
  const selectedForceJump = forceEventByJumpId.get(selectedJumpId);
  const repeated = athletes.filter((item) => item.sessionCount > 1);
  const reviewRowsRaw = athletes.map((item) => {
    let flag = "On track";
    let priority = 4;
    if (item.sessionCount < 2) { flag = "Repeat baseline"; priority = 3; }
    if (item.consistencyCvPct > 10) { flag = "Review variability"; priority = 2; }
    if (Number.isFinite(item.sessionChangePct) && item.sessionChangePct < -5) { flag = "Check recent change"; priority = 1; }
    return { ...item, ...accelerationById.get(item.athleteId), flag, priority };
  }).filter((item) => item.flag !== "On track").toSorted((a, b) => a.priority - b.priority || b.consistencyCvPct - a.consistencyCvPct).slice(0, 12);
  const reviewRows = reviewRowsRaw.map((item) => ({
    athlete: item.athleteLabel,
    flag: item.flag,
    sessions: item.sessionCount,
    average: `${fmt(item.avgHeightCm)} cm`,
    consistency: `${fmt(item.consistencyCvPct)}%`,
    change: signed(item.sessionChangePct),
    peakAcceleration: `${fmt(item.avgPeakAccelerationG, 2)} g`,
    accelerationChange: signed(item.peakAccelerationSessionChangePct),
  }));
  const rosterRowsRaw = athletes.toSorted((a, b) => b.avgHeightCm - a.avgHeightCm)
    .map((item) => ({ ...item, ...accelerationById.get(item.athleteId) }));
  const rosterRows = rosterRowsRaw.map((item) => ({
      athlete: item.athleteLabel,
      sessions: item.sessionCount,
      jumps: item.jumpCount,
      average: `${fmt(item.avgHeightCm)} cm`,
      best: `${fmt(item.bestHeightCm)} cm`,
      impulse: `${fmt(item.avgImpulseNsKg, 2)} N·s/kg`,
      consistency: `${fmt(item.consistencyCvPct)}%`,
      change: signed(item.sessionChangePct),
      peakAcceleration: `${fmt(item.avgPeakAccelerationG, 2)} g`,
      accelerationVariability: `${fmt(item.avgAccelerationVariabilityG, 2)} g`,
      accelerationChange: signed(item.peakAccelerationSessionChangePct),
    }));
  const accelerationChangeRows = accelerationAthletes
    .filter((item) => Number.isFinite(item.peakAccelerationSessionChangePct))
    .toSorted((a, b) => b.peakAccelerationSessionChangePct - a.peakAccelerationSessionChangePct);
  const teamJumpRows = accelerationEvents.map((item) => ({
    jumpId: item.jumpId,
    athlete: item.athleteLabel,
    date: item.date,
    session: item.sessionId,
    jump: item.jumpNumber,
    height: `${fmt(item.jumpHeightCm)} cm`,
    impulse: `${fmt(item.impulseNsKg, 2)} N·s/kg`,
    peakAcceleration: `${fmt(item.accelerationMagnitudeMaxG, 2)} g`,
    accelerationVariability: `${fmt(item.accelerationMagnitudeStdG, 2)} g`,
    sync: `${fmt(item.syncErrorMs, 1)} ms`,
  }));
  const rows = [
    row("coach-metrics", ["coach-athletes", "coach-sessions", "coach-jumps", "coach-repeat-coverage"], "none"),
    row("coach-comparison", ["coach-athlete-comparison"]),
    row("coach-overview", ["coach-team-trend", "coach-performance-map"]),
    row("coach-acceleration", ["coach-acceleration-trend", "coach-acceleration-change"]),
    row("coach-actions", ["coach-review-queue", "coach-athlete-table"]),
    row("coach-jump-history-row", ["coach-jump-history"]),
  ];
  return <>
    <RoleContext title="Coach overview"
      meta={`${daily[0]?.date ?? "—"} to ${daily.at(-1)?.date ?? "—"}`} control={renameControl} />
    <Canvas id="kivomotion-coach-canvas" label="Coach performance blocks" rows={rows}>
      <Metric id="coach-athletes" queryId="quality" title="Athletes" value={`${athletes.length}`} comparison="Unique IDs"
        sourceRows={queries.quality?.rows ?? []} description="Unique athlete IDs represented in matched force and IMU data." />
      <Metric id="coach-sessions" queryId="sessions" title="Sessions" value={`${sessions.length}`} comparison="Accepted"
        sourceRows={sessions} description="Completed sessions accepted by the matching pipeline." />
      <Metric id="coach-jumps" queryId="events" title="Matched jumps" value={`${events.length}`} comparison="Force + IMU"
        sourceRows={events} description="Jump events with matched force-plate and IMU records." />
      <Metric id="coach-repeat-coverage" queryId="athletes" title="Repeat coverage" value={`${fmt(athletes.length ? repeated.length / athletes.length * 100 : null, 0)}%`}
        comparison={`${repeated.length} of ${athletes.length} athletes`} sourceRows={athletes} description="Share of athletes with at least two accepted sessions. Longitudinal change remains preliminary at the current coverage." />
      <AthleteComparison id="coach-athlete-comparison" athletes={rankedComparisonAthletes}
        leftId={comparisonLeftId} rightId={comparisonRightId}
        onLeftChange={chooseLeftAthlete} onRightChange={chooseRightAthlete} />
      <Chart id="coach-team-trend" queryId="daily" title="Team testing activity" rows={daily} sourceRows={daily}
        spec={{ type: "bar", x: "date", y: "jumpCount", fields: ["jumpCount", "athleteCount"], stackable: false,
          colors: { jumpCount: "var(--chart-1)", athleteCount: "var(--chart-3)" },
          legend: { labels: { jumpCount: "Matched jumps", athleteCount: "Athletes" } } }}
        description="Testing throughput by recording date. Athlete counts and jump counts are shown as separate, non-additive series." />
      <Chart id="coach-performance-map" queryId="athletes" title="Performance vs repeatability" rows={athletes} sourceRows={athletes}
        spec={{ type: "scatter", x: "avgHeightCm", y: "consistencyCvPct", valueDecimals: 1,
          colors: { consistencyCvPct: "var(--chart-4)" } }}
        description="Each point is one athlete. Higher average height is to the right; lower variation is toward the bottom. Use both dimensions when planning follow-up." />
      <Chart id="coach-acceleration-trend" queryId="acceleration_daily" title="Team acceleration trend"
        rows={accelerationDaily} sourceRows={accelerationDaily}
        spec={{ type: "line", x: "date", y: "avgPeakAccelerationG", fields: ["avgPeakAccelerationG", "avgAccelerationRmsG"],
          startAtZero: false, valueDecimals: 2,
          colors: { avgPeakAccelerationG: "var(--chart-2)", avgAccelerationRmsG: "var(--chart-5)" },
          legend: { labels: { avgPeakAccelerationG: "Average peak", avgAccelerationRmsG: "Average RMS" } } }}
        description="Daily team-average peak and RMS acceleration magnitude from matched IMU jumps. Compare dates only when testing conditions are similar." />
      <Chart id="coach-acceleration-change" queryId="acceleration_athletes" title="Acceleration change by athlete"
        rows={accelerationChangeRows} sourceRows={accelerationAthletes}
        spec={{ type: "horizontalBar", x: "athleteLabel", y: "peakAccelerationSessionChangePct", valueDecimals: 1,
          colors: { peakAccelerationSessionChangePct: "var(--chart-2)" } }}
        description="First-to-latest change in session-average peak acceleration for athletes with repeated sessions. Direction alone is not a readiness or fatigue assessment." />
      <Table id="coach-review-queue" queryId="athletes" title="Follow-up queue" rows={reviewRows} sourceRows={reviewRowsRaw}
        columns={[column("athlete", "Athlete ID"), column("flag", "Review prompt"), column("sessions", "Sessions"), column("average", "Average height"), column("consistency", "Variation"), column("change", "Session change"), column("peakAcceleration", "Average peak acceleration"), column("accelerationChange", "Acceleration change")]}
        description="Rule-based prompts for coaching review. The list does not diagnose fatigue, injury, or readiness." />
      <Table id="coach-athlete-table" queryId="acceleration_athletes" title="Athlete performance roster" rows={rosterRows} sourceRows={rosterRowsRaw} searchable
        columns={[column("athlete", "Athlete ID"), column("sessions", "Sessions"), column("jumps", "Jumps"), column("average", "Average height"), column("best", "Best height"), column("impulse", "Relative impulse"), column("consistency", "Variation"), column("change", "Session change"), column("peakAcceleration", "Average peak acceleration"), column("accelerationVariability", "Acceleration variability"), column("accelerationChange", "Acceleration change")]}
        description="Searchable roster ordered by average jump height. Read force, acceleration, repeatability, and session coverage together." />
      <Table id="coach-jump-history" queryId="acceleration_events" title="Team jump history" rows={teamJumpRows} sourceRows={accelerationEvents} searchable
        rowKey="jumpId" selectedRowKey={selectedJumpId} onRowSelect={(selected) => setSelectedJumpId(selected.jumpId)}
        rowActionLabel={(selected) => `View jump ${selected.jump} details for ${selected.athlete} from ${selected.date}`}
        columns={[column("athlete", "Athlete ID"), column("date", "Date"), column("session", "Session"), column("jump", "Jump"), column("height", "Height"), column("impulse", "Relative impulse"), column("peakAcceleration", "Peak acceleration"), column("accelerationVariability", "Acceleration variability"), column("sync", "Sync error")]}
        description="Searchable team-wide matched jump history. Select a row to inspect the complete force-plate and IMU record." />
    </Canvas>
    <JumpDetailDialog jump={selectedAccelerationJump} event={selectedForceJump}
      onClose={() => setSelectedJumpId(null)} />
  </>;
}

function PredictionModelReview({ id, predictions, models, accelerationEvents, onOpenJump }) {
  const candidateModels = useMemo(() => {
    const selectedFeatureSet = models[0]?.feature_set;
    const scopedModels = selectedFeatureSet
      ? models.filter((item) => item.feature_set === selectedFeatureSet)
      : models;
    return [...new Map(scopedModels.map((item) => [item.model, item])).values()];
  }, [models]);
  const athleteOptions = useMemo(() => [...new Map(predictions.map((item) => [item.athleteId, {
    athleteId: item.athleteId,
    athleteLabel: item.athleteLabel,
  }])).values()].toSorted((a, b) => a.athleteLabel.localeCompare(b.athleteLabel)), [predictions]);
  const [athleteId, setAthleteId] = useState(() => athleteOptions[0]?.athleteId ?? "");
  const sessionOptions = useMemo(() => predictions
    .filter((item) => item.athleteId === athleteId)
    .reduce((sessions, item) => sessions.some((session) => session.sessionId === item.sessionId)
      ? sessions
      : [...sessions, { sessionId: item.sessionId, date: item.date }], []), [athleteId, predictions]);
  const [sessionId, setSessionId] = useState(() => sessionOptions[0]?.sessionId ?? "");
  const [selectedJumpId, setSelectedJumpId] = useState(null);
  const [series, setSeries] = useState({ actual: true, predicted: true });
  const [selectedModelId, setSelectedModelId] = useState(() => candidateModels[0]?.model ?? "");

  useEffect(() => {
    if (!athleteOptions.some((item) => item.athleteId === athleteId)) {
      setAthleteId(athleteOptions[0]?.athleteId ?? "");
    }
  }, [athleteId, athleteOptions]);

  useEffect(() => {
    if (!sessionOptions.some((item) => item.sessionId === sessionId)) {
      setSessionId(sessionOptions[0]?.sessionId ?? "");
    }
  }, [sessionId, sessionOptions]);

  useEffect(() => {
    if (!candidateModels.some((item) => item.model === selectedModelId)) {
      setSelectedModelId(candidateModels[0]?.model ?? "");
    }
  }, [candidateModels, selectedModelId]);

  const accelerationByJumpId = useMemo(() => new Map(accelerationEvents.map((item) => [item.jumpId, item])), [accelerationEvents]);
  const sessionPredictions = useMemo(() => predictions
    .filter((item) => item.athleteId === athleteId && item.sessionId === sessionId)
    .toSorted((a, b) => a.jumpId - b.jumpId)
    .map((item, index) => ({ ...item, jumpNumber: accelerationByJumpId.get(item.jumpId)?.jumpNumber ?? index + 1 })),
  [accelerationByJumpId, athleteId, predictions, sessionId]);

  useEffect(() => {
    if (!sessionPredictions.some((item) => item.jumpId === selectedJumpId)) {
      setSelectedJumpId(sessionPredictions[0]?.jumpId ?? null);
    }
  }, [selectedJumpId, sessionPredictions]);

  const selectedJump = sessionPredictions.find((item) => item.jumpId === selectedJumpId) ?? sessionPredictions[0];
  const selectedModel = candidateModels.find((item) => item.model === selectedModelId) ?? candidateModels[0] ?? {};
  const predictionModel = candidateModels[0] ?? {};
  const validationMean = mean(predictions.map((item) => item.actualImpulseNsKg));
  const residualSd = sampleStandardDeviation(predictions.map((item) => item.residualNsKg));
  const relativeResidualSd = Number.isFinite(residualSd) && Number.isFinite(validationMean)
    ? (residualSd / validationMean) * 100
    : null;
  const selectedModelHasJumpPredictions = selectedModel.model === predictionModel.model;
  const relativeMae = Number.isFinite(selectedModel.maeNsKg) && Number.isFinite(validationMean)
    ? (selectedModel.maeNsKg / validationMean) * 100
    : null;
  const relativeRmse = Number.isFinite(selectedModel.rmseNsKg) && Number.isFinite(validationMean)
    ? (selectedModel.rmseNsKg / validationMean) * 100
    : null;

  const chartWidth = 860;
  const chartHeight = 310;
  const chartPadding = { top: 24, right: 28, bottom: 42, left: 62 };
  const plotWidth = chartWidth - chartPadding.left - chartPadding.right;
  const plotHeight = chartHeight - chartPadding.top - chartPadding.bottom;
  const visibleValues = sessionPredictions.flatMap((item) => [
    ...(series.actual ? [item.actualImpulseNsKg] : []),
    ...(series.predicted ? [item.predictedImpulseNsKg] : []),
  ]).filter(Number.isFinite);
  const rawMinimum = visibleValues.length ? Math.min(...visibleValues) : 0;
  const rawMaximum = visibleValues.length ? Math.max(...visibleValues) : 1;
  const yPadding = Math.max((rawMaximum - rawMinimum) * 0.18, 0.08);
  const yMinimum = Math.max(0, rawMinimum - yPadding);
  const yMaximum = rawMaximum + yPadding;
  const xAt = (index) => chartPadding.left + (sessionPredictions.length > 1 ? (index / (sessionPredictions.length - 1)) * plotWidth : plotWidth / 2);
  const yAt = (value) => chartPadding.top + ((yMaximum - value) / Math.max(yMaximum - yMinimum, 0.01)) * plotHeight;
  const seriesPath = (field) => sessionPredictions.map((item, index) => `${index ? "L" : "M"} ${xAt(index)} ${yAt(item[field])}`).join(" ");
  const yTicks = Array.from({ length: 5 }, (_, index) => yMinimum + ((yMaximum - yMinimum) * index) / 4);
  const selectJump = (jumpId) => setSelectedJumpId(jumpId);
  const toggleSeries = (key) => setSeries((current) => {
    if (current[key] && Object.values(current).filter(Boolean).length === 1) return current;
    return { ...current, [key]: !current[key] };
  });

  return <SortableItem id={id} label="Impulse prediction review" kind="chart" span={12} minSpan={8} className="km-content-sized km-plot-block">
    <DataComponent variant="card" id={id} queryId="predictions" title="Impulse prediction review" kind="chart"
      sourceRows={predictions} displayRows={sessionPredictions}
      description={`Interactive out-of-fold actual-versus-predicted relative impulse from ${predictionModel.model?.replaceAll("_", " ") ?? "the retained model"}. Model metrics use grouped cross-validation; no validated confidence interval or external holdout is available.`}>
      <div className="km-prediction-review" data-reviewed-rows>
        <div className="km-prediction-controls">
          <div className="km-prediction-filter">
            <Dropdown label="Athlete" showLabel value={athleteId}
              choices={athleteOptions.map((item) => item.athleteId)} onChange={setAthleteId}
              formatChoice={(value) => athleteOptions.find((item) => item.athleteId === value)?.athleteLabel ?? value} />
          </div>
          <div className="km-prediction-filter">
            <Dropdown label="Session" showLabel value={sessionId}
              choices={sessionOptions.map((item) => item.sessionId)} onChange={setSessionId}
              formatChoice={(value) => {
                const session = sessionOptions.find((item) => item.sessionId === value);
                return session ? `${session.date} · ${session.sessionId}` : value;
              }} />
          </div>
          <div className="km-prediction-series" role="group" aria-label="Chart series">
            <span>Series</span>
            <button type="button" data-active={series.actual || undefined} onClick={() => toggleSeries("actual")}
              aria-pressed={series.actual}><i className="km-series-dot km-series-actual" />Actual</button>
            <button type="button" data-active={series.predicted || undefined} onClick={() => toggleSeries("predicted")}
              aria-pressed={series.predicted}><i className="km-series-dot km-series-predicted" />Predicted</button>
          </div>
        </div>

        <section className="km-prediction-chart-panel" aria-labelledby="km-prediction-chart-title">
          <header>
            <div><h3 id="km-prediction-chart-title">Actual vs predicted relative impulse</h3>
              <p>{sessionPredictions.length} matched jump{sessionPredictions.length === 1 ? "" : "s"} · click a point or jump below to inspect it</p></div>
            <span className="km-research-badge">Grouped-CV research result</span>
          </header>
          {sessionPredictions.length ? <>
            <div className="km-prediction-svg-wrap">
              <svg className="km-prediction-svg" viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img"
                aria-label="Line chart comparing actual and predicted relative impulse for each jump in the selected session">
                <text className="km-chart-axis-title" transform={`translate(18 ${chartPadding.top + plotHeight / 2}) rotate(-90)`}>Relative impulse (N·s/kg)</text>
                {yTicks.map((tick) => <g key={tick}>
                  <line className="km-chart-gridline" x1={chartPadding.left} x2={chartWidth - chartPadding.right} y1={yAt(tick)} y2={yAt(tick)} />
                  <text className="km-chart-y-label" x={chartPadding.left - 10} y={yAt(tick) + 4}>{fmt(tick, 2)}</text>
                </g>)}
                {series.actual ? <path className="km-chart-line km-chart-line-actual" d={seriesPath("actualImpulseNsKg")} /> : null}
                {series.predicted ? <path className="km-chart-line km-chart-line-predicted" d={seriesPath("predictedImpulseNsKg")} /> : null}
                {sessionPredictions.map((item, index) => <g key={item.jumpId}>
                  <text className="km-chart-x-label" x={xAt(index)} y={chartHeight - 14}>J{item.jumpNumber}</text>
                  {series.actual ? <circle className="km-chart-point km-chart-point-actual" data-selected={item.jumpId === selectedJump?.jumpId || undefined}
                    cx={xAt(index)} cy={yAt(item.actualImpulseNsKg)} r={item.jumpId === selectedJump?.jumpId ? 7 : 5}
                    role="button" tabIndex={0} aria-label={`Jump ${item.jumpNumber}, actual ${fmt(item.actualImpulseNsKg, 3)} newton seconds per kilogram`}
                    onClick={() => selectJump(item.jumpId)} onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectJump(item.jumpId); }
                    }} /> : null}
                  {series.predicted ? <circle className="km-chart-point km-chart-point-predicted" data-selected={item.jumpId === selectedJump?.jumpId || undefined}
                    cx={xAt(index)} cy={yAt(item.predictedImpulseNsKg)} r={item.jumpId === selectedJump?.jumpId ? 7 : 5}
                    role="button" tabIndex={0} aria-label={`Jump ${item.jumpNumber}, predicted ${fmt(item.predictedImpulseNsKg, 3)} newton seconds per kilogram`}
                    onClick={() => selectJump(item.jumpId)} onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectJump(item.jumpId); }
                    }} /> : null}
                </g>)}
              </svg>
            </div>
            <div className="km-jump-selector" role="group" aria-label="Session jumps">
              {sessionPredictions.map((item) => <button type="button" key={item.jumpId}
                data-selected={item.jumpId === selectedJump?.jumpId || undefined} onClick={() => selectJump(item.jumpId)}>
                <span>Jump {item.jumpNumber}</span><strong>{fmt(item.actualImpulseNsKg, 2)}</strong><small>N·s/kg actual</small>
              </button>)}
            </div>
            {selectedJump ? <div className="km-selected-prediction" aria-live="polite">
              <div><span>Selected</span><strong>Jump {selectedJump.jumpNumber}</strong></div>
              <div><span>Actual</span><strong>{fmt(selectedJump.actualImpulseNsKg, 3)} N·s/kg</strong></div>
              <div><span>Predicted</span><strong>{fmt(selectedJump.predictedImpulseNsKg, 3)} N·s/kg</strong></div>
              <div><span>Residual</span><strong>{selectedJump.residualNsKg > 0 ? "+" : ""}{fmt(selectedJump.residualNsKg, 3)} N·s/kg</strong></div>
              <button type="button" onClick={() => onOpenJump(selectedJump.jumpId)}>View full jump details</button>
            </div> : null}
          </> : <p className="km-prediction-empty">No retained predictions are available for this athlete and session.</p>}
        </section>

        <section className="km-model-review" aria-labelledby="km-model-review-title">
          <header><div><h3 id="km-model-review-title">Candidate model comparison</h3>
            <p>Errors are grouped-CV results. Relative errors use the validation-set mean actual impulse as the denominator.</p></div>
            <span>{candidateModels.length} candidates</span>
          </header>
          <div className="km-model-summary">
            <div><span>Selected model</span><strong>{selectedModel.model?.replaceAll("_", " ") ?? "—"}</strong></div>
            <div><span>MAE</span><strong>{fmt(selectedModel.maeNsKg, 3)} N·s/kg</strong><small>{fmt(relativeMae, 2)}% relative</small></div>
            <div><span>RMSE</span><strong>{fmt(selectedModel.rmseNsKg, 3)} N·s/kg</strong><small>{fmt(relativeRmse, 2)}% relative</small></div>
            <div><span>R²</span><strong>{fmt(selectedModel.r2, 3)}</strong><small>Grouped by athlete</small></div>
            <div><span>Residual SD</span><strong>{selectedModelHasJumpPredictions ? `${fmt(residualSd, 3)} N·s/kg` : "Not retained"}</strong>
              <small>{selectedModelHasJumpPredictions ? `${fmt(relativeResidualSd, 2)}% relative SD` : "Jump-level predictions unavailable"}</small></div>
          </div>
          <div className="km-model-table-wrap">
            <table className="km-model-table">
              <thead><tr><th>Model</th><th>MAE</th><th>Relative MAE</th><th>RMSE</th><th>Relative RMSE</th><th>R²</th><th>Residual SD</th></tr></thead>
              <tbody>{candidateModels.map((model) => {
                const hasJumpPredictions = model.model === predictionModel.model;
                return <tr key={model.model} data-selected={model.model === selectedModel.model || undefined}>
                  <th scope="row"><button type="button" onClick={() => setSelectedModelId(model.model)}>{model.model.replaceAll("_", " ")}</button></th>
                  <td>{fmt(model.maeNsKg, 3)}</td>
                  <td>{fmt((model.maeNsKg / validationMean) * 100, 2)}%</td>
                  <td>{fmt(model.rmseNsKg, 3)}</td>
                  <td>{fmt((model.rmseNsKg / validationMean) * 100, 2)}%</td>
                  <td>{fmt(model.r2, 3)}</td>
                  <td>{hasJumpPredictions ? `${fmt(residualSd, 3)} (${fmt(relativeResidualSd, 2)}%)` : "—"}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
          <p className="km-model-footnote">Residual SD and the interactive jump trace are shown only for {predictionModel.model?.replaceAll("_", " ") ?? "the leading model"}, because other candidates do not retain jump-level out-of-fold predictions in the reviewed artifact. No confidence band is shown because validated prediction intervals are not available.</p>
        </section>
      </div>
    </DataComponent>
  </SortableItem>;
}

const reviewReasonChoices = [
  ["event_alignment", "Event-alignment correction"],
  ["athlete_assignment", "Athlete assignment correction"],
  ["source_transcription", "Source transcription correction"],
  ["exclude_invalid_trial", "Exclude invalid trial"],
];
const reviewOutcomeChoices = [
  ["correction_verified", "Correction verified"],
  ["second_review", "Needs second review"],
  ["reprocess", "Return for reprocessing"],
];
const reviewDecisionChoices = [
  ["accepted", "Accepted"],
  ["excluded", "Excluded"],
  ["needs_reprocessing", "Needs reprocessing"],
];

function SignalTracePlot({ trace, kind }) {
  const isForce = kind === "force";
  const points = useMemo(() => {
    try {
      const parsed = JSON.parse(isForce ? trace?.forceTraceJson ?? "[]" : trace?.imuTraceJson ?? "[]");
      return Array.isArray(parsed) ? parsed.filter((point) => Array.isArray(point) && point.length === 2
        && Number.isFinite(point[0]) && Number.isFinite(point[1])) : [];
    } catch {
      return [];
    }
  }, [trace, isForce]);
  const [activeIndex, setActiveIndex] = useState(0);
  const title = isForce ? "Force-plate signal" : "IMU acceleration";
  const unit = isForce ? "N" : "g";
  const sourceFile = isForce ? trace?.forceSourceFile : trace?.imuSourceFile;
  const markers = isForce ? [
    { label: "Movement onset", time: trace?.forceMovementOnsetS, color: "#8b5cf6" },
    { label: "Takeoff", time: trace?.forceTakeoffS, color: "#0284c7" },
    { label: "Landing", time: trace?.forceLandingS, color: "#e11d48" },
  ] : [
    { label: "Takeoff", time: trace?.imuTakeoffS, color: "#0284c7" },
    { label: "Free-fall end", time: trace?.imuFreefallEndS, color: "#8b5cf6" },
    { label: "Landing", time: trace?.imuLandingS, color: "#e11d48" },
  ];
  if (!points.length) return <div className="km-signal-empty">{title} is unavailable for this jump.</div>;

  const width = 800;
  const height = 240;
  const left = 58;
  const right = 18;
  const top = 16;
  const bottom = 40;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const times = points.map(([time]) => time);
  const values = points.map(([, value]) => value);
  const markerTimes = markers.map((marker) => marker.time).filter(Number.isFinite);
  const xMin = Math.min(0, ...times, ...markerTimes);
  const xMax = Math.max(0, ...times, ...markerTimes);
  const yMin = Math.min(0, ...values);
  const reference = isForce ? trace?.bodyweightN : trace?.imuThresholdG;
  const yMax = Math.max(1, ...values, Number.isFinite(reference) ? reference : 0) * 1.08;
  const x = (time) => left + ((time - xMin) / Math.max(xMax - xMin, 0.001)) * plotWidth;
  const y = (value) => top + plotHeight - ((value - yMin) / Math.max(yMax - yMin, 0.001)) * plotHeight;
  const path = points.map(([time, value], index) => `${index ? "L" : "M"}${x(time).toFixed(1)} ${y(value).toFixed(1)}`).join(" ");
  const active = points[Math.min(activeIndex, points.length - 1)];
  const lineColor = isForce ? "#2563eb" : "#0891b2";
  const referenceLabel = isForce ? "Body weight" : "IMU threshold";
  const valueDigits = isForce ? 0 : 2;
  const timeLabel = (time) => `${time > 0 ? "+" : ""}${fmt(time, 2)} s`;
  const inspectAt = (clientX, svg) => {
    const box = svg.getBoundingClientRect();
    const svgX = ((clientX - box.left) / box.width) * width;
    let nearest = 0;
    for (let index = 1; index < points.length; index += 1) {
      if (Math.abs(x(points[index][0]) - svgX) < Math.abs(x(points[nearest][0]) - svgX)) nearest = index;
    }
    setActiveIndex(nearest);
  };

  return <figure className="km-signal-figure">
    <figcaption><strong>{title}</strong><span>{isForce ? "Processed vertical force" : "Filtered acceleration magnitude"} · {unit}</span></figcaption>
    <span className="km-signal-scroll-hint">Swipe sideways to see the full trace</span>
    <svg viewBox={`0 0 ${width} ${height}`} role="group" tabIndex={0}
      aria-label={`${title}. Use left and right arrow keys or move the pointer to inspect samples.`}
      onPointerMove={(event) => inspectAt(event.clientX, event.currentTarget)}
      onKeyDown={(event) => {
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          event.preventDefault();
          setActiveIndex((current) => Math.max(0, Math.min(points.length - 1, current + (event.key === "ArrowRight" ? 1 : -1))));
        }
      }}>
      <title>{title} for jump {trace.jumpId}, relative to detected takeoff</title>
      {[0, 0.5, 1].map((fraction) => {
        const value = yMin + (yMax - yMin) * fraction;
        return <g key={fraction}><line className="km-signal-gridline" x1={left} x2={width - right} y1={y(value)} y2={y(value)} />
          <text className="km-signal-axis-label" x={left - 9} y={y(value) + 4} textAnchor="end">{fmt(value, valueDigits)}</text></g>;
      })}
      {[xMin, 0, xMax].filter((time, index, list) => list.findIndex((item) => Math.abs(item - time) < 0.01) === index)
        .map((time) => <text key={time} className="km-signal-axis-label" x={x(time)} y={height - 13}
          textAnchor={time === xMin ? "start" : time === xMax ? "end" : "middle"}>{timeLabel(time)}</text>)}
      {Number.isFinite(reference) ? <line className="km-signal-reference" x1={left} x2={width - right} y1={y(reference)} y2={y(reference)} /> : null}
      {markers.filter((marker) => Number.isFinite(marker.time)).map((marker) =>
        <line key={marker.label} className="km-signal-marker" x1={x(marker.time)} x2={x(marker.time)} y1={top} y2={top + plotHeight}
          stroke={marker.color} />)}
      <path className="km-signal-path" d={path} stroke={lineColor} />
      <circle className="km-signal-active-point" cx={x(active[0])} cy={y(active[1])} r="5" fill={lineColor} />
    </svg>
    <div className="km-signal-inspection"><span>Selected sample</span><strong>{timeLabel(active[0])} · {fmt(active[1], valueDigits)} {unit}</strong></div>
    <div className="km-signal-legend">
      {markers.filter((marker) => Number.isFinite(marker.time)).map((marker) => <span key={marker.label}><i style={{ backgroundColor: marker.color }} />{marker.label} {timeLabel(marker.time)}</span>)}
      {Number.isFinite(reference) ? <span><i className="km-signal-reference-key" />{referenceLabel} {fmt(reference, valueDigits)} {unit}</span> : null}
    </div>
    <p className="km-signal-source">Source: {sourceFile || "Not recorded"}. Times are relative to this sensor's detected takeoff; the traces are downsampled review previews.</p>
  </figure>;
}

function ScientistDataReview({ id, events, predictions, signalTraces, persistent }) {
  const flaggedEvents = useMemo(() => events
    .filter((item) => Number(item.flightTimeDiffMs) > 100)
    .toSorted((a, b) => b.flightTimeDiffMs - a.flightTimeDiffMs), [events]);
  const traceByJumpId = useMemo(() => new Map(signalTraces.map((item) => [item.jumpId, item])), [signalTraces]);
  const [reviews, setReviews] = useState([]);
  const [queueFilter, setQueueFilter] = useState("needs_review");
  const [page, setPage] = useState(0);
  const [selectedJumpId, setSelectedJumpId] = useState(flaggedEvents[0]?.jumpId ?? null);
  const [proposedImuFlightTimeMs, setProposedImuFlightTimeMs] = useState("");
  const [decision, setDecision] = useState("accepted");
  const [reason, setReason] = useState("event_alignment");
  const [outcome, setOutcome] = useState("correction_verified");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [showSignals, setShowSignals] = useState(false);
  const [signalView, setSignalView] = useState("both");
  const pageSize = 5;
  const reviewByJumpId = useMemo(() => new Map(reviews.map((item) => [item.jumpId, item])), [reviews]);
  const filteredEvents = useMemo(() => flaggedEvents.filter((item) => {
    const status = reviewByJumpId.get(item.jumpId)?.status;
    if (queueFilter === "draft") return status === "draft";
    if (queueFilter === "submitted") return status === "submitted";
    if (queueFilter === "needs_review") return !status;
    return true;
  }), [flaggedEvents, queueFilter, reviewByJumpId]);
  const pageCount = Math.max(1, Math.ceil(filteredEvents.length / pageSize));
  const pageEvents = filteredEvents.slice(page * pageSize, (page + 1) * pageSize);
  const selectedEvent = filteredEvents.find((item) => item.jumpId === selectedJumpId) ?? pageEvents[0] ?? filteredEvents[0];
  const selectedReview = selectedEvent ? reviewByJumpId.get(selectedEvent.jumpId) : null;
  const selectedPrediction = selectedEvent ? predictions.find((item) => item.jumpId === selectedEvent.jumpId) : null;
  const selectedTrace = selectedEvent ? traceByJumpId.get(selectedEvent.jumpId) : null;
  const selectedIndex = selectedEvent ? filteredEvents.findIndex((item) => item.jumpId === selectedEvent.jumpId) : -1;
  const proposedImu = Number(proposedImuFlightTimeMs);
  const proposedDifference = selectedEvent && Number.isFinite(proposedImu)
    ? Math.abs(selectedEvent.forceFlightTimeMs - proposedImu)
    : null;
  const duplicateCount = selectedEvent ? events.filter((item) => item.jumpId === selectedEvent.jumpId).length - 1 : 0;
  const reasonLabel = reviewReasonChoices.find(([value]) => value === reason)?.[1] ?? reason;

  useEffect(() => {
    if (!persistent) return undefined;
    const controller = new AbortController();
    fetch("/api/scientist-reviews", { signal: controller.signal })
      .then(async (response) => {
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.error || "The review ledger could not be loaded.");
        setReviews(result.reviews ?? []);
      })
      .catch((reasonCaught) => {
        if (reasonCaught?.name !== "AbortError") setLoadError(reasonCaught instanceof Error ? reasonCaught.message : "The review ledger could not be loaded.");
      });
    return () => controller.abort();
  }, [persistent]);

  useEffect(() => {
    setPage((current) => Math.min(current, pageCount - 1));
  }, [pageCount]);

  useEffect(() => {
    if (!selectedEvent || !pageEvents.some((item) => item.jumpId === selectedEvent.jumpId)) {
      setSelectedJumpId(pageEvents[0]?.jumpId ?? filteredEvents[0]?.jumpId ?? null);
    }
  }, [filteredEvents, pageEvents, selectedEvent]);

  useEffect(() => {
    if (!selectedEvent) return;
    setProposedImuFlightTimeMs(String(selectedReview?.proposed?.imuFlightTimeMs ?? selectedEvent.imuFlightTimeMs ?? ""));
    setDecision(selectedReview?.proposed?.decision ?? "accepted");
    setReason(selectedReview?.reason ?? "event_alignment");
    setOutcome(selectedReview?.outcome ?? "correction_verified");
    setNote(selectedReview?.note ?? "");
  }, [selectedEvent, selectedReview]);

  useEffect(() => {
    setMessage("");
    setError("");
  }, [selectedEvent?.jumpId]);

  const selectEvent = (event) => {
    setSelectedJumpId(event.jumpId);
    const index = filteredEvents.findIndex((item) => item.jumpId === event.jumpId);
    if (index >= 0) setPage(Math.floor(index / pageSize));
  };
  const selectOffset = (offset) => {
    if (!filteredEvents.length) return;
    const nextIndex = (Math.max(selectedIndex, 0) + offset + filteredEvents.length) % filteredEvents.length;
    selectEvent(filteredEvents[nextIndex]);
  };
  const resetProposed = () => {
    if (!selectedEvent) return;
    setProposedImuFlightTimeMs(String(selectedReview?.proposed?.imuFlightTimeMs ?? selectedEvent.imuFlightTimeMs ?? ""));
    setDecision(selectedReview?.proposed?.decision ?? "accepted");
    setReason(selectedReview?.reason ?? "event_alignment");
    setOutcome(selectedReview?.outcome ?? "correction_verified");
    setNote(selectedReview?.note ?? "");
    setMessage("Proposed changes were discarded.");
    setError("");
  };
  const saveReview = async (status) => {
    if (!selectedEvent || !Number.isFinite(proposedImu)) {
      setError("Enter a valid proposed IMU flight time.");
      return;
    }
    if (status === "submitted" && !note.trim()) {
      setError("Add a reviewer note before submitting the correction.");
      return;
    }
    setSaving(true);
    setMessage("");
    setError("");
    const payload = {
      jumpId: selectedEvent.jumpId,
      proposedImuFlightTimeMs: proposedImu,
      decision,
      reason,
      outcome,
      note: note.trim(),
      status,
    };
    try {
      let saved;
      if (persistent) {
        const response = await fetch("/api/scientist-reviews", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.error || "The scientist review could not be saved.");
        saved = result.review;
      } else {
        const now = new Date().toISOString();
        saved = {
          reviewId: selectedReview?.reviewId ?? `jump-${selectedEvent.jumpId}`,
          jumpId: selectedEvent.jumpId,
          athleteId: selectedEvent.athleteId,
          sessionId: selectedEvent.sessionId,
          original: {
            forceFlightTimeMs: selectedEvent.forceFlightTimeMs,
            imuFlightTimeMs: selectedEvent.imuFlightTimeMs,
            flightTimeDiffMs: selectedEvent.flightTimeDiffMs,
            decision: "accepted",
          },
          proposed: {
            imuFlightTimeMs: proposedImu,
            flightTimeDiffMs: Math.abs(selectedEvent.forceFlightTimeMs - proposedImu),
            decision,
          },
          reason,
          reasonLabel,
          outcome,
          note: note.trim(),
          status,
          reviewer: { displayName: "Local preview" },
          createdAt: selectedReview?.createdAt ?? now,
          updatedAt: now,
        };
      }
      setReviews((current) => [...current.filter((item) => item.jumpId !== saved.jumpId), saved]);
      setQueueFilter(status === "draft" ? "draft" : "submitted");
      setMessage(`${status === "draft" ? "Draft saved" : "Correction submitted"}${persistent ? " to the audit ledger" : " for this preview session"}. Source measurements remain unchanged.`);
    } catch (reasonCaught) {
      setError(reasonCaught instanceof Error ? reasonCaught.message : "The scientist review could not be saved.");
    } finally {
      setSaving(false);
    }
  };

  const reviewedAt = selectedReview?.updatedAt
    ? new Date(selectedReview.updatedAt).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })
    : "Not reviewed";
  const queueChoices = [
    ["needs_review", "Needs review"],
    ["all", "All flagged records"],
    ["draft", "Drafts"],
    ["submitted", "Submitted"],
  ];

  return <SortableItem id={id} label="Scientific data review" kind="table" span={12} minSpan={8} className="km-content-sized">
    <DataComponent variant="card" id={id} queryId="events" queryIds={["events", "signal_traces"]} title="Scientific data review" kind="table"
      sourceRows={events} sourceRowsByQuery={{ events, signal_traces: selectedTrace ? [selectedTrace] : [] }} displayRows={pageEvents}
      description="Audited review queue for force-versus-IMU flight-time mismatches above 100 ms. Corrections are recorded separately; original reviewed source values are never silently overwritten.">
      <div className="km-scientist-review" data-reviewed-rows>
        <div className="km-review-integrity">
          <div><Icon name="shieldCheck" size={16} /><span><strong>Audited corrections</strong>Original values remain available in history.</span></div>
          <div><Icon name="refresh" size={16} /><span><strong>Controlled recalculation</strong>Submitted changes remain pending until the data pipeline is refreshed.</span></div>
          <span>{persistent ? "Persistent audit ledger" : "Preview session only"}</span>
        </div>
        {loadError ? <p className="km-review-alert" role="alert">{loadError}</p> : null}
        <div className="km-review-layout">
          <aside className="km-review-sidebar" aria-label="Data review queue">
            <div className="km-review-sidebar-head"><div><span>Review queue</span><h3>Flagged records</h3></div><strong>{flaggedEvents.length}</strong></div>
            <label className="km-review-filter"><span>Status</span><select value={queueFilter} onChange={(event) => { setQueueFilter(event.target.value); setPage(0); }}>
              {queueChoices.map(([value, label]) => <option value={value} key={value}>{label}</option>)}
            </select></label>
            <div className="km-review-queue" role="listbox" aria-label="Flagged records">
              {pageEvents.map((event) => {
                const review = reviewByJumpId.get(event.jumpId);
                const selected = event.jumpId === selectedEvent?.jumpId;
                return <button type="button" key={event.jumpId} role="option" aria-selected={selected}
                  className="km-review-record" data-selected={selected || undefined} onClick={() => selectEvent(event)}>
                  <span><strong>Jump {event.jumpId}</strong><em data-priority={event.flightTimeDiffMs >= 200 ? "high" : "medium"}>{event.flightTimeDiffMs >= 200 ? "High" : "Medium"}</em></span>
                  <b>{event.athleteLabel}</b><small>{event.sessionId} · Jump {event.jumpNumber}</small>
                  <i>{fmt(event.flightTimeDiffMs, 0)} ms difference{review ? ` · ${review.status}` : ""}</i>
                </button>;
              })}
              {!pageEvents.length ? <p className="km-review-empty">No records match this review status.</p> : null}
            </div>
            <footer className="km-review-queue-footer"><span>{filteredEvents.length ? `${page * pageSize + 1}–${Math.min((page + 1) * pageSize, filteredEvents.length)} of ${filteredEvents.length}` : "0 records"}</span>
              <button type="button" aria-label="Previous review page" disabled={!page} onClick={() => setPage((current) => Math.max(0, current - 1))}><Icon name="chevronLeft" /></button>
              <button type="button" aria-label="Next review page" disabled={page >= pageCount - 1} onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}><Icon name="chevronRight" /></button>
            </footer>
          </aside>

          <section className="km-review-editor" aria-label="Selected record correction editor">
            {selectedEvent ? <>
              <header className="km-review-record-head"><div><span>Selected record</span><div><h3>Jump {selectedEvent.jumpId}</h3><em data-status={selectedReview?.status ?? "needs_review"}>{selectedReview?.status === "draft" ? "Draft saved" : selectedReview?.status === "submitted" ? "Submitted" : "Needs review"}</em></div>
                <p>{selectedEvent.date} · {selectedEvent.sessionId} · Jump {selectedEvent.jumpNumber}</p></div>
                <nav aria-label="Selected review record"><button type="button" aria-label="Previous flagged record" onClick={() => selectOffset(-1)}><Icon name="chevronLeft" /></button><span>{selectedIndex + 1} of {filteredEvents.length}</span><button type="button" aria-label="Next flagged record" onClick={() => selectOffset(1)}><Icon name="chevronRight" /></button></nav>
              </header>
              <div className="km-review-meta"><div><span>Athlete</span><strong>{selectedEvent.athleteLabel}</strong></div><div><span>Source artifact</span><strong>all_matched_events.csv</strong></div><div><span>Prediction record</span><strong>{selectedPrediction ? "Available" : "Not retained"}</strong></div><div><span>Last reviewed</span><strong>{reviewedAt}</strong></div></div>
              <div className="km-review-issue"><Icon name="alertBell" size={18} /><div><strong>Flight-time mismatch</strong><p>Force and IMU flight times differ by {fmt(selectedEvent.flightTimeDiffMs, 0)} ms, above the 100 ms review threshold.</p></div><span>Review required</span></div>

              <section className="km-review-section km-review-signals">
                <header><div><h4>Force and IMU signal evidence</h4><p>Inspect takeoff, landing, and detection timing for this jump.</p></div>
                  <button type="button" className="km-review-signal-toggle" aria-expanded={showSignals} aria-controls={showSignals ? "km-review-signal-content" : undefined}
                    onClick={() => setShowSignals((current) => !current)}>{showSignals ? "Hide graphs" : "View graphs"}</button></header>
                {showSignals ? <div id="km-review-signal-content" className="km-review-signal-content">
                  <div className="km-review-signal-switch" role="group" aria-label="Signal graph view">
                    {[["both", "Both"], ["force", "Force plate"], ["imu", "IMU"]].map(([value, label]) =>
                      <button type="button" key={value} aria-pressed={signalView === value} onClick={() => setSignalView(value)}>{label}</button>)}
                  </div>
                  {selectedTrace ? <div className="km-review-signal-plots" data-view={signalView}>
                    {signalView !== "imu" ? <SignalTracePlot key={`${selectedTrace.jumpId}-force`} trace={selectedTrace} kind="force" /> : null}
                    {signalView !== "force" ? <SignalTracePlot key={`${selectedTrace.jumpId}-imu`} trace={selectedTrace} kind="imu" /> : null}
                  </div> : <p className="km-signal-empty">No force or IMU trace was retained for this jump. Review the recorded timing values below.</p>}
                </div> : null}
              </section>

              <section className="km-review-section">
                <header><div><h4>Verify and correct</h4><p>Compare the retained source value with the proposed correction.</p></div><span>Original values are read-only</span></header>
                <div className="km-review-fields km-review-fields-head" aria-hidden="true"><span>Field</span><span>Original value</span><span>Proposed value</span></div>
                <div className="km-review-fields"><div><strong>Force flight time</strong><small>Detected force-plate interval</small></div><div data-label="Original value"><strong>{fmt(selectedEvent.forceFlightTimeMs, 0)} ms</strong><small>Force plate</small></div><div className="km-review-readonly" data-label="Proposed value"><Icon name="lock" size={14} />Reference field</div></div>
                <label className="km-review-fields"><span><strong>IMU flight time</strong><small>Detected inertial interval</small></span><span data-label="Original value"><strong>{fmt(selectedEvent.imuFlightTimeMs, 0)} ms</strong><small>IMU detector</small></span><span className="km-review-control" data-label="Proposed value"><input type="number" min="0" max="5000" step="1" value={proposedImuFlightTimeMs} onChange={(event) => setProposedImuFlightTimeMs(event.target.value)} aria-label="Proposed IMU flight time in milliseconds" /><em>ms</em></span></label>
                <label className="km-review-fields"><span><strong>Record decision</strong><small>Controls analytical inclusion</small></span><span data-label="Original value"><strong>Accepted</strong><small>Current pipeline state</small></span><span className="km-review-control" data-label="Proposed value"><select value={decision} onChange={(event) => setDecision(event.target.value)} aria-label="Proposed record decision">{reviewDecisionChoices.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></span></label>
                <div className="km-review-fields" data-derived><div><strong>Relative impulse</strong><small>Force-plate outcome</small></div><div data-label="Original value"><strong>{fmt(selectedEvent.impulseNsKg, 3)} N·s/kg</strong><small>Reviewed source metric</small></div><div className="km-review-readonly" data-label="Proposed value"><Icon name="refresh" size={14} />Recalculated after approval</div></div>
              </section>

              <div className="km-review-validation" aria-live="polite"><div data-valid={Number.isFinite(proposedDifference) && proposedDifference <= 100 || undefined}><span>Proposed flight-time difference</span><strong>{Number.isFinite(proposedDifference) ? `${fmt(proposedDifference, 0)} ms` : "Invalid"}</strong><small>{Number.isFinite(proposedDifference) && proposedDifference <= 100 ? "Within 100 ms threshold" : "Exceeds 100 ms threshold"}</small></div><div data-valid={!duplicateCount || undefined}><span>Identity validation</span><strong>{duplicateCount ? "Review" : "Pass"}</strong><small>{duplicateCount ? `${duplicateCount} duplicate ID record(s)` : "No duplicate jump ID"}</small></div><div><span>Downstream impact</span><strong>{selectedPrediction ? "1 prediction" : "Matched event"}</strong><small>Refresh required after approval</small></div></div>

              <section className="km-review-section km-review-documentation">
                <header><div><h4>Document the correction</h4><p>A reason and reviewer note are required for submission.</p></div><span>Audit trail required</span></header>
                <div><label><span>Reason category</span><select value={reason} onChange={(event) => setReason(event.target.value)}>{reviewReasonChoices.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><label><span>Review outcome</span><select value={outcome} onChange={(event) => setOutcome(event.target.value)}>{reviewOutcomeChoices.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><label data-note><span>Reviewer note</span><textarea rows={3} maxLength={800} value={note} onChange={(event) => setNote(event.target.value)} placeholder="Describe the evidence used and the proposed correction." /></label></div>
              </section>

              <div className="km-review-audit"><div><Icon name="textDocument" size={16} /><span><strong>Audit entry preview</strong><small>Old value, proposed value, reason, reviewer identity, and timestamp are retained.</small></span></div><div><span>IMU flight time</span><strong>{fmt(selectedEvent.imuFlightTimeMs, 0)} ms → {Number.isFinite(proposedImu) ? `${fmt(proposedImu, 0)} ms` : "—"}</strong><small>{reasonLabel}</small></div></div>
              {error ? <p className="km-review-alert" role="alert">{error}</p> : null}
              {message ? <p className="km-review-message" aria-live="polite">{message}</p> : null}
              <footer className="km-review-actions"><button type="button" data-ghost onClick={resetProposed} disabled={saving}>Discard changes</button><span /><button type="button" onClick={() => void saveReview("draft")} disabled={saving}>{saving ? "Saving…" : "Save draft"}</button><button type="button" data-primary onClick={() => void saveReview("submitted")} disabled={saving}>Submit correction</button></footer>
            </> : <div className="km-review-no-selection"><h3>No review record selected</h3><p>Choose another status or return after new quality flags are generated.</p></div>}
          </section>
        </div>
      </div>
    </DataComponent>
  </SortableItem>;
}

function ScientistView({ queries, renameControl, reviewPersistent }) {
  const models = queries.model_performance?.rows ?? [];
  const predictions = queries.predictions?.rows ?? [];
  const correlations = queries.feature_correlations?.rows ?? [];
  const accelerationEvents = queries.acceleration_events?.rows ?? [];
  const signalTraces = queries.signal_traces?.rows ?? [];
  const accelerationCorrelations = queries.acceleration_correlations?.rows ?? [];
  const sessionQuality = queries.session_quality?.rows ?? [];
  const quality = queries.quality?.rows?.[0] ?? {};
  const validation = queries.model_validation?.rows?.[0] ?? {};
  const events = queries.events?.rows ?? [];
  const [selectedJumpId, setSelectedJumpId] = useState(null);
  const forceEventByJumpId = new Map(events.map((item) => [item.jumpId, item]));
  const selectedAccelerationJump = accelerationEvents.find((item) => item.jumpId === selectedJumpId);
  const selectedForceJump = forceEventByJumpId.get(selectedJumpId);
  const best = models[0] ?? {};
  const residualRows = predictions.map((item) => ({ ...item, absoluteResidualNsKg: Math.abs(item.residualNsKg) }));
  const qualityRows = [
    { check: "Athlete ID completeness", result: `${fmt(quality.athleteIdCompletenessPct, 0)}%`, interpretation: "Matched rows map to a project athlete ID" },
    { check: "Duplicate jump IDs", result: `${quality.duplicateJumpIds ?? "—"}`, interpretation: "Expected: zero" },
    { check: "Missing selected metrics", result: `${quality.missingMetricCells ?? "—"}`, interpretation: "Expected: zero across dashboard measures" },
    { check: "Missing acceleration cells", result: `${quality.missingAccelerationCells ?? "—"}`, interpretation: "Expected: zero across the reviewed IMU acceleration feature set" },
    { check: "Repeat-session coverage", result: `${fmt(quality.repeatCoveragePct, 1)}%`, interpretation: `${quality.repeatedAthletes ?? 0} athletes support longitudinal change; current coverage is ${quality.longitudinalCoverageStatus ?? "unknown"}` },
    { check: "IMU detector settings", result: validation.thresholdMode === "fixed" ? "Fixed" : "Exploratory / unknown", interpretation: "Deployment evidence requires detector settings locked before evaluation" },
    { check: "External athlete holdout", result: validation.externalHoldoutEvaluated ? "Completed" : "Not completed", interpretation: validation.deploymentReady ? "Deployment gate passed" : "Model remains a research benchmark" },
    { check: "Median absolute sync error", result: `${fmt(quality.medianSyncErrorMs, 2)} ms`, interpretation: "Force-to-IMU temporal alignment" },
    { check: "Maximum absolute sync error", result: `${fmt(quality.maxSyncErrorMs, 2)} ms`, interpretation: "All reviewed matches remain at or below 100 ms" },
    { check: "Force-event retention", result: `${fmt(quality.forceEventRetentionPct, 1)}%`, interpretation: `${quality.forceEventsRejected ?? 0} of ${quality.forceEventsDetected ?? 0} force detections were rejected` },
    { check: "IMU-event retention", result: `${fmt(quality.imuEventRetentionPct, 1)}%`, interpretation: `${quality.imuEventsRejected ?? 0} of ${quality.imuEventsDetected ?? 0} IMU detections were rejected` },
    { check: "Low-yield sessions", result: `${quality.lowForceRetentionSessions ?? 0}`, interpretation: "Sessions retaining fewer than 80% of force detections" },
    { check: "Median flight-time difference", result: `${fmt(quality.medianFlightTimeDiffMs, 1)} ms`, interpretation: `${fmt(quality.flightTimeDiffOver100msPct, 1)}% exceed 100 ms; ${fmt(quality.flightTimeDiffOver200msPct, 1)}% exceed 200 ms` },
    { check: "Model artifact provenance", result: `v${validation.modelArtifactVersion ?? "—"} · ${validation.modelTrainingRows ?? "—"} rows`, interpretation: `${validation.modelFeatureCount ?? "—"} approved IMU features; dataset ${validation.modelDatasetSha256?.slice(0, 12) ?? "unavailable"}` },
  ];
  const eventRows = events.map((item) => ({
    jumpId: item.jumpId,
    jump: item.jumpId,
    athlete: item.athleteLabel,
    session: item.sessionId,
    date: item.date,
    height: `${fmt(item.jumpHeightCm, 2)} cm`,
    impulse: `${fmt(item.impulseNsKg, 3)} N·s/kg`,
    peakForce: `${fmt(item.peakForceNkg, 2)} N/kg`,
    sync: `${fmt(item.syncErrorMs, 2)} ms`,
    flightDifference: `${fmt(item.flightTimeDiffMs, 1)} ms`,
  }));
  const accelerationAssociationRows = accelerationCorrelations
    .toSorted((a, b) => b.absoluteCorrelation - a.absoluteCorrelation)
    .slice(0, 12);
  const accelerationRows = accelerationEvents.map((item) => ({
    jumpId: item.jumpId,
    jump: item.jumpId,
    athlete: item.athleteLabel,
    session: item.sessionId,
    date: item.date,
    jumpNumber: item.jumpNumber,
    flightMinimum: `${fmt(item.flightMinAccelerationG, 3)} g`,
    xMean: `${fmt(item.xAccelerationMeanG, 3)} g`,
    xStd: `${fmt(item.xAccelerationStdG, 3)} g`,
    xMin: `${fmt(item.xAccelerationMinG, 3)} g`,
    xMax: `${fmt(item.xAccelerationMaxG, 3)} g`,
    xRange: `${fmt(item.xAccelerationRangeG, 3)} g`,
    xRms: `${fmt(item.xAccelerationRmsG, 3)} g`,
    yMean: `${fmt(item.yAccelerationMeanG, 3)} g`,
    yStd: `${fmt(item.yAccelerationStdG, 3)} g`,
    yMin: `${fmt(item.yAccelerationMinG, 3)} g`,
    yMax: `${fmt(item.yAccelerationMaxG, 3)} g`,
    yRange: `${fmt(item.yAccelerationRangeG, 3)} g`,
    yRms: `${fmt(item.yAccelerationRmsG, 3)} g`,
    zMean: `${fmt(item.zAccelerationMeanG, 3)} g`,
    zStd: `${fmt(item.zAccelerationStdG, 3)} g`,
    zMin: `${fmt(item.zAccelerationMinG, 3)} g`,
    zMax: `${fmt(item.zAccelerationMaxG, 3)} g`,
    zRange: `${fmt(item.zAccelerationRangeG, 3)} g`,
    zRms: `${fmt(item.zAccelerationRmsG, 3)} g`,
    magnitudeMean: `${fmt(item.accelerationMagnitudeMeanG, 3)} g`,
    magnitudeStd: `${fmt(item.accelerationMagnitudeStdG, 3)} g`,
    magnitudeMin: `${fmt(item.accelerationMagnitudeMinG, 3)} g`,
    magnitudeMax: `${fmt(item.accelerationMagnitudeMaxG, 3)} g`,
    magnitudeRange: `${fmt(item.accelerationMagnitudeRangeG, 3)} g`,
    magnitudeRms: `${fmt(item.accelerationMagnitudeRmsG, 3)} g`,
    area: `${fmt(item.accelerationAreaGs, 3)} g·s`,
  }));
  const sessionQualityRows = sessionQuality.map((item) => ({
    session: item.sessionId,
    athlete: item.athleteLabel,
    date: item.date,
    accepted: item.acceptedEvents,
    forceDetections: item.forceEventsDetected,
    forceRejected: item.forceEventsRejected,
    forceRetention: `${fmt(item.forceEventRetentionPct, 1)}%`,
    imuDetections: item.imuEventsDetected,
    imuRejected: item.imuEventsRejected,
    imuRetention: `${fmt(item.imuEventRetentionPct, 1)}%`,
    sync: `${fmt((item.medianSyncErrorS ?? 0) * 1000, 1)} ms`,
    flightDifference: `${fmt((item.medianFlightTimeDiffS ?? 0) * 1000, 1)} ms`,
    over100: `${fmt(item.flightTimeDiffOver100msPct, 1)}%`,
    over200: `${fmt(item.flightTimeDiffOver200msPct, 1)}%`,
  }));
  const rows = [
    row("scientist-metrics", ["scientist-model-r2", "scientist-model-rmse", "scientist-sync-quality", "scientist-id-coverage"], "none"),
    row("scientist-acceleration-metrics", ["scientist-acceleration-average-peak", "scientist-acceleration-max-peak", "scientist-acceleration-variability", "scientist-acceleration-area"]),
    row("scientist-review-workspace", ["scientist-data-review"]),
    row("scientist-validation", ["scientist-prediction-review"]),
    row("scientist-diagnostics", ["scientist-residuals", "scientist-feature-associations"]),
    row("scientist-acceleration-diagnostics", ["scientist-acceleration-distribution", "scientist-acceleration-associations"]),
    row("scientist-quality", ["scientist-quality-table", "scientist-event-table"]),
    row("scientist-session-quality-row", ["scientist-session-quality"]),
    row("scientist-acceleration-table", ["scientist-acceleration-explorer"]),
  ];
  return <>
    <RoleContext title="Sport scientist workspace"
      meta={`${validation.thresholdMode === "fixed" ? "Fixed IMU threshold" : "Exploratory detector"} · grouped research CV`} control={renameControl} />
    <Canvas id="kivomotion-scientist-canvas" label="Sport scientist analysis blocks" rows={rows}>
      <Metric id="scientist-model-r2" queryId="model_performance" title="Research grouped-CV R²" value={fmt(best.r2, 3)} comparison="Not deployment evidence"
        sourceRows={models} description={`Coefficient of determination from cross-validation grouped by athlete ID using ${best.model?.replaceAll("_", " ") ?? "the leading model"}. An untouched external athlete cohort is still required.`} />
      <Metric id="scientist-model-rmse" queryId="model_performance" title="Best RMSE" value={`${fmt(best.rmseNsKg, 3)} N·s/kg`} comparison="Lower is better"
        sourceRows={models} description="Root mean squared error for relative impulse predictions in grouped cross-validation." negative />
      <Metric id="scientist-sync-quality" queryId="quality" title="Median sync error" value={`${fmt(quality.medianSyncErrorMs, 2)} ms`}
        comparison={`max ${fmt(quality.maxSyncErrorMs, 2)} ms`} sourceRows={queries.quality?.rows ?? []}
        description="Median and maximum absolute timing error between matched force and IMU takeoff events." negative />
      <Metric id="scientist-id-coverage" queryId="quality" title="Athlete ID coverage" value={`${fmt(quality.athleteIdCompletenessPct, 0)}%`}
        comparison="Matched rows" sourceRows={queries.quality?.rows ?? []}
        description="Share of matched jump events with an athlete ID found in the project athlete map." />
      <Metric id="scientist-acceleration-average-peak" queryId="acceleration_events" title="Average peak acceleration"
        value={`${fmt(mean(accelerationEvents.map((item) => item.accelerationMagnitudeMaxG)), 3)} g`} comparison={`${accelerationEvents.length} matched jumps`}
        sourceRows={accelerationEvents} description="Mean event-level maximum filtered acceleration magnitude." />
      <Metric id="scientist-acceleration-max-peak" queryId="acceleration_events" title="Maximum peak acceleration"
        value={`${fmt(Math.max(...accelerationEvents.map((item) => item.accelerationMagnitudeMaxG)), 3)} g`} comparison="Dataset maximum"
        sourceRows={accelerationEvents} description="Largest event-level acceleration-magnitude maximum in the reviewed dataset." />
      <Metric id="scientist-acceleration-variability" queryId="acceleration_events" title="Mean acceleration variability"
        value={`${fmt(mean(accelerationEvents.map((item) => item.accelerationMagnitudeStdG)), 3)} g`} comparison="Within-jump SD"
        sourceRows={accelerationEvents} description="Mean of the event-level standard deviation of filtered acceleration magnitude." />
      <Metric id="scientist-acceleration-area" queryId="acceleration_events" title="Mean acceleration area"
        value={`${fmt(mean(accelerationEvents.map((item) => item.accelerationAreaGs)), 3)} g·s`} comparison="Integrated magnitude"
        sourceRows={accelerationEvents} description="Mean event-level area under the filtered acceleration-magnitude signal." />
      <ScientistDataReview id="scientist-data-review" events={events} predictions={predictions} signalTraces={signalTraces}
        persistent={reviewPersistent} />
      <PredictionModelReview id="scientist-prediction-review" predictions={predictions} models={models} accelerationEvents={accelerationEvents}
        onOpenJump={setSelectedJumpId} />
      <Chart id="scientist-residuals" queryId="predictions" title="Residual distribution" rows={residualRows} sourceRows={predictions}
        spec={{ type: "histogram", y: "residualNsKg", valueDecimals: 3,
          colors: { residualNsKg: "var(--chart-2)" } }}
        description="Observed minus predicted relative impulse. A distribution centered near zero indicates limited overall bias, while tails show larger errors." />
      <Chart id="scientist-feature-associations" queryId="feature_correlations" title="Feature association with relative impulse" rows={correlations} sourceRows={correlations}
        spec={{ type: "horizontalBar", x: "feature", y: "correlation", valueDecimals: 2, colors: { correlation: "var(--chart-3)" } }}
        description="Pearson correlation between selected IMU features and force-plate relative impulse. Associations do not establish causality." />
      <Chart id="scientist-acceleration-distribution" queryId="acceleration_events" title="Peak acceleration distribution"
        rows={accelerationEvents} sourceRows={accelerationEvents}
        spec={{ type: "histogram", y: "accelerationMagnitudeMaxG", valueDecimals: 2,
          colors: { accelerationMagnitudeMaxG: "var(--chart-5)" } }}
        description="Distribution of event-level maximum filtered acceleration magnitude across all matched jumps." />
      <Chart id="scientist-acceleration-associations" queryId="acceleration_correlations" title="Acceleration association with relative impulse"
        rows={accelerationAssociationRows} sourceRows={accelerationCorrelations}
        spec={{ type: "horizontalBar", x: "feature", y: "correlation", valueDecimals: 2, colors: { correlation: "var(--chart-4)" } }}
        description="Twelve strongest absolute Pearson associations between acceleration features and force-plate relative impulse. Association does not establish causality." />
      <Table id="scientist-quality-table" queryId="quality" title="Data-quality checks" rows={qualityRows} sourceRows={queries.quality?.rows ?? []}
        columns={[column("check", "Check"), column("result", "Result"), column("interpretation", "Interpretation")]}
        description="Quality checks applied to the dataset used by every role view." />
      <Table id="scientist-event-table" queryId="events" title="Matched event explorer" rows={eventRows} sourceRows={events} searchable
        rowKey="jumpId" selectedRowKey={selectedJumpId} onRowSelect={(selected) => setSelectedJumpId(selected.jumpId)}
        rowActionLabel={(selected) => `View matched jump ${selected.jump} details for ${selected.athlete}`}
        columns={[column("jump", "Jump ID"), column("athlete", "Athlete ID"), column("session", "Session"), column("date", "Date"), column("height", "Height"), column("impulse", "Relative impulse"), column("peakForce", "Peak force"), column("sync", "Sync error"), column("flightDifference", "Flight-time difference")]}
        description="Event-level force and IMU measures joined by matched jump ID and stable anonymized athlete ID. Select a row to inspect the complete matched record." />
      <Table id="scientist-session-quality" queryId="session_quality" title="Session detection and agreement QA"
        rows={sessionQualityRows} sourceRows={sessionQuality} searchable
        columns={[
          column("session", "Session"), column("athlete", "Athlete ID"), column("date", "Date"),
          column("accepted", "Accepted"), column("forceDetections", "Force detections"), column("forceRejected", "Force rejected"), column("forceRetention", "Force retention"),
          column("imuDetections", "IMU detections"), column("imuRejected", "IMU rejected"), column("imuRetention", "IMU retention"),
          column("sync", "Median sync error"), column("flightDifference", "Median flight-time difference"), column("over100", ">100 ms"), column("over200", ">200 ms"),
        ]}
        description="Per-session event yield and force-versus-IMU agreement. Review sessions below 80% force-event retention or with large flight-time disagreement." />
      <Table id="scientist-acceleration-explorer" queryId="acceleration_events" title="Acceleration feature explorer"
        rows={accelerationRows} sourceRows={accelerationEvents} searchable
        rowKey="jumpId" selectedRowKey={selectedJumpId} onRowSelect={(selected) => setSelectedJumpId(selected.jumpId)}
        rowActionLabel={(selected) => `View acceleration details for matched jump ${selected.jump}`}
        columns={[
          column("jump", "Jump ID"), column("athlete", "Athlete ID"), column("session", "Session"), column("date", "Date"), column("jumpNumber", "Jump"), column("flightMinimum", "Flight minimum"),
          column("xMean", "X mean"), column("xStd", "X SD"), column("xMin", "X min"), column("xMax", "X max"), column("xRange", "X range"), column("xRms", "X RMS"),
          column("yMean", "Y mean"), column("yStd", "Y SD"), column("yMin", "Y min"), column("yMax", "Y max"), column("yRange", "Y range"), column("yRms", "Y RMS"),
          column("zMean", "Z mean"), column("zStd", "Z SD"), column("zMin", "Z min"), column("zMax", "Z max"), column("zRange", "Z range"), column("zRms", "Z RMS"),
          column("magnitudeMean", "Magnitude mean"), column("magnitudeStd", "Magnitude SD"), column("magnitudeMin", "Magnitude min"), column("magnitudeMax", "Magnitude max"), column("magnitudeRange", "Magnitude range"), column("magnitudeRms", "Magnitude RMS"), column("area", "Acceleration area"),
        ]}
        description="Complete reviewed IMU acceleration feature set for each matched jump. Axis and magnitude measures are in g; acceleration area is in g·s. Select a row for the full matched detail view." />
    </Canvas>
    <JumpDetailDialog jump={selectedAccelerationJump} event={selectedForceJump}
      onClose={() => setSelectedJumpId(null)} />
  </>;
}

export function DashboardContent({ initialView = {} }) {
  const { queries, viewFocus } = useDataApp();
  const viewRef = useRef(null);
  const auth = currentAuth();
  const [athleteAliases, setAthleteAliases] = useState(() => Object.fromEntries(
    (queries.athletes?.rows ?? []).map((athlete) => [athlete.athleteId, athlete.renamedAthlete ?? ""]),
  ));
  const displayQueries = useMemo(() => Object.fromEntries(
    Object.entries(queries ?? {}).map(([queryId, query]) => [queryId, {
      ...query,
      rows: (query.rows ?? []).map((item) => {
        if (!item.athleteId || !Object.hasOwn(athleteAliases, item.athleteId)) return item;
        const renamedAthlete = athleteAliases[item.athleteId] || null;
        const generatedAthleteLabel = `Athlete ${item.athleteId.split("-")[0].toUpperCase()}`;
        return { ...item, renamedAthlete, athleteLabel: renamedAthlete ?? generatedAthleteLabel };
      }),
    }]),
  ), [athleteAliases, queries]);
  const renameAthlete = async (athleteId, label) => {
    let savedLabel = label;
    if (auth.authenticated) {
      const response = await fetch("/api/athletes/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ athleteId, renamedAthlete: label }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.error || "The athlete name could not be saved.");
      savedLabel = result.renamedAthlete ?? "";
    }
    setAthleteAliases((current) => ({ ...current, [athleteId]: savedLabel }));
  };
  const athleteOnly = auth.role === "athlete";
  const availableTabs = athleteOnly ? roleTabs.filter((item) => item.id === "athlete") : roleTabs;
  const { activeTabId } = useDashboardTabs(availableTabs);
  const requestedTab = initialView.tab ?? activeTabId ?? "athlete";
  const tab = availableTabs.some((item) => item.id === requestedTab) ? requestedTab : "athlete";
  const quality = displayQueries.quality?.rows?.[0] ?? {};
  const validation = displayQueries.model_validation?.rows?.[0] ?? {};
  const renameControl = <AthleteRenameControl athletes={queries.athletes?.rows ?? []}
    aliases={athleteAliases} onRename={renameAthlete} persistent={auth.authenticated} />;
  useEffect(() => {
    const page = document.querySelector("main.page");
    page?.setAttribute("data-data-app-content", "dashboard");
    return () => page?.removeAttribute("data-data-app-content");
  }, []);
  useLayoutEffect(() => {
    const view = viewRef.current;
    if (!view || typeof IntersectionObserver === "undefined"
      || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;

    const reveal = (card) => {
      card.dataset.motionVisible = "true";
      const plot = card.closest(".km-plot-block");
      if (plot) plot.dataset.plotVisible = "true";
    };
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(({ target, isIntersecting }) => {
        if (!isIntersecting) return;
        reveal(target);
        observer.unobserve(target);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -24px 0px" });

    view.querySelectorAll(".sortable-canvas-row > .sortable-item > .dashboard-component, .km-comparison-kpi").forEach((card) => {
      const bounds = card.getBoundingClientRect();
      if (bounds.top < window.innerHeight - 24 && bounds.bottom > 0) {
        reveal(card);
      } else {
        observer.observe(card);
      }
    });
    return () => observer.disconnect();
  }, [tab]);
  return <div className="km-page" data-role={tab}>
    <header className="km-brand-panel">
      <div className="km-brand-copy">
        <p>Welcome to</p>
        <h1><span>KivoMotion</span><em>Performance Intelligence</em></h1>
      </div>
      <div className="km-brand-actions">
        <span className="km-connection"><span className="km-live-dot" />Reviewed data</span>
        <span className="km-data-count">{quality.matchedJumps ?? 0} jumps · {quality.matchedSessions ?? 0} sessions · {quality.athletes ?? 0} athletes</span>
        <DashboardGuide activeRole={tab} quality={quality} validation={validation} athleteOnly={athleteOnly} />
        {auth.authenticated && <div className="km-account">
          <span><strong>{auth.displayName}</strong><small>{athleteOnly ? "Athlete access" : "Coach access"}</small></span>
          <form action="/logout" method="post"><button type="submit">Sign out</button></form>
        </div>}
      </div>
    </header>
    <div className="km-view" key={tab} ref={viewRef}>
      {tab === "athlete" && <AthleteView queries={displayQueries}
        initialAthleteId={athleteOnly ? auth.athleteId : initialView.athleteId ?? viewFocus?.athleteId}
        canSelectAthlete={!athleteOnly} />}
      {tab === "coach" && <CoachView queries={displayQueries} renameControl={renameControl} />}
      {tab === "scientist" && <ScientistView queries={displayQueries} renameControl={renameControl}
        reviewPersistent={auth.authenticated && auth.role === "coach"} />}
    </div>
  </div>;
}
