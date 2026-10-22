CREATE TABLE events (
  jumpId INTEGER PRIMARY KEY,
  athleteId TEXT NOT NULL,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionId TEXT NOT NULL,
  date TEXT NOT NULL,
  testNumber INTEGER,
  jumpNumber INTEGER NOT NULL,
  jumpHeightCm REAL,
  flightHeightCm REAL,
  takeoffVelocityMs REAL,
  impulseNsKg REAL,
  peakForceN REAL,
  peakForceNkg REAL,
  peakLandingForceN REAL,
  syncErrorMs REAL,
  forceFlightTimeMs REAL,
  imuFlightTimeMs REAL,
  flightTimeDiffMs REAL,
  minAccelerationG REAL,
  massKg REAL
);

CREATE INDEX idx_events_athlete_date ON events(athleteId, date);
CREATE INDEX idx_events_session ON events(sessionId);

CREATE TABLE acceleration_events (
  jumpId INTEGER PRIMARY KEY,
  athleteId TEXT NOT NULL,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionId TEXT NOT NULL,
  date TEXT NOT NULL,
  jumpNumber INTEGER NOT NULL,
  jumpHeightCm REAL,
  takeoffVelocityMs REAL,
  impulseNsKg REAL,
  syncErrorMs REAL,
  flightMinAccelerationG REAL,
  xAccelerationMeanG REAL,
  xAccelerationStdG REAL,
  xAccelerationMinG REAL,
  xAccelerationMaxG REAL,
  xAccelerationRangeG REAL,
  xAccelerationRmsG REAL,
  yAccelerationMeanG REAL,
  yAccelerationStdG REAL,
  yAccelerationMinG REAL,
  yAccelerationMaxG REAL,
  yAccelerationRangeG REAL,
  yAccelerationRmsG REAL,
  zAccelerationMeanG REAL,
  zAccelerationStdG REAL,
  zAccelerationMinG REAL,
  zAccelerationMaxG REAL,
  zAccelerationRangeG REAL,
  zAccelerationRmsG REAL,
  accelerationMagnitudeMeanG REAL,
  accelerationMagnitudeStdG REAL,
  accelerationMagnitudeMinG REAL,
  accelerationMagnitudeMaxG REAL,
  accelerationMagnitudeRangeG REAL,
  accelerationMagnitudeRmsG REAL,
  accelerationAreaGs REAL
);

CREATE INDEX idx_acceleration_events_athlete_date ON acceleration_events(athleteId, date);
CREATE INDEX idx_acceleration_events_session ON acceleration_events(sessionId);

CREATE TABLE acceleration_sessions (
  sessionId TEXT PRIMARY KEY,
  athleteId TEXT NOT NULL,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  date TEXT NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgMeanAccelerationG REAL,
  avgPeakAccelerationG REAL,
  maxPeakAccelerationG REAL,
  avgAccelerationVariabilityG REAL,
  avgAccelerationRangeG REAL,
  avgAccelerationRmsG REAL,
  avgAccelerationAreaGs REAL,
  peakAccelerationCvPct REAL
);

CREATE INDEX idx_acceleration_sessions_athlete_date ON acceleration_sessions(athleteId, date);

CREATE TABLE acceleration_athletes (
  athleteId TEXT PRIMARY KEY,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionCount INTEGER NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgMeanAccelerationG REAL,
  avgPeakAccelerationG REAL,
  maxPeakAccelerationG REAL,
  avgAccelerationVariabilityG REAL,
  avgAccelerationRangeG REAL,
  avgAccelerationRmsG REAL,
  avgAccelerationAreaGs REAL,
  peakAccelerationCvPct REAL,
  peakAccelerationSessionChangePct REAL
);

CREATE TABLE acceleration_daily (
  date TEXT PRIMARY KEY,
  sessionCount INTEGER NOT NULL,
  athleteCount INTEGER NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgPeakAccelerationG REAL,
  maxPeakAccelerationG REAL,
  avgAccelerationVariabilityG REAL,
  avgAccelerationRmsG REAL,
  avgAccelerationAreaGs REAL
);

CREATE TABLE athlete_comparison (
  athleteId TEXT PRIMARY KEY,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionCount INTEGER NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgHeightCm REAL,
  bestHeightCm REAL,
  avgVelocityMs REAL,
  avgImpulseNsKg REAL,
  avgPeakForceNkg REAL,
  firstDate TEXT,
  lastDate TEXT,
  consistencyCvPct REAL,
  sessionChangePct REAL,
  avgPeakAccelerationG REAL,
  maxPeakAccelerationG REAL,
  avgAccelerationVariabilityG REAL,
  peakAccelerationSessionChangePct REAL
);

CREATE TABLE acceleration_correlations (
  feature TEXT PRIMARY KEY,
  field TEXT NOT NULL,
  correlation REAL NOT NULL,
  absoluteCorrelation REAL NOT NULL
);

CREATE TABLE sessions (
  sessionId TEXT PRIMARY KEY,
  athleteId TEXT NOT NULL,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  date TEXT NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgHeightCm REAL,
  bestHeightCm REAL,
  avgVelocityMs REAL,
  avgImpulseNsKg REAL,
  avgPeakForceNkg REAL,
  medianSyncErrorMs REAL,
  maxSyncErrorMs REAL,
  consistencyCvPct REAL
);

CREATE INDEX idx_sessions_athlete_date ON sessions(athleteId, date);

CREATE TABLE athletes (
  athleteId TEXT PRIMARY KEY,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionCount INTEGER NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgHeightCm REAL,
  bestHeightCm REAL,
  avgVelocityMs REAL,
  avgImpulseNsKg REAL,
  avgPeakForceNkg REAL,
  firstDate TEXT,
  lastDate TEXT,
  consistencyCvPct REAL,
  sessionChangePct REAL
);

CREATE TABLE daily (
  date TEXT PRIMARY KEY,
  sessionCount INTEGER NOT NULL,
  athleteCount INTEGER NOT NULL,
  jumpCount INTEGER NOT NULL,
  avgHeightCm REAL,
  bestHeightCm REAL,
  avgImpulseNsKg REAL,
  medianSyncErrorMs REAL
);

CREATE TABLE model_performance (
  model TEXT PRIMARY KEY,
  maeNsKg REAL NOT NULL,
  rmseNsKg REAL NOT NULL,
  r2 REAL NOT NULL
);

CREATE TABLE model_validation (
  thresholdMode TEXT NOT NULL,
  imuThresholdG REAL,
  imuMinFreefallS REAL,
  groupedCrossValidation TEXT NOT NULL,
  researchReadyForUnseenAthletes INTEGER NOT NULL,
  externalHoldoutEvaluated INTEGER NOT NULL,
  deploymentReady INTEGER NOT NULL,
  deploymentBlockers TEXT NOT NULL
);

CREATE TABLE predictions (
  jumpId INTEGER PRIMARY KEY,
  athleteId TEXT NOT NULL,
  athleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  sessionId TEXT NOT NULL,
  date TEXT NOT NULL,
  actualImpulseNsKg REAL NOT NULL,
  predictedImpulseNsKg REAL NOT NULL,
  residualNsKg REAL NOT NULL
);

CREATE INDEX idx_predictions_athlete ON predictions(athleteId);

CREATE TABLE athlete_aliases (
  athleteId TEXT PRIMARY KEY,
  generatedAthleteLabel TEXT NOT NULL,
  renamedAthlete TEXT,
  athleteLabel TEXT NOT NULL
);

CREATE TABLE feature_correlations (
  feature TEXT PRIMARY KEY,
  field TEXT NOT NULL,
  correlation REAL NOT NULL,
  absoluteCorrelation REAL NOT NULL
);

CREATE TABLE quality (
  matchedSessions INTEGER NOT NULL,
  athletes INTEGER NOT NULL,
  matchedJumps INTEGER NOT NULL,
  athleteIdCompletenessPct REAL NOT NULL,
  duplicateJumpIds INTEGER NOT NULL,
  missingMetricCells INTEGER NOT NULL,
  missingAccelerationCells INTEGER NOT NULL,
  medianSyncErrorMs REAL NOT NULL,
  maxSyncErrorMs REAL NOT NULL,
  goodSyncPct REAL NOT NULL,
  repeatedAthletes INTEGER NOT NULL,
  singleSessionAthletes INTEGER NOT NULL,
  repeatCoveragePct REAL NOT NULL,
  longitudinalCoverageStatus TEXT NOT NULL,
  imuThresholdMode TEXT NOT NULL,
  externalHoldoutEvaluated INTEGER NOT NULL,
  deploymentReady INTEGER NOT NULL,
  dataStartDate TEXT NOT NULL,
  dataEndDate TEXT NOT NULL,
  snapshotGeneratedAt TEXT NOT NULL
);

-- The build script also mirrors the source summaries into raw_* tables so the
-- reviewed role-level tables can be traced back to the project outputs.
