# KivoMotion dashboard

This dashboard presents the same reviewed matched-jump dataset at three levels:

- **Athlete:** personal jump and acceleration performance in plain language
- **Coach:** team coverage, interactive two-athlete comparison, and rule-based follow-up prompts
- **Sport Scientist:** synchronization quality, acceleration features, model diagnostics, and deployment gates

## Role-based login and privacy

The authenticated server enforces two roles:

- **Coach:** receives the complete reviewed snapshot and can use Athlete, Coach, and Sport Scientist pages. The Athlete ID selector remains available, the coach view can compare any two athletes, and Coach/Sport Scientist workspaces can assign persistent display names without changing stable athlete IDs.
- **Athlete:** receives only rows assigned to their account, sees only the Athlete page, and cannot change athlete identity.

This restriction is enforced before data is sent to the browser. It is not only a hidden tab or client-side filter.

The server-rendered login page in `auth_server.py` uses the Kivo Motion geometric mark, split wordmark, movement tagline, and a dark performance visual system. Motion is decorative only, respects `prefers-reduced-motion`, and does not change the authentication form, account roles, or privacy boundary.

Create local accounts. Passwords are stored as salted scrypt hashes in the ignored `auth/users.local.json` file:

```bash
python3 manage_users.py add-coach coach@example.com --display-name "Coach Name"
python3 manage_users.py add-athlete athlete@example.com \
  --display-name "Athlete Name" \
  --athlete-id 93a9e72e-7fd1-4aed-a345-657297a97d4a
```

Refresh the synchronized OneDrive shortcut, then build and start the authenticated dashboard. This is the normal application start command and always opens at the login boundary:

```bash
npm start
```

At startup the application looks for `T17 Shared Folder/Phase 2 Data` under the macOS OneDrive `CloudStorage` directories. It compares the synchronized CSV metadata with the last successful run, reruns the impulse analysis only when inputs changed or outputs are missing, refreshes SQLite and `src/data.json`, and then starts the login server. To use a different synchronized location, set `KIVOMOTION_DATA_DIR` or run:

```bash
KIVOMOTION_DATA_DIR="/path/to/T17 Shared Folder/Phase 2 Data" npm start
```

Wait for OneDrive to show that the shortcut has finished synchronizing before starting the application. Generated source-state metadata and credentials remain local and are not committed.

Open [http://127.0.0.1:4180](http://127.0.0.1:4180). Sessions use random, HTTP-only, same-site cookies and expire after eight hours. Use HTTPS and `--secure-cookies` outside localhost. The included server keeps sessions in memory, so production deployment should replace that store with the platform's managed session service.

`npm run dev` remains available for trusted local authoring only. It bypasses the login page and must not be used as the normal application entry point.

## Authoring preview with automatic data refresh

From this folder:

```bash
npm ci
python3 data_pipeline/dev.py
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). The start command:

1. validates the project summary files;
2. atomically refreshes `src/data.json` and `data/kivomotion.sqlite3`;
3. starts the Vite development server;
4. watches the analysis summaries and refreshes when they change;
5. reruns the fixed-threshold batch pipeline when raw Phase 2 CSV files change.

If a refresh fails, the last valid snapshot and database remain in place and the error is printed in the terminal.

This Vite authoring preview intentionally uses coach-level project data. Use the authenticated server above when testing athlete access.

## Manual refresh and production build

Rebuild the complete analysis, validation manifest, SQLite database, and reviewed snapshot:

```bash
python3 data_pipeline/refresh_all.py
```

Refresh only the dashboard layer from existing analysis summaries, then build the frontend:

```bash
python3 data_pipeline/build_dashboard_data.py --build-status complete
npm run build
```

The dashboard browser bundle uses `src/data.json`, which is a bounded reviewed snapshot with source metadata and metric definitions. SQLite contains the same normalized tables for local inspection and downstream analysis. The database is intentionally not exposed directly to the browser because production artifacts must not contain a live local database connection.

Athlete display-name changes made from the authenticated Coach or Sport Scientist workspace are persisted in three places:

- `Phase 2 Data jumps/athlete_id_map.csv` in the `renamed_athlete` column (source of truth);
- `dashboard/data/kivomotion.sqlite3` in the `renamedAthlete` columns and `athlete_aliases` table;
- `dashboard/src/data.json` in each athlete-bearing reviewed row.

The UUID `athleteId` is never renamed and remains the identity used for joins, login authorization, and privacy filtering. Use the authenticated dashboard at port `4180` for persistent renames; the Vite authoring preview at port `4173` intentionally keeps rename experiments browser-local.

Rename persistence uses a coordinated staged commit. The CSV mapping, SQLite database and reviewed JSON snapshot are generated and validated before replacement. If any replacement fails, every artifact is restored to its previous bytes so the three stores cannot be left with different names.

## Source files

The refresh pipeline reads:

- `cmj_jump_analysis/batch_results/all_matched_events.csv`
- `cmj_jump_analysis/batch_results/all_model_features.csv`
- `cmj_jump_analysis/batch_results/batch_session_status.csv`
- `cmj_jump_analysis/batch_results/reports/model_leaderboard.csv`
- `cmj_jump_analysis/batch_results/reports/out_of_fold_predictions.csv`
- `cmj_jump_analysis/batch_results/model_validation.json`
- `Phase 2 Data jumps/athlete_id_map.csv`

The builder rejects duplicate jump IDs, mismatched jump sets, missing athlete mappings, conflicting athlete keys, duplicate source-file mappings, missing acceleration columns, and session-count disagreement. The Sport Scientist view also exposes force/IMU event retention, rejected-event counts, flight-duration agreement by session, and saved-model provenance including its dataset hash, training row count and feature count.

## Tests

From the repository root:

```bash
python -m pytest
cd dashboard
npm test
npm run verify:runtime
npm run build
```

The Node tests verify dashboard query coverage, reconciled counts, conservative deployment status, role restrictions, three audience views, and unique layout row identifiers. Python integration tests verify the transformations, coordinated three-artifact rollback, SQLite integrity, password hashing, per-IP-and-username login throttling, athlete-level snapshot isolation, stable model provenance, and stable dashboard identity.

## Current analytical limitations

- Only 4 of 51 athletes have repeated sessions, so first-to-latest change is shown only for those athletes and is labelled preliminary.
- Model results use athlete-grouped cross-validation but no untouched external cohort has been evaluated.
- The fixed detector settings were locked after reviewing the current development dataset and must be evaluated on newly collected data before deployment.

Use the dashboard guide and each card's source action for definitions, exact rows, coverage, and lineage.
