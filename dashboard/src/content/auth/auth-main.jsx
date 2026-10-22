import React from "react";
import { createRoot } from "react-dom/client";

import "../../theme.css";
import "../../styles.css";
import "../../theme-picker.css";
import "../../source-preview.css";
import "../dashboard/dashboard.css";
import "../report/report.css";
import "../../print.css";
import "../../theme-runtime.js";
import { DataAppRuntime } from "../../DataAppRuntime.jsx";
import { DashboardContent } from "../dashboard/DashboardContent.jsx";
import { ReportContent } from "../report/ReportContent.jsx";

const reviewedSnapshot = globalThis.__KIVO_SNAPSHOT__;

if (!reviewedSnapshot?.queries) {
  throw new Error("Authenticated dashboard snapshot is unavailable.");
}

document.title = reviewedSnapshot.title || "KivoMotion — Performance Intelligence";
delete globalThis.__KIVO_SNAPSHOT__;

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <React.StrictMode>
      <DataAppRuntime
        reviewedSnapshot={reviewedSnapshot}
        DashboardContent={DashboardContent}
        ReportContent={ReportContent}
        hosted={false}
      />
    </React.StrictMode>,
  );
}
