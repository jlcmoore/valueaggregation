window.JATOS_CONDITIONS = [
  // Current launch configuration:
  // - dependent measure: best compromise
  // - display condition: area
  // - qualification_mode: "full" (all qual items) or "area_only" (first 3 stacked)
  {
    dependent_measure: "best compromise",
    chart_type: "area",
    qualification_mode: "area_only",
  },
  // Optional variants (disabled):
  // { dependent_measure: "best compromise", chart_type: "none" },
  // { dependent_measure: "best", chart_type: "area" },
  // { dependent_measure: "best", chart_type: "none" },
];
