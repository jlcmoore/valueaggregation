// main.js

// Only auto-run on document ready when not using JATOS
if (!window.jatos) {
  $(document).ready(() => {
    main();
  });
}

window.runTaskMain = async function () {
  await main();
};

window.renderTaskQuestions = function () {
  afterLoadData();
};

// Example data used by charts
const base_example_data = [
  [
    { utility: 10, agent: "A", credence: 10, action: "one" },
    { utility: 20, agent: "B", credence: 10, action: "one" },
    { utility: 15, agent: "C", credence: 10, action: "one" },
  ],
  [
    { utility: 12, agent: "A", credence: 10, action: "one" },
    { utility: 14, agent: "B", credence: 10, action: "one" },
    { utility: 8, agent: "C", credence: 10, action: "one" },
    { utility: 18, agent: "A", credence: 10, action: "two" },
    { utility: 6, agent: "B", credence: 10, action: "two" },
    { utility: 10, agent: "C", credence: 10, action: "two" },
  ],
  [
    { utility: 10, agent: "A", credence: 10, action: "one" },
    { utility: 8, agent: "B", credence: 10, action: "one" },
    { utility: 14, agent: "C", credence: 10, action: "one" },
    { utility: 12, agent: "A", credence: 10, action: "two" },
    { utility: 20, agent: "B", credence: 10, action: "two" },
    { utility: 15, agent: "C", credence: 10, action: "two" },
    { utility: 6, agent: "A", credence: 10, action: "three" },
    { utility: 9, agent: "B", credence: 10, action: "three" },
    { utility: 11, agent: "C", credence: 10, action: "three" },
  ],
  [
    { utility: 10, agent: "", credence: 1, action: "x" },
    { utility: 10, agent: "", credence: 2, action: "x" },
    { utility: 10, agent: "elephant", credence: 3, action: "x" },
  ],
];

const example_data = [
  base_example_data[0],
  base_example_data[1],
  base_example_data[2],
  base_example_data[3],
  base_example_data[0],
  base_example_data[1],
  base_example_data[2],
  base_example_data[1],
  base_example_data[2],
];

const color_scheme = d3.schemeCategory10;
const chart_funcs = {
  volume: volume_chart,
  area: area_chart,
  both: both_charts,
};

// Globals populated after loading
let variables;
let chart_func = null;
let maximize = false;
let use_charts = false;
let qual_answers = null;
let valence_default_min, valence_default_max, dependent_measure;

function ensureHiddenOutput(name, id) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement("input");
    el.type = "hidden";
    el.name = name;
    el.id = id;
    const form =
      document.getElementById("jatos_form") || document.querySelector("form");
    if (form) {
      form.appendChild(el);
    } else if (document.body) {
      document.body.appendChild(el);
    }
  }
  return el;
}

// Utils: Base64 <-> bytes
function base64ToBytes(base64) {
  const binString = atob(base64);
  return Uint8Array.from(binString, (m) => m.codePointAt(0));
}
function bytesToBase64(bytes) {
  const binString = Array.from(bytes, (x) => String.fromCodePoint(x)).join("");
  return btoa(binString);
}
function bytesToJson(bytes) {
  return eval(new TextDecoder().decode(base64ToBytes(bytes)));
}

async function failJatosIfAvailable(message, details) {
  if (window.jatos && typeof window.failAndEndStudy === "function") {
    await window.failAndEndStudy(message, details);
    return true;
  }
  return false;
}

// Get embedded per-question data (base64 JSON in hidden inputs)
function get_data(selector) {
  const id = $(selector).attr("id") + "-values";
  const str = $("#" + id).val();
  return bytesToJson(str);
}

function get_example_data(id) {
  const parts = id.split("-");
  const example_num = Number(parts[parts.length - 1]) - 1;
  return example_data[example_num];
}

// Async loaders
async function loadVariables() {
  // In JATOS, prefer component properties over an external variables.json file
  if (typeof window !== "undefined" && window.jatos) {
    const jsonInput = jatos.componentJsonInput || {};
    const input = jatos.componentInput || {};
    variables = { ...jsonInput, ...input };
  } else {
    const dataURI = $("#variables").attr("src");
    if (dataURI && dataURI.startsWith("data:application/json;base64")) {
      variables = JSON.parse(atob(dataURI.split(",")[1]));
    } else if (dataURI) {
      try {
        variables = await d3.json(dataURI);
      } catch (e) {
        alert("Failed to load variables.json. Please reload the page.");
        throw e;
      }
    } else {
      variables = {};
    }
  }

  const chartType = variables.chart_type || "none";
  const validChartTypes = new Set(["area", "volume", "both", "none"]);
  if (window.jatos && !validChartTypes.has(chartType)) {
    const handled = await failJatosIfAvailable(
      "Invalid chart_type in component config.",
      { chartType },
    );
    if (handled) return;
    throw new Error("Invalid chart_type");
  }
  use_charts = chartType !== "none";
  if (use_charts) {
    chart_func = chart_funcs[chartType];
    if (!chart_func) {
      const handled = await failJatosIfAvailable(
        "chart_type not supported by chart_funcs.",
        { chartType },
      );
      if (handled) return;
      throw new Error("chart_type not supported");
    }
  }
  maximize = !!variables.maximize;

  // Compute derived strings once maximize is known
  valence_default_min = maximize ? "decrease" : "increase";
  valence_default_max = maximize ? "increase" : "decrease";
  dependent_measure = variables.dependent_measure || "best";

  // Record condition and maximize flag as hidden outputs
  try {
    const conditionValue = variables.chart_type || "";
    const maximizeValue = maximize ? "true" : "false";
    ensureHiddenOutput("condition_name", "condition_name").value =
      conditionValue;
    ensureHiddenOutput("maximize", "maximize").value = maximizeValue;
    ensureHiddenOutput("dependent_measure", "dependent_measure").value =
      variables.dependent_measure;
  } catch (_) {
    // Best-effort only; do not block task if DOM is not ready yet.
  }
}

async function loadQualAnswers() {
  // Keep button disabled until we finish
  $("#qualificationButton").prop("disabled", true);

  const qualURI = $("#qualification_answers").attr("src");
  if (qualURI && qualURI.startsWith("data:application/json;base64")) {
    qual_answers = JSON.parse(atob(qualURI.split(",")[1]));
  } else if (qualURI) {
    try {
      qual_answers = await d3.json(qualURI);
    } catch (e) {
      const handled = await failJatosIfAvailable(
        "Failed to load qualification answers.",
        { url: qualURI, error: String(e) },
      );
      if (handled) return;
      alert("Failed to load qualification answers. Please reload the page.");
      throw e;
    }
  } else {
    qual_answers = {};
  }
  if (window.jatos && Object.keys(qual_answers || {}).length === 0) {
    const handled = await failJatosIfAvailable(
      "Qualification answers were empty.",
      { url: qualURI },
    );
    if (handled) return;
    throw new Error("Qualification answers empty");
  }

  // Bind inputs and only enable button when all are filled
  bindQualificationInputs();
  updateQualificationButtonState();
}

// Build task description after variables are loaded
function buildTaskDescription() {
  let task_description = `<p>In this task we assess how to choose between different views.</p>
<p>Your task is to choose what you believe is the <strong>${dependent_measure}</strong> proposal.</p>`;

  if (dependent_measure === "best compromise") {
    task_description += `<p>By "best compromise", we mean the proposal that you think best balances the competing interests of the different groups. Proposals which seem unfair can still be considered the best compromise if they align better with your preferences. The concept of "best compromise" is subjective rather than objectively fair.</p>`;
  }

  task_description += `<p>Whether one option is a better than another is up to you.</p>
<p>In this version of the task, groups <strong>${maximize ? "prefer" : "dislike"}</strong> higher outcomes.
This means that higher outcomes are ${maximize ? "better" : "worse"}.</p>`;
  if (use_charts) {
    task_description += `<p>The charts shown might aid your reasoning about the proposals, but they do not contain an obvious answer like in the qualification task.
</p>`;
  }
  $("#task-description").each(function () {
    $(this).append($("<div>").html(task_description));
  });
}

function render_charts(parent) {
  // Accept selector string, DOM node, jQuery object, or nothing
  const $root = parent ? $(parent) : $(document);

  // AREA charts
  $root.find(".area-chart").each(function () {
    const width = 540;
    const height = 300;
    const chartId = $(this).attr("id"); // avoid shadowing "selector"
    const data = get_example_data(chartId);
    const color = get_color(data);
    const chart = make_area_chart(data, color, width, height);
    $(this).empty().append(chart.node());
  });

  // VOLUME charts
  $root.find(".volume-chart").each(function () {
    const width = 500;
    const height = 360;
    const scale = $(this).attr("data-scale");
    const chartId = $(this).attr("id");
    const data = get_example_data(chartId);
    const color = get_color(data);
    const chart = make_volume_chart(data, color, width, height, scale);
    $(this).empty().append(chart.node());
  });
}

// Main entry
async function main() {
  // Wire the click handler but keep the button disabled until ready
  $("#qualificationButton").on("click", submit_qual).prop("disabled", true);

  // Load variables and qual answers in parallel
  await Promise.all([loadVariables(), loadQualAnswers()]);

  buildTaskDescription();
  // Now it's safe to render UI that depends on maximize/use_charts

  render_charts();

  // Pre-fill and lock example answers
  select_qual_answers();

  // Note: The qualification button will be enabled by updateQualificationButtonState()
  // once all answers are filled. If you want to show the survey immediately after passing,
  // that logic remains in pass_qual().
}

function afterLoadData() {
  console.log("after load data");
  const container = $("#questions-container");
  if (container.length === 0) return;

  const generic_context = "will affect the outcome for each group";
  const generic_unit = "points";
  const scenarioMeta = [
    {
      title: "Wait Times",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of days a group member must <strong>wait for an appointment</strong>",
      unit: "days",
    },
    {
      title: "Life Expectancy",
      context:
        "will <strong>" +
        valence_default_max +
        "</strong> the average number of <strong>years a group member will live</strong>",
      unit: "years",
    },
    {
      title: "Medical Costs",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the <strong>average cost of a medical visit</strong> for each group",
      unit: "dollars",
    },
    {
      title: "Travel Times",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of minutes a group member must <strong>travel for an appointment</strong>",
      unit: "minutes",
    },
    {
      title: "Emergency Response Time",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average <strong>ambulance response time</strong>",
      unit: "minutes",
    },
    {
      title: "Hospital Readmission Rates",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the <strong>30-day readmission rate</strong> at the local hospital",
      unit: "percent",
    },
    {
      title: "Chronic Disease Prevalence",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the number of adults with a preventable <strong>chronic disease</strong> (like diabetes)",
      unit: "cases per 1,000 people",
    },
    {
      title: "Communicable Disease Spread",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>people infected per disease outbreak</strong>",
      unit: "people",
    },
    {
      title: "Mental Health Access",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>people per mental health professional</strong>",
      unit: "people",
    },
    {
      title: "Infant Mortality",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the number of <strong>infant deaths per 1,000 live births</strong>",
      unit: "deaths per 1,000 births",
    },
    {
      title: "Overdose Incidence",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the number of <strong>overdose events per 100,000 people</strong>",
      unit: "cases per 100,000 people",
    },
    {
      title: "Mental Health Waitlist Length",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>days to the first therapy appointment</strong>",
      unit: "days",
    },
    {
      title: "Missed Work",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>work days missed due to illness</strong>",
      unit: "days",
    },
    {
      title: "Specialist Referral Delay",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>days between referral and specialist visit</strong>",
      unit: "days",
    },
    {
      title: "Primary Care Access",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>days to a primary care appointment</strong>",
      unit: "days",
    },
    {
      title: "Medication Adherence",
      context:
        "will <strong>" +
        valence_default_max +
        "</strong> the share of patients who <strong>take prescribed medications as directed</strong>",
      unit: "percent",
    },
    {
      title: "Post-Surgery Recovery Time",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the average number of <strong>days needed for recovery after surgery</strong>",
      unit: "days",
    },
    {
      title: "Preventable Hospitalizations",
      context:
        "will <strong>" +
        valence_default_min +
        "</strong> the number of <strong>hospital admissions that could have been prevented</strong>",
      unit: "cases per 100,000 people",
    },
  ];

  const numScenarios = variables.num_scenarios || 1; // Default to 1 if not specified

  // Loop through all potential scenarios
  for (let i = 1; i <= numScenarios; i++) {
    // Check if the hidden input with the scenario data exists
    const scenarioInput = $(`#question-${i}-values`);
    if (scenarioInput.length > 0 && scenarioInput.val()) {
      const meta = scenarioMeta[i - 1] || {
        title: `Scenario ${i}`,
        context: generic_context,
        unit: generic_unit,
      };

      // Create the HTML structure for the question dynamically
      const questionHTML = `
        <div class="question" style="margin-top: 20px;">
          <div class="row question-choice-container col-12">
            <div class="row col-12 border pt-1 pb-1 mb-1 ml-2 border-warning" style="background-color:#eee">
              <span>Scenario: <span class="goal-sm">${meta.title}</span></span>
            </div>
          </div>
          <div class="row">
            <div id="question-${i}" class="question-data offset-1 col-11 pt-4">
            </div>
          </div>
        </div>
      `;
      const $questionElement = $(questionHTML);
      container.append($questionElement);

      // Call make_question on the newly created element
      make_question(
        $questionElement.find(".question-data"),
        meta.context,
        meta.unit,
      );
    }
  }
}

// QUALIFICATION: enable button only when answers are filled and JSON is loaded
function bindQualificationInputs() {
  if (!qual_answers) return;

  // For each expected answer key, ensure inputs are marked as qualification-input
  Object.keys(qual_answers).forEach((key) => {
    const $inputs = $(`input[name="${key}"]`);
    if ($inputs.length === 0) {
      return;
    }

    $inputs.addClass("qualification-input");

    const type = ($inputs.first().attr("type") || "").toLowerCase();
    if (type === "radio" || type === "checkbox") {
      // Using HTML constraint validation is fine; set required on the group
      // Setting required on any one of the radios in the group is enough
      $inputs.prop("required", true);
      // Listen to change on the group
      $inputs.on("change", updateQualificationButtonState);
    } else {
      // Text or number inputs
      $inputs
        .prop("required", true)
        .on("input", updateQualificationButtonState);
    }
  });

  // Also guard against any late-added inputs
  $(document).on(
    "change input",
    ".qualification-input",
    updateQualificationButtonState,
  );
}

function areAllQualificationAnswersFilled() {
  if (!qual_answers) return false;

  const qualificationMode = String(window.qualificationMode || "full");
  const keysToCheck = Object.keys(qual_answers).filter((key) => {
    if (qualificationMode === "area_only") {
      return (
        key === "q_question-stacked-1" ||
        key === "q_question-stacked-2" ||
        key === "q_question-stacked-3"
      );
    }
    return true;
  });

  for (const key of keysToCheck) {
    const $inputs = $(`input[name="${key}"]`);
    if ($inputs.length === 0) continue;

    const type = ($inputs.first().attr("type") || "").toLowerCase();
    if (type === "radio" || type === "checkbox") {
      if (!$inputs.is(":checked")) return false;
    } else {
      const val = $inputs.val();
      if (val == null || String(val).trim() === "") return false;
    }
  }
  return true;
}

function updateQualificationButtonState() {
  const ready = !!qual_answers && areAllQualificationAnswersFilled();
  $("#qualificationButton").prop("disabled", !ready);
}

// Submit qualification (guarded by qual_answers loaded + form completeness)
function submit_qual() {
  if (!qual_answers) {
    alert(
      "Still loading the qualification answers. Please try again in a moment.",
    );
    return;
  }
  if (!areAllQualificationAnswersFilled()) {
    alert("Please complete all qualification answers before submitting.");
    return;
  }

  let correct = 0;
  let total = 0;
  const qualificationMode = String(window.qualificationMode || "full");
  for (const [key, answer] of Object.entries(qual_answers)) {
    if (
      qualificationMode === "area_only" &&
      key !== "q_question-stacked-1" &&
      key !== "q_question-stacked-2" &&
      key !== "q_question-stacked-3"
    ) {
      continue;
    }
    const $inputs = $(`input[name="${key}"]`);
    if ($inputs.length === 0) {
      continue;
    }
    total++;

    let response;
    if (key == "q_question-3D-0") {
      response = $inputs.val();
    } else {
      response = $inputs.filter(":checked").val();
    }
    if (response == answer) {
      correct++;
    }
  }

  $("#qualificationButton").prop("disabled", true);
  $(".qualification-input").prop("disabled", true);

  if (total > 0 && correct === total) {
    pass_qual();
  } else {
    fail_qual();
  }
}

function fail_qual() {
  $("#qualificationButton").prop("disabled", true);
  $(".qualification-input").prop("disabled", true);
  $("#surveyButton").prop("disabled", false);
  $("#surveyInput").show();
  window.__qualificationComplete = true;
  window.onQualificationFailed(); // JATOS will handle assignment for FAIL
}

function pass_qual() {
  $("#qualificationButton").prop("disabled", true);
  $(".qualification-input").prop("disabled", true);
  $("#surveyButton").prop("disabled", false);
  $("#surveyInput").show();
  window.__qualificationComplete = true;
  window.onQualificationPassed(); // JATOS will claim from queue for PASS
}

function select_qual_answers() {
  // Disable and prefill example answers (unchanged)
  $("#examples input").prop("disabled", "disabled");
  $("#examples #q_question-stacked-1_option_two").prop("checked", "checked");
  $("#examples #q_question-stacked-2_option_two").prop("checked", "checked");
  $("#examples #q_question-stacked-3_option_two").prop("checked", "checked");
  $("#examples #q_question-stacked-4_option_two").prop("checked", "checked");
  $("#examples #q_question-stacked-5_option_one").prop("checked", "checked");
  $("#examples #q_question-stacked-6_option_three").prop("checked", "checked");
  $("#examples #q_question-3D-0_option_one").prop("value", "elephant");
  $("#examples #q_question-3D-1_option_two").prop("checked", "checked");
  $("#examples #q_question-3D-2_option_two").prop("checked", "checked");
  $("#examples #q_question-3D-3_option_two").prop("checked", "checked");
}

function area_chart(data, color, width, height) {
  return $("<div>")
    .attr("class", "pt-4 col-12")
    .append(
      $("<p>").attr("style", "text-align: center").text(`Stacked Bar Chart`),
    )
    .append(make_area_chart(data, color, width, height).node());
}

function volume_chart(data, color, width, height) {
  return $("<div>")
    .attr("class", "pt-4 col-12")
    .append($("<p>").attr("style", "text-align: center").text(`3D Bar Chart`))
    .append(make_volume_chart(data, color, width, height).node());
}

function both_charts(data, color, width, height) {
  const both = [
    area_chart(data, color, width, height),
    volume_chart(data, color, width, height),
  ];
  // Not that random, but random enough
  both.sort(() => 0.5 - Math.random());
  return $("<div>").attr("class", "row").append(both[0]).append(both[1]);
}

function get_color(data) {
  const scenarios = d3.union(data.map((d) => d.action));
  const color = d3
    .scaleOrdinal()
    .range(color_scheme)
    .domain(scenarios)
    .range(color_scheme);
  return color;
}

function make_question(element, context, unit) {
  var data;
  try {
    data = get_data(element);
  } catch (_) {
    return;
  }
  const selector = $(element).attr("id");

  const width = 900;
  const height = 500;

  const scenarios = d3.union(data.map((d) => d.action));
  const sizes = d3.rollup(
    data,
    (v) => d3.mean(v, (d) => d.credence),
    (d) => d.agent,
  );
  const outcomes = d3.index(
    data,
    (d) => d.action,
    (d) => d.agent,
  );
  const all_outcomes = d3.union(data.map((d) => d.utility));

  const color = d3
    .scaleOrdinal()
    .range(color_scheme)
    .domain(scenarios)
    .range(color_scheme);

  const description = make_question_description(
    sizes,
    outcomes,
    scenarios,
    color,
    selector,
    context,
    unit,
  );

  const q_input = make_question_input(scenarios, color, selector);

  const attn = make_question_attention_check(
    sizes,
    outcomes,
    all_outcomes,
    scenarios,
    color,
    selector,
  );

  let scenario_and_chart = $("<div>")
    .attr("class", "row")
    .append(
      $("<div>").attr("class", "col-md-12 col-lg-4 ").append(description),
    );
  const right_side_class = "col-md-12 col-lg-8";
  let question_class = right_side_class;
  let question_container = $("<div>");
  if (use_charts) {
    const chart = chart_func(data, color, width, height);
    scenario_and_chart.append(
      $("<div>").attr("class", right_side_class).append(chart),
    );
    question_class = "col col-10 offset-1";
    question_container = question_container.attr("class", "row");
  }

  const questions = $("<div>")
    .attr("class", question_class)
    .append(attn)
    .append($("<hr />"))
    .append(q_input);

  if (use_charts) {
    $("#" + selector)
      .append(scenario_and_chart)
      .append(question_container.append(questions));
  } else {
    $("#" + selector).append(scenario_and_chart.append(questions));
  }
}

function make_question_description(
  sizes,
  outcomes,
  scenarios,
  color,
  q_num,
  context,
  unit,
) {
  const groups = Array.from(sizes.keys());

  let group_description = $("<p>").text(
    "In this scenario, there are " + groups.length + " groups:",
  );
  let group_list = $("<ul>");
  for (let i = 0; i < groups.length; i++) {
    let group = groups[i];
    let text = "<span class=group-".concat(
      group,
      ">group ",
      group,
      "</span>",
      " with <strong>",
      sizes.get(group),
      "</strong> people in it",
    );
    if (i < groups.length - 2) {
      text += ", ";
    } else if (i < groups.length - 1) {
      text += ", and ";
    } else {
      text += ".";
    }
    group_list.append($("<li>").html(text));
  }

  let proposals_description = "There are ".concat(
    scenarios.size,
    " proposals, each of which ",
    context,
    " by:",
  );
  let proposals_list = $("<ul>");

  scenarios.forEach(function (proposal) {
    // Legend squares
    const dot = d3.create("svg").attr("width", 15).attr("height", 15);
    dot
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 15)
      .attr("height", 15)
      .style("fill", color(proposal));
    const square = $("<span>").append(dot.node());

    let scen_text = 'proposal <span class="font-weight-bold"'.concat(
      'style="color : ',
      color(proposal),
      '">',
      proposal,
      "</span> ",
      square.html(),
      ":<br>",
    );
    for (let i = 0; i < groups.length; i++) {
      let group = groups[i];
      let utility = outcomes.get(proposal).get(group).utility;
      let text = utility.toString() + " " + unit + " for group " + group;
      if (i < groups.length - 2) {
        text += ",<br>";
      } else if (i < groups.length - 1) {
        text += ",<br>and ";
      } else {
        text += ".";
      }
      scen_text += text;
    }

    proposals_list.append($("<li>").html(scen_text));
  });

  return $("<div>")
    .append(group_description)
    .append(group_list)
    .append($("<p>").html(proposals_description))
    .append(proposals_list);
}

function make_question_attention_check(
  sizes,
  outcomes,
  all_outcomes,
  scenarios,
  color,
  q_num,
) {
  // randomly choose size or outcome
  // randomly choose an agent
  // if outcome randomly choose an action

  let show_size = Math.random() < 0.5 ? true : false;
  const equal_groups = new Set(sizes.values()).size == 1;
  if (equal_groups) {
    show_size = false;
  }
  const chosen_group = shuffle(Array.from(sizes.keys()))[0];
  const chosen_proposal = shuffle(Array.from(outcomes.keys()))[0];

  let html = "What is the size of group " + chosen_group + "?";
  let answer = sizes.get(chosen_group);
  let options = Array.from(sizes.values());
  if (show_size === false) {
    let prop = '<span class="font-weight-bold" style="color : '.concat(
      color(chosen_proposal),
      ';">',
      chosen_proposal,
      "</span>",
    );
    html = "What is the outcome for group ".concat(
      chosen_group,
      " of proposal ",
      prop,
      "?",
    );
    answer = outcomes.get(chosen_proposal).get(chosen_group).utility;
    options = all_outcomes;
  }

  var outside_div = $("<div>")
    .attr("class", "row")
    .append(
      $("<div>")
        .attr("class", "offset-md-1 col col-md-5 col-sm-12 pt-4")
        .append($("<p>").html(html)),
    )
    .append(
      $("<input>")
        .attr("name", "q_" + q_num + "_attn_answer")
        .attr("id", "q_" + q_num + "_attn" + "_answer")
        .attr("value", answer)
        .attr("hidden", true),
    );

  var answer_div = $("<div>").attr("class", "mt-2 col-md-5 col-sm-12");

  options.forEach(function (option) {
    answer_div.append(
      $("<div>")
        .attr("class", "col-12")
        .append(
          $("<input>")
            .attr("type", "radio")
            .attr("name", "q_" + q_num + "_attn")
            .attr("id", "q_" + q_num + "_attn" + "_opt_" + option)
            .attr("value", option)
            .prop("required", true),
        )
        .append($("<label>").text(option)),
    );
  });

  outside_div.append(answer_div);

  return outside_div;
}

function make_question_input(proposals, color, q_num) {
  var outside_div = $("<div>")
    .attr("class", "row")
    .append(
      $("<div>")
        .attr("class", "offset-md-1 col col-md-5 col-sm-12 pt-4")
        .append(
          $("<p>").text(
            `Which proposal is the ${dependent_measure} in this situation?`,
          ),
        ),
    );

  var prop_div = $("<div>").attr("class", "mt-2 col-md-5 col-sm-12");

  proposals.forEach(function (proposal) {
    const span =
      'Proposal <span class="font-weight-bold" style="color : '.concat(
        color(proposal),
        ';">',
        proposal,
        "</span>",
      );

    prop_div.append(
      $("<div>")
        .attr("class", "col-12")
        .append(
          $("<input>")
            .attr("type", "radio")
            .attr("name", "q_" + q_num)
            .attr("id", "q_" + q_num + "_proposal_" + proposal)
            .attr("value", proposal)
            .prop("required", true),
        )
        .append($("<label>").html(span)),
    );
  });

  outside_div.append(prop_div);

  return outside_div;
}

// from : https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-arrays
function shuffle(unshuffled) {
  let shuffled = unshuffled
    .map((value) => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value);
  return shuffled;
}
