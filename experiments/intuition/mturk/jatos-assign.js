let C = {};
// Per-condition keys in Batch Session (so area/none do not conflict)
let QUEUE_KEYS = { queue: "queue", inProgress: "inProgress", done: "done" };
let PATHS = { queue: "/queue", inProgress: "/inProgress", done: "/done" };
const REQUEUE_LOCK_KEY = "__requeue_lock__";
let SUBMIT_LOCK = false;
let QUAL_COMPLETE = false;
let TASK_READY = false;
let USE_BATCH_SET = false;

window.__qualificationComplete = false;
window.__taskReady = false;
let SUBMIT_ERROR = null;

async function bsCall(label, fn) {
  try {
    const result = await fn();
    console.debug("batch_call_ok:", label);
    return result;
  } catch (e) {
    console.error("batch_call_failed:", label, e);
    if (e && e.stack) console.error("batch_call_failed_stack:", e.stack);
    throw e;
  }
}

function bsGetAll() {
  try {
    const data = jatos.batchSession.getAll() || {};
    console.debug("batch_get_all_ok");
    return data;
  } catch (e) {
    console.error("batch_get_all_failed", e);
    if (e && e.stack) console.error("batch_get_all_failed_stack:", e.stack);
    throw e;
  }
}

async function verifyBatchSession() {
  console.debug(
    "batchSession object keys:",
    Object.keys(jatos.batchSession || {}),
  );
  const probeKey = "__batch_probe__";
  console.debug("batch probe attempt 1/1");
  await bsCall("probe_add", () => jatos.batchSession.add(`/${probeKey}`, 1));
  await bsCall("probe_remove", () => jatos.batchSession.remove(`/${probeKey}`));
  console.debug("batch probe ok");
  USE_BATCH_SET = false;
  console.debug("batch set disabled; using add/remove only");
}

async function failAndEndStudy(message, details) {
  const text = details ? `${message} (${JSON.stringify(details)})` : message;
  console.error("fatal_error:", text);
  const payload = { event: "fatal_error", message, details };
  try {
    await jatos.submitResultData(payload);
  } catch (_) {}
  if (C && C.debugNoEndStudy) {
    console.warn("debugNoEndStudy enabled; not ending study.");
    return;
  }
  try {
    jatos.endStudy();
  } catch (_) {}
}

window.failAndEndStudy = failAndEndStudy;

function showSubmitError(message) {
  SUBMIT_ERROR = message;
  const existing = document.getElementById("submitError");
  if (existing) {
    existing.textContent = message;
    existing.style.display = "block";
    return;
  }
  const submitBtn = document.getElementById("submitButton");
  if (!submitBtn || !submitBtn.parentNode) return;
  const el = document.createElement("div");
  el.id = "submitError";
  el.style.marginTop = "0.5rem";
  el.style.fontSize = "0.9rem";
  el.style.color = "#a94442";
  el.textContent = message;
  submitBtn.parentNode.appendChild(el);
}

async function endIfNoDataLeft(items) {
  await ensureAllConditionQueues(items);
  await ensureBatchInitialized(items);
  await requeueStale();
  const bs = bsGetAll();
  const qNode = bs[QUEUE_KEYS.queue];

  const queueKeys = Object.keys(bs).filter((k) => k.startsWith("queue_"));
  if (queueKeys.length < 2) {
    // Do not end early if the other condition has not been initialized yet.
    return false;
  }

  if (!qNode || qNode.length === 0) {
    // Ensure all condition queues are empty and have no in-progress items.
    for (const key of queueKeys) {
      const cond = key.replace(/^queue_/, "");
      const q = bs[key];
      if (!q || q.length > 0) return false;
    }

    try {
      await jatos.submitResultData({ event: "no_items_left_prequal" });
    } catch (_) {}
    try {
      jatos.endStudy();
    } catch (_) {}
    return true;
  }
  return false;
}

function cfg() {
  const jsonInput = jatos.componentJsonInput || {};
  const input = jatos.componentInput || {};
  return { ...jsonInput, ...input };
}

function applyConditionPrefix(prefix) {
  if (!prefix) {
    QUEUE_KEYS = { queue: "queue", inProgress: "inProgress", done: "done" };
  } else {
    const p = String(prefix);
    QUEUE_KEYS = {
      queue: `queue_${p}`,
      inProgress: `inProgress_${p}`,
      done: `done_${p}`,
    };
  }
  PATHS = {
    queue: `/${QUEUE_KEYS.queue}`,
    inProgress: `/${QUEUE_KEYS.inProgress}`,
    done: `/${QUEUE_KEYS.done}`,
  };
}

async function loadCsv(url) {
  try {
    return await d3.csv(url);
  } catch (e) {
    await failAndEndStudy("Failed to load CSV data.", {
      url,
      error: String(e),
    });
    throw e;
  }
}

function buildQueue(numRows, m) {
  const q = [];
  for (let i = 0; i < numRows; i++) {
    for (let k = 0; k < m; k++) {
      q.push(i);
    }
  }
  return q;
}

async function ensureBatchInitialized(items) {
  const bs = bsGetAll();
  const qNode = bs[QUEUE_KEYS.queue];
  const ipNode = bs[QUEUE_KEYS.inProgress];
  const dNode = bs[QUEUE_KEYS.done];
  if (qNode && ipNode && dNode) return;
  const q = buildQueue(items.length, C.annotationsPerItem || 3);
  if (!qNode)
    await bsCall(`add ${PATHS.queue}`, () =>
      jatos.batchSession.add(PATHS.queue, q),
    );
  if (!ipNode)
    await bsCall(`add ${PATHS.inProgress}`, () =>
      jatos.batchSession.add(PATHS.inProgress, {}),
    );
  if (!dNode)
    await bsCall(`add ${PATHS.done}`, () =>
      jatos.batchSession.add(PATHS.done, {}),
    );
}

async function ensureAllConditionQueues(items) {
  let rawConditions = Array.isArray(C?.conditions) ? C.conditions : [];
  if (rawConditions.length === 0 && Array.isArray(window?.JATOS_CONDITIONS)) {
    rawConditions = window.JATOS_CONDITIONS;
  }
  if (rawConditions.length === 0) return;
  const conditions = rawConditions
    .map((cond) => ({
      chart_type: String(cond?.chart_type || ""),
      dependent_measure: String(cond?.dependent_measure || ""),
    }))
    .filter((cond) => cond.chart_type !== "" && cond.dependent_measure !== "");

  const bs = bsGetAll();
  const q = buildQueue(items.length, C.annotationsPerItem || 3);

  for (const cond of conditions) {
    const key = `${cond.chart_type}-${cond.dependent_measure}`;
    const qKey = `queue_${key}`;
    const ipKey = `inProgress_${key}`;
    const dKey = `done_${key}`;
    if (!(qKey in bs))
      await bsCall(`add /${qKey}`, () => jatos.batchSession.add(`/${qKey}`, q));
    if (!(ipKey in bs))
      await bsCall(`add /${ipKey}`, () =>
        jatos.batchSession.add(`/${ipKey}`, {}),
      );
    if (!(dKey in bs))
      await bsCall(`add /${dKey}`, () =>
        jatos.batchSession.add(`/${dKey}`, {}),
      );
  }
}

async function requeueStale() {
  const ttlMs = (C.claimStaleAfterSec ?? 900) * 1000;
  const now = Date.now();
  const bs = bsGetAll();
  const ip = bs[QUEUE_KEYS.inProgress] || {};
  for (const [wk, obj] of Object.entries(ip)) {
    if (!obj || obj.itemId === undefined) continue;
    const ts = obj.ts ?? 0;
    if (now - ts < ttlMs) continue;

    const itemId = obj.itemId;

    // Acquire a per-worker requeue lock to avoid double requeue in races
    let haveLock = false;
    try {
      await bsCall(`add ${PATHS.inProgress}/${wk}/${REQUEUE_LOCK_KEY}`, () =>
        jatos.batchSession.add(
          `${PATHS.inProgress}/${wk}/${REQUEUE_LOCK_KEY}`,
          1,
        ),
      );
      haveLock = true;
    } catch (_) {
      // Someone else requeued or is requeuing this worker already
      haveLock = false;
    }
    if (!haveLock) {
      // Best-effort cleanup if lock exists but entry still lingers
      try {
        await bsCall(`remove ${PATHS.inProgress}/${wk}`, () =>
          jatos.batchSession.remove(`${PATHS.inProgress}/${wk}`),
        );
      } catch (_) {}
      continue;
    }

    // Append back exactly one copy – never remove from queue here
    try {
      await bsCall(`add ${PATHS.queue}/-`, () =>
        jatos.batchSession.add(`${PATHS.queue}/-`, itemId),
      );
    } catch (e) {
      // If add failed due to a transient race, we still proceed to cleanup.
      // The worst that can happen is the item wasn't added this time; a later sweep can recover.
      console.warn("requeueStale: add queue failed (continuing to cleanup)", e);
    }

    // Clean up inProgress slot (removes ts and lock too)
    try {
      await bsCall(`remove ${PATHS.inProgress}/${wk}`, () =>
        jatos.batchSession.remove(`${PATHS.inProgress}/${wk}`),
      );
    } catch (e) {
      console.warn("requeueStale: cleanup failed", e);
    }
  }
}

async function claimNextRow() {
  const wk = String(jatos.workerId);

  // Ensure a container for this worker exists
  try {
    await bsCall(`add ${PATHS.inProgress}/${wk}`, () =>
      jatos.batchSession.add(`${PATHS.inProgress}/${wk}`, {}),
    );
  } catch (_) {}

  // If we already have an assignment, continue with it (prevents queue churn on reload)
  {
    const bs0 = bsGetAll();
    const ip0 = bs0[QUEUE_KEYS.inProgress] || {};
    const existing = ip0[wk]?.itemId;
    if (existing !== undefined) return existing;
  }

  while (true) {
    try {
      // Atomically move queue head into our inProgress slot
      await bsCall(
        `move ${PATHS.queue}/0 -> ${PATHS.inProgress}/${wk}/itemId`,
        () =>
          jatos.batchSession.move(
            `${PATHS.queue}/0`,
            `${PATHS.inProgress}/${wk}/itemId`,
          ),
      );
    } catch (e) {
      console.error("claimNextRow: move failed", e);

      // Only treat as empty if queue is actually empty
      const bs = bsGetAll();
      const qNode = bs[QUEUE_KEYS.queue];
      if (!qNode || qNode.length === 0) {
        return null;
      }

      // If another process concurrently assigned us (e.g., reload race), use that
      const ipRetry = bs[QUEUE_KEYS.inProgress] || {};
      const maybe = ipRetry[wk]?.itemId;
      if (maybe !== undefined) return maybe;

      // Otherwise, ensure our container exists (race) and retry
      try {
        await bsCall(`add ${PATHS.inProgress}/${wk}`, () =>
          jatos.batchSession.add(`${PATHS.inProgress}/${wk}`, {}),
        );
      } catch (_) {}
      await new Promise((r) => setTimeout(r, 50));
      continue;
    }

    // We successfully moved an item; stamp timestamp (best effort)
    try {
      await bsCall(`add ${PATHS.inProgress}/${wk}/ts`, () =>
        jatos.batchSession.add(`${PATHS.inProgress}/${wk}/ts`, Date.now()),
      );
    } catch (_) {
      try {
        await bsCall(`set ${PATHS.inProgress}/${wk}/ts`, () =>
          jatos.batchSession.set(`${PATHS.inProgress}/${wk}/ts`, Date.now()),
        );
      } catch (_) {}
    }

    // Read back the claimed item
    let bs = bsGetAll();
    const ipNode = bs[QUEUE_KEYS.inProgress] || {};
    const itemId = ipNode[wk]?.itemId;
    if (itemId === undefined) {
      // Rare race: if someone cleared it immediately, retry
      await new Promise((r) => setTimeout(r, 30));
      continue;
    }

    // Avoid re-assigning the same item to the same worker
    const dNode = bs[QUEUE_KEYS.done] || {};
    const alreadyDone = (dNode[itemId] || []).includes(wk);
    if (!alreadyDone) return itemId;

    // Put it back and try again (append tail via add, then clean)
    try {
      await bsCall(`add ${PATHS.queue}/-`, () =>
        jatos.batchSession.add(`${PATHS.queue}/-`, itemId),
      );
    } catch (e) {
      console.warn("claimNextRow: requeue after alreadyDone add failed", e);
    }
    try {
      await bsCall(`remove ${PATHS.inProgress}/${wk}`, () =>
        jatos.batchSession.remove(`${PATHS.inProgress}/${wk}`),
      );
      await bsCall(`add ${PATHS.inProgress}/${wk}`, () =>
        jatos.batchSession.add(`${PATHS.inProgress}/${wk}`, {}),
      );
    } catch (e) {
      console.warn("claimNextRow: requeue after alreadyDone cleanup failed", e);
    }
  }
}

function fillHiddenInputsFromRow(row) {
  const ensureHiddenQuestionInput = (questionIndex) => {
    const inputId = `question-${questionIndex}-values`;
    let el = document.getElementById(inputId);
    if (!el) {
      el = document.createElement("input");
      el.type = "hidden";
      el.className = "question-json";
      el.id = inputId;
      el.name = inputId;
      const form =
        document.getElementById("jatos_form") || document.querySelector("form");
      if (form) {
        form.appendChild(el);
      } else if (document.body) {
        document.body.appendChild(el);
      }
    }
    return el;
  };
  const setVal = (i, key) => {
    const el = ensureHiddenQuestionInput(i);
    if (row && Object.prototype.hasOwnProperty.call(row, key) && row[key]) {
      el.value = row[key];
    } else {
      el.value = "";
    }
  };
  const configuredNumScenarios = Number(C?.num_scenarios || 0);
  let inferredNumScenarios = 0;
  if (row && typeof row === "object") {
    for (const key of Object.keys(row)) {
      const match = key.match(/^scenario_(\d+)_json$/);
      if (!match) continue;
      const idx = Number(match[1]);
      if (idx > inferredNumScenarios) inferredNumScenarios = idx;
    }
  }
  const numScenarios = Math.max(
    configuredNumScenarios,
    inferredNumScenarios,
    9,
  );
  for (let i = 1; i <= numScenarios; i++) {
    setVal(i, `scenario_${i}_json`);
  }
}

function collectAnswers() {
  const answers = {};
  const radioHandled = new Set();

  document.querySelectorAll("input[name]").forEach((inp) => {
    const { name, type, value, checked } = inp;

    if (type === "radio") {
      if (radioHandled.has(name)) return;
      radioHandled.add(name);
      const sel = document.querySelector(
        `input[type="radio"][name="${CSS.escape(name)}"]:checked`,
      );
      answers[name] = sel ? sel.value : null;
    } else if (type === "checkbox") {
      if (!answers[name]) answers[name] = [];
      if (checked) answers[name].push(value || true);
      // ensure key exists even if none are checked
      if (!(name in answers)) answers[name] = [];
    } else {
      answers[name] = value;
    }
  });

  return answers;
}

function evaluateAttentionChecks(answers) {
  const hidden = document.querySelectorAll('input[name$="_attn_answer"]');
  if (!hidden || hidden.length === 0) return true;

  let allPass = true;
  hidden.forEach((el) => {
    const expected = String(el.value ?? "").trim();
    const baseName = String(el.name || "").replace(/_answer$/, "");
    const actual = String(answers[baseName] ?? "").trim();
    if (expected === "" || baseName === "") return;
    if (expected !== actual) allPass = false;
  });

  return allPass;
}

async function submitAndFinish(rowIndex, row, answers, opts) {
  const wk = String(jatos.workerId);
  const countForQuota = !!opts?.countForQuota;
  const attentionPass = evaluateAttentionChecks(answers);
  const effectiveCountForQuota = countForQuota && attentionPass;

  if (effectiveCountForQuota) {
    const path = `${PATHS.done}/${rowIndex}`;
    const bs = bsGetAll();
    const dNode = bs[QUEUE_KEYS.done] || {};
    if (!(String(rowIndex) in dNode))
      await bsCall(`add ${path}`, () => jatos.batchSession.add(path, []));
    const bs2 = bsGetAll();
    const dNode2 = bs2[QUEUE_KEYS.done] || {};
    const workers = dNode2[rowIndex] || [];
    if (!workers.includes(wk))
      await bsCall(`add ${path}/-`, () =>
        jatos.batchSession.add(`${path}/-`, wk),
      );
    await bsCall(`remove ${PATHS.inProgress}/${wk}`, () =>
      jatos.batchSession.remove(`${PATHS.inProgress}/${wk}`),
    );
  } else if (countForQuota) {
    // Not counted (e.g., attention failed) -> free the slot and requeue the item
    try {
      await bsCall(`add ${PATHS.queue}/-`, () =>
        jatos.batchSession.add(`${PATHS.queue}/-`, rowIndex),
      );
    } catch (_) {}
    try {
      await bsCall(`remove ${PATHS.inProgress}/${wk}`, () =>
        jatos.batchSession.remove(`${PATHS.inProgress}/${wk}`),
      );
    } catch (_) {}
  }

  const tmEl = document.getElementById("tm");
  if (tmEl && (!tmEl.value || tmEl.value === "")) {
    tmEl.value = TimeMe.getTimeOnCurrentPageInSeconds();
  }

  const result = {
    assignedCondition: `${C.dependent_measure || ""}-${C.chart_type || ""}`,
    dependent_measure: C.dependent_measure || "",
    chart_type: C.chart_type || "",
    qualificationModeUsed: String(window.qualificationMode || "full"),
    rowIndex,
    qualificationPassed: !!opts?.qualificationPassed,
    attentionPassed: attentionPass,
    row,
    answers,
    prolific: jatos.urlQueryParameters,
    meta: {
      workerId: jatos.workerId,
      batchId: jatos.batchId,
      studyResultId: jatos.studyResultId,
      when: new Date().toISOString(),
    },
  };
  try {
    await jatos.submitResultData(result);
  } catch (e) {
    showSubmitError("Submission failed. Please wait a moment and try again.");
    SUBMIT_LOCK = false;
    const submitBtn = document.getElementById("submitButton");
    if (submitBtn) submitBtn.disabled = false;
    throw e;
  }

  try {
    if (C.prolificCompletionCode) {
      const url = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(C.prolificCompletionCode)}`;
      jatos.endStudyAndRedirect(url);
    } else {
      jatos.endStudy();
    }
  } catch (e) {
    showSubmitError(
      "Submission saved but we could not finish the study. Please notify the researcher.",
    );
    throw e;
  }
}

function attachSubmitHandler(rowIndex, row, opts) {
  const form =
    document.getElementById("jatos_form") || document.querySelector("form");
  if (!form) return;

  const submitBtn = document.getElementById("submitButton");

  const handler = async (e) => {
    e.preventDefault();

    // Block any re-entrant submits (double clicks / Enter)
    if (SUBMIT_LOCK) return;

    // Block submission unless all required inputs are satisfied
    if (!form.checkValidity()) {
      // Triggers native “please fill out this field” messages and focuses the first invalid control
      form.reportValidity();
      return;
    }

    // Lock and disable submit immediately to prevent double-clicks
    SUBMIT_LOCK = true;
    if (submitBtn) submitBtn.disabled = true;

    const answers = collectAnswers();
    await submitAndFinish(rowIndex, row, answers, opts);
  };
  form.addEventListener("submit", handler, { once: true });
}

async function waitForVariablesReady() {
  // Wait until variables/maximize are set by main.js (in case variables.json loads async)
  let tries = 0;
  while (typeof window.maximize === "undefined" && tries < 40) {
    await new Promise((r) => setTimeout(r, 50));
    tries++;
  }
}
jatos.onLoad(async () => {
  console.log(
    "Component received jatos.componentJsonInput:",
    jatos.componentJsonInput,
  );
  console.log("Component received jatos.componentInput:", jatos.componentInput);
  console.log(
    "Component received jatos.urlQueryParameters:",
    jatos.urlQueryParameters,
  );
  try {
    console.debug("jatos onLoad start");
    C = cfg();
    console.debug("component config:", C);
    console.debug("jatos ids:", {
      batchId: jatos.batchId,
      studyId: jatos.studyId,
      studyResultId: jatos.studyResultId,
      workerId: jatos.workerId,
      componentId: jatos.componentId,
      componentPos: jatos.componentPos,
    });
    console.debug("window context:", {
      href: window.location.href,
      inIframe: window.self !== window.top,
    });
    console.debug("batchSession available:", !!jatos.batchSession);
    const chartType = C.chart_type || "none";
    const dependentMeasure = C.dependent_measure || "best";
    const qualificationMode =
      C.qualification_mode || C.qualificationMode || "full";
    window.qualificationMode = String(qualificationMode);
    console.debug("qualification mode:", window.qualificationMode);
    const condPrefix = `${chartType}-${dependentMeasure}`;
    applyConditionPrefix(condPrefix);
    console.debug("condition prefix:", condPrefix, "paths:", PATHS);
    await verifyBatchSession();
    console.debug("batch session verified");

    const csvUrl = C.dataCsvUrl || "data.csv";
    console.debug("loading csv:", csvUrl);
    const items = await loadCsv(csvUrl);
    console.debug("csv rows:", items.length);
    if (!items || items.length === 0) {
      await failAndEndStudy("CSV loaded but contained no items.", {
        url: csvUrl,
      });
      return;
    }
    if (await endIfNoDataLeft(items)) return;
    try {
      const bsKeys = Object.keys(bsGetAll());
      console.debug("batch session keys after init:", bsKeys);
    } catch (_) {}

    // Make qualification decision handlers visible to main.js
    window.onQualificationPassed = async function () {
      console.log("qual passed");
      QUAL_COMPLETE = true;
      window.__qualificationComplete = true;
      try {
        await ensureBatchInitialized(items);
        await requeueStale();
        const rowIndex = await claimNextRow();
        if (rowIndex === null) {
          await jatos.submitResultData({
            event: "no_items_left_after_qual_pass",
          });
          return jatos.endStudy();
        }
        const row = items[rowIndex];
        fillHiddenInputsFromRow(row);
        await waitForVariablesReady();
        window.renderTaskQuestions();
        TASK_READY = true;
        window.__taskReady = true;
        attachSubmitHandler(rowIndex, row, {
          qualificationPassed: true,
          countForQuota: true,
        });
      } catch (e) {
        await failAndEndStudy("Batch session error after qualification.", {
          error: String(e),
        });
      }
    };

    window.onQualificationFailed = async function () {
      QUAL_COMPLETE = true;
      window.__qualificationComplete = true;
      const rowIndex = Math.floor(Math.random() * items.length);
      const row = items[rowIndex];
      fillHiddenInputsFromRow(row);
      await waitForVariablesReady();
      window.renderTaskQuestions();
      TASK_READY = true;
      window.__taskReady = true;
      attachSubmitHandler(rowIndex, row, {
        qualificationPassed: false,
        countForQuota: false,
      });
    };

    if (window.runTaskMain) await window.runTaskMain();
    else main(); // fallback if you forgot to expose runTaskMain
  } catch (e) {
    await failAndEndStudy("JATOS initialization failed.", { error: String(e) });
  }
});

(function () {
  const form = document.getElementById("jatos_form");
  const submitBtn = document.getElementById("submitButton");
  if (!form || !submitBtn) return;

  let helpEl = document.getElementById("submitHelp");
  if (!helpEl) {
    helpEl = document.createElement("div");
    helpEl.id = "submitHelp";
    helpEl.style.marginTop = "0.5rem";
    helpEl.style.fontSize = "0.9rem";
    helpEl.style.color = "#8a6d3b";
    helpEl.style.display = "none";
    helpEl.textContent =
      "Please answer all required questions to enable submit.";
    submitBtn.parentNode.appendChild(helpEl);
  }

  function updateSubmitState() {
    const qualOk = QUAL_COMPLETE || window.__qualificationComplete;
    const taskOk = TASK_READY || window.__taskReady;
    const formOk = form.checkValidity();
    submitBtn.disabled = SUBMIT_LOCK || !formOk || !qualOk || !taskOk;
    helpEl.style.display = !formOk && qualOk && taskOk ? "block" : "none";
    const errorEl = document.getElementById("submitError");
    if (errorEl && SUBMIT_ERROR) {
      errorEl.style.display = "block";
    }
  }

  // Re-evaluate when inputs change
  document.addEventListener("input", updateSubmitState, true);
  document.addEventListener("change", updateSubmitState, true);

  // Re-evaluate when dynamic content is inserted (your questions render after load)
  new MutationObserver(updateSubmitState).observe(form, {
    childList: true,
    subtree: true,
  });

  // Initial state
  updateSubmitState();
})();
