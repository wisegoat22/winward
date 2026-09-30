"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  let examples = [];
  let scenario = null;
  let ready = false;
  let running = false;
  let hasResult = false;
  const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
  const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null) node.textContent = String(content);
    return node;
  }

  function humanize(value) {
    return String(value).replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function displayNumber(value, suffix = "") {
    return finite(value) ? `${number.format(value)}${suffix}` : "Not reported";
  }

  function percentage(value) {
    return finite(value) ? `${number.format(value * 100)}%` : "Not reported";
  }

  function hasBit(mask, index) {
    return Math.floor(Number(mask || 0) / (2 ** index)) % 2 === 1;
  }

  function labelsFor(mask) {
    return (scenario?.facts || []).filter((_, index) => hasBit(mask, index)).map(humanize);
  }

  function permitted(action) {
    if (!scenario || action.allowed === false) return false;
    const requires = Number(action.requires || 0);
    const forbids = Number(action.forbids || 0);
    return (scenario.state & requires) === requires && (scenario.state & forbids) === 0;
  }

  function showError(message) {
    $("policy-error").textContent = message;
    $("policy-error").hidden = !message;
  }

  async function request(path, options) {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: { "Content-Type": "application/json", ...options?.headers },
    });
    let payload;
    try { payload = await response.json(); } catch (_) {
      throw new Error(`The local server returned an unreadable response (${response.status}).`);
    }
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail
        : typeof payload.error === "string" ? payload.error
          : `The local request failed (${response.status}).`;
      throw new Error(detail);
    }
    return payload;
  }

  function stat(label, value) {
    const node = element("div", "policy-stat");
    node.append(element("strong", "", value), element("span", "", label));
    $("model-metrics").append(node);
  }

  function flattenScalars(value, prefix = "", result = []) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return result;
    for (const [key, item] of Object.entries(value)) {
      const path = prefix ? `${prefix} · ${key}` : key;
      if (item && typeof item === "object" && !Array.isArray(item)) flattenScalars(item, path, result);
      else if (typeof item === "number" || typeof item === "string" || typeof item === "boolean") result.push([path, item]);
    }
    return result;
  }

  function updateStatus(data) {
    ready = data.status === "ready";
    $("policy-connection").className = `connection ${ready ? "ready" : "unavailable"}`;
    $("policy-connection-label").textContent = ready ? "Local policy ready" : "Awaiting trained checkpoint";
    $("training-status").textContent = ready ? "Checkpoint ready" : "Not trained yet";
    $("training-status").className = `review-badge${ready ? "" : " review"}`;
    $("policy-model-name").textContent = data.model_name || "Model name not reported";
    $("training-json").textContent = JSON.stringify(data, null, 2);
    $("model-metrics").replaceChildren();
    stat("Trainable parameters", finite(data.parameter_count) ? integer.format(data.parameter_count) : "Not reported");
    const trainingFields = flattenScalars(data.training || {});
    const exampleCount = trainingFields.find(([key, value]) => finite(value) && /(?:train(?:ing)?[_ ]?(?:examples|samples|rows|cases)|n_train|train_size)/i.test(key))
      || trainingFields.find(([key, value]) => finite(value) && /examples|samples/i.test(key));
    if (exampleCount) stat(humanize(exampleCount[0]), integer.format(exampleCount[1]));
    const provenance = data.evaluation?.dataset_provenance || {};
    const freshInstances = provenance.source === "freshly generated instances";
    const testLabel = freshInstances ? "Fresh simulated test" : "Simulated test";
    const provenanceText = typeof data.evaluation_provenance === "string"
      ? data.evaluation_provenance
      : [freshInstances ? "Freshly generated test instances." : "", provenance.structure_family_status || ""].filter(Boolean).join(" ");
    $("policy-provenance").textContent = provenanceText;
    $("policy-provenance").hidden = !provenanceText;
    const metrics = flattenScalars(data.metrics || {});
    const headlineMetrics = metrics.filter(([key, value]) => finite(value) && /accuracy|success|agreement|latency|win_rate/i.test(key)).slice(0, 6);
    for (const [key, value] of headlineMetrics) {
      const isRate = /accuracy|success|agreement|rate/i.test(key) && !/count|total|steps|time|latency/i.test(key) && value >= 0 && value <= 1;
      const suffix = /(?:^|_)ms(?:$|_)/i.test(key) ? " ms" : "";
      const metricLabels = {
        test_accuracy: `${testLabel} · next-action agreement`,
        nontrivial_test_accuracy: `${testLabel} · unfinished-goal agreement`,
      };
      stat(metricLabels[key] || humanize(key), isRate ? percentage(value) : displayNumber(value, suffix));
    }
    if (!ready && headlineMetrics.length === 0) stat("Training status", "Awaiting checkpoint");
    const rollout = data.evaluation?.closed_loop;
    const neuralRollout = rollout?.methods?.neural;
    $("policy-audit-note").hidden = true;
    if (finite(neuralRollout?.success_rate)) {
      stat(`${testLabel} · goals reached within ${finite(rollout.horizon) ? rollout.horizon : "the allowed"} actions`, percentage(neuralRollout.success_rate));
      const counts = finite(neuralRollout.successes) && finite(neuralRollout.cases)
        ? `${integer.format(neuralRollout.successes)} of ${integer.format(neuralRollout.cases)} simulated tasks`
        : "the reported simulated tasks";
      $("policy-audit-note").textContent = `When making successive decisions, the model completed ${counts}. These started unfinished and were solvable by the search reference. This measures performance inside the simulator; reliability with real tools remains untested.`;
      $("policy-audit-note").hidden = false;
    }
    const changedGoal = data.evaluation?.counterfactuals?.changed_goal;
    $("policy-probe-note").hidden = true;
    if (finite(changedGoal?.neural_accuracy)) {
      const pairSummary = finite(changedGoal.neural_both_members_correct) && finite(changedGoal.pairs)
        ? ` Both decisions were correct in ${integer.format(changedGoal.neural_both_members_correct)} of ${integer.format(changedGoal.pairs)} paired cases.` : "";
      const weakness = changedGoal.neural_accuracy < 0.8 ? " Changing goals remains a weakness in this check." : "";
      $("policy-probe-note").textContent = `Changed-goal probe: ${percentage(changedGoal.neural_accuracy)} next-action agreement.${pairSummary}${weakness} These are reusable diagnostics, separate from the ${freshInstances ? "fresh simulated test" : "main simulated test"}.`;
      $("policy-probe-note").hidden = false;
    }
    $("policy-limitation").textContent = typeof data.limitation === "string" && data.limitation
      ? data.limitation
      : "Results apply to synthetic simulations. Real repositories and arbitrary natural-language goals require further work and evaluation.";
    refreshRunButton();
  }

  function refreshRunButton() {
    $("policy-run").disabled = running || !ready || !scenario;
    $("policy-run").textContent = running ? "Evaluating…" : ready ? "Choose the next action →" : "Waiting for trained model";
  }

  function markChanged() {
    if (hasResult) $("stale-results").hidden = false;
  }

  function renderFacts() {
    $("scenario-facts").replaceChildren();
    (scenario.facts || []).forEach((fact, index) => {
      const row = element("tr");
      row.append(element("td", "", humanize(fact)));
      for (const [property, text] of [["state", "True now"], ["goal", "Needed for win"]]) {
        const cell = element("td");
        const input = element("input");
        input.type = "checkbox";
        input.checked = hasBit(scenario[property], index);
        input.setAttribute("aria-label", `${humanize(fact)}: ${text}`);
        input.addEventListener("change", () => {
          const bit = 2 ** index;
          scenario[property] += input.checked ? bit : -bit;
          markChanged();
          renderActions();
        });
        cell.append(input);
        row.append(cell);
      }
      $("scenario-facts").append(row);
    });
  }

  function addActionDetail(container, label, values) {
    if (!values.length) return;
    const paragraph = element("p", "policy-action-detail");
    paragraph.append(element("strong", "", `${label}: `), document.createTextNode(values.join("; ")));
    container.append(paragraph);
  }

  function renderActions() {
    $("scenario-actions").replaceChildren();
    const actions = scenario.actions || [];
    $("eligible-count").textContent = `${actions.filter(permitted).length} permitted now / ${actions.length} total`;
    for (const action of actions) {
      const isPermitted = permitted(action);
      const card = element("article", `policy-action${isPermitted ? "" : " unavailable"}`);
      const heading = element("div", "policy-action-heading");
      heading.append(element("h4", "", action.name || action.id), element("span", "policy-action-cost", `${displayNumber(action.tokens)} tokens · ${displayNumber(action.latency_ms, " ms")}`));
      card.append(heading);
      addActionDetail(card, "Needs", labelsFor(action.requires));
      addActionDetail(card, "Blocked by", labelsFor(action.forbids));
      addActionDetail(card, "Makes true", labelsFor(action.sets));
      addActionDetail(card, "Makes false", labelsFor(action.clears));
      if (!isPermitted) card.append(element("p", "action-availability", action.allowed === false ? "Not permitted in this scenario." : "Unavailable until its conditions are met."));
      $("scenario-actions").append(card);
    }
  }

  function loadExample(id) {
    const example = examples.find((entry) => String(entry.id) === id);
    if (!example) return;
    scenario = JSON.parse(JSON.stringify(example.scenario));
    $("scenario-summary").textContent = example.summary || example.title || "";
    renderFacts();
    renderActions();
    refreshRunButton();
    markChanged();
  }

  function renderExamples(data) {
    examples = Array.isArray(data.examples) ? data.examples : [];
    $("policy-example").replaceChildren();
    for (const example of examples) {
      const option = element("option", "", example.title || example.id);
      option.value = String(example.id);
      $("policy-example").append(option);
    }
    if (examples.length) {
      $("policy-fields").disabled = false;
      loadExample(String(examples[0].id));
    } else {
      $("scenario-summary").textContent = "No simulated examples are available yet.";
    }
  }

  function renderProbabilities(data) {
    $("policy-probabilities").replaceChildren();
    const values = Array.isArray(data.probabilities) ? [...data.probabilities] : [];
    values.sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0));
    for (const item of values) {
      const isWinner = data.decision?.id === item.id;
      const row = element("div", `probability-row${isWinner ? " winner" : ""}${item.eligible === false ? " ineligible" : ""}`);
      const header = element("div", "probability-header");
      const name = element("span", "probability-name", item.name || item.id);
      if (item.eligible === false) name.append(element("span", "probability-tag", "Unavailable"));
      header.append(name, element("span", "probability-value", percentage(item.probability)));
      const bar = element("div", "probability-bar");
      bar.setAttribute("aria-hidden", "true");
      const fill = element("div", "probability-fill");
      fill.style.width = `${finite(item.probability) ? Math.min(100, Math.max(0, item.probability * 100)) : 0}%`;
      bar.append(fill);
      row.append(header, bar);
      $("policy-probabilities").append(row);
    }
    if (!values.length) $("policy-probabilities").append(element("p", "empty-small", "No action scores were returned."));
  }

  function renderPlan(planner) {
    $("policy-plan").replaceChildren();
    const plan = Array.isArray(planner.plan) ? planner.plan : [];
    const explored = finite(planner.explored_states) ? ` ${integer.format(planner.explored_states)} states explored.` : "";
    const outcome = planner.outcome === "needs_clarification" ? " More information or different actions are needed." : "";
    $("plan-summary").textContent = `${planner.can_stop === true ? "The goal is already satisfied. No further work is needed." : planner.success === true ? "The search found a path to the stated goal." : "The search did not find a successful path within its limit."}${explored}${outcome}`;
    let currentState = scenario.state;
    for (const step of plan.slice(0, 5)) {
      const actionId = typeof step === "string" ? step : step.action_id ?? step.id;
      const action = scenario.actions.find((entry) => entry.id === actionId);
      const node = element("li");
      node.append(element("h4", "", (typeof step === "object" ? step.name || step.action_name : null) || action?.name || actionId || "Simulated action"));
      const before = finite(step.state_before) ? step.state_before : currentState;
      const after = finite(step.state_after) ? step.state_after
        : action ? (before | Number(action.sets || 0)) & ~Number(action.clears || 0) : null;
      if (finite(after)) {
        const becameTrue = labelsFor(after & ~before);
        const becameFalse = labelsFor(before & ~after);
        const effects = [becameTrue.length ? `Becomes true: ${becameTrue.join("; ")}.` : "", becameFalse.length ? `Becomes false: ${becameFalse.join("; ")}.` : ""].filter(Boolean);
        node.append(element("p", "plan-effect", effects.join(" ") || "No fact changes in this simulated step."));
        currentState = after;
      }
      const tokens = finite(step.tokens) ? step.tokens : action?.tokens;
      const latency = finite(step.latency_ms) ? step.latency_ms : action?.latency_ms;
      if (finite(tokens) || finite(latency)) node.append(element("p", "plan-cost", `${displayNumber(tokens)} tokens · ${displayNumber(latency, " ms")} (supplied simulation costs)`));
      $("policy-plan").append(node);
    }
    if (!plan.length) $("policy-plan").append(element("li", "", planner.can_stop ? "Stop: the required goal facts are already true." : "No action sequence returned."));
  }

  function renderSearchScores(planner) {
    const scores = Array.isArray(planner.action_scores) ? planner.action_scores : [];
    $("search-scores").replaceChildren();
    $("search-scores-details").hidden = scores.length === 0;
    for (const score of scores) {
      const node = element("div", "search-score-row");
      const id = score.action_id ?? score.id;
      const action = scenario.actions.find((entry) => entry.id === id);
      node.append(element("strong", "", score.name || score.action_name || action?.name || id || "Action"));
      const values = Object.entries(score).filter(([key, value]) => !["id", "action_id", "name", "action_name"].includes(key) && (typeof value === "number" || typeof value === "boolean" || typeof value === "string"));
      for (const [key, value] of values) node.append(element("p", "", `${humanize(key)}: ${typeof value === "number" ? number.format(value) : typeof value === "boolean" ? (value ? "Yes" : "No") : value}`));
      $("search-scores").append(node);
    }
  }

  function renderDecision(data) {
    const planner = data.planner || {};
    $("model-choice").textContent = data.decision?.name || data.decision?.id || "No action returned";
    $("model-choice-detail").textContent = `Measured inference: ${displayNumber(data.model_latency_ms, " ms")}. ${finite(data.decision?.probability) ? `Top preference: ${percentage(data.decision.probability)}.` : ""}`;
    $("planner-choice").textContent = planner.chosen_action_name || (planner.can_stop ? "Stop — goal reached" : "No successful action found");
    $("planner-choice-detail").textContent = planner.can_stop ? "The current facts already meet the goal." : "Win first; then weighted token and time cost. Search limited to five actions.";
    const agrees = data.agrees_with_planner;
    $("policy-agreement").textContent = agrees === true
      ? "The learned choice is one of the search reference’s optimal next actions."
      : agrees === false ? "The policy and search disagree. This is a useful failure case to inspect, not a reason to assume the model is right."
        : "Agreement was not reported for this decision.";
    $("policy-agreement").className = `comparison-note${agrees === false ? " disagrees" : ""}`;
    renderProbabilities(data);
    renderPlan(planner);
    renderSearchScores(planner);
    $("decision-json").textContent = JSON.stringify(data, null, 2);
    $("policy-empty").hidden = true;
    $("policy-results").hidden = false;
    $("stale-results").hidden = true;
    hasResult = true;
  }

  $("policy-example").addEventListener("change", (event) => loadExample(event.target.value));
  $("reset-scenario").addEventListener("click", () => loadExample($("policy-example").value));
  $("token-weight").addEventListener("input", markChanged);
  $("latency-weight").addEventListener("input", markChanged);
  $("policy-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!scenario || running || !ready) return;
    const tokenWeight = Number($("token-weight").value);
    const latencyWeight = Number($("latency-weight").value);
    if (!finite(tokenWeight) || !finite(latencyWeight) || tokenWeight < 0 || latencyWeight < 0 || tokenWeight + latencyWeight === 0) {
      showError("Use non-negative costs, with at least one weight greater than zero.");
      return;
    }
    showError("");
    running = true;
    $("policy-fields").disabled = true;
    $("policy-results-panel").setAttribute("aria-busy", "true");
    $("policy-run-status").hidden = false;
    refreshRunButton();
    try {
      const result = await request("/api/policy/decide", {
        method: "POST",
        body: JSON.stringify({ scenario, token_weight: tokenWeight, latency_weight: latencyWeight, max_depth: 5 }),
      });
      renderDecision(result);
    } catch (error) {
      showError(error.message || "The decision could not be evaluated.");
    } finally {
      running = false;
      $("policy-fields").disabled = false;
      $("policy-results-panel").setAttribute("aria-busy", "false");
      $("policy-run-status").hidden = true;
      refreshRunButton();
    }
  });

  async function init() {
    const [status, samples] = await Promise.allSettled([request("/api/policy/status"), request("/api/policy/examples")]);
    const errors = [];
    if (status.status === "fulfilled") updateStatus(status.value);
    else {
      $("policy-connection").className = "connection unavailable";
      $("policy-connection-label").textContent = "Local policy unavailable";
      $("training-status").textContent = "Status unavailable";
      $("model-metrics").replaceChildren(element("p", "field-help", "The local training report could not be loaded."));
      errors.push(status.reason.message);
    }
    if (samples.status === "fulfilled") renderExamples(samples.value);
    else errors.push(samples.reason.message);
    if (errors.length) showError([...new Set(errors)].join("\n"));
    refreshRunButton();
  }

  init();
})();
