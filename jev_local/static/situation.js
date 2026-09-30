(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  const samples = {
    unknown: {
      situation: "The save button in our app stopped working. I have not reproduced the problem, read the error, or inspected the relevant code yet.",
      goal: "Make the save button work and verify the fix with a focused test."
    },
    test: {
      situation: "I reproduced a failing test, inspected the relevant code, and identified the cause. I have applied a small fix, but I have not run the test again since the change.",
      goal: "Fix the bug and get a passing test that verifies the change."
    },
    done: {
      situation: "I inspected the code, found the cause, and applied the fix. The focused test passes after the fix, and I checked that the requested behavior works. There is no further requested work.",
      goal: "Fix the reported bug and verify that the requested behavior works."
    }
  };
  let busy = false;
  let revision = 0;
  let current = null;
  const node = (tag, value, className) => {
    const element = document.createElement(tag);
    if (value !== undefined && value !== null) element.textContent = String(value);
    if (className) element.className = className;
    return element;
  };
  const list = value => Array.isArray(value) ? value : [];
  const text = value => typeof value === "string" ? value : "";
  function showError(message = "") {
    $("situation-error").textContent = message;
    $("situation-error").hidden = !message;
  }
  function lock(value, step = "") {
    busy = value;
    $("interpret-button").disabled = value;
    $("decide-button").disabled = value || !current;
    $("fact-fields").disabled = value;
    $("restriction-fields").disabled = value;
    $("task-kind").disabled = value || !current;
    document.querySelectorAll("[data-sample]").forEach(button => { button.disabled = value; });
    $("interpret-button").textContent = value && step === "read" ? "Reading your situation…" : "Read my situation →";
    $("decide-button").textContent = value && step === "decide" ? "Choosing your next step…" : "Get my next step →";
    $("reading-status").hidden = !(value && step === "read");
    $("decision-status").hidden = !(value && step === "decide");
    $("situation-form").setAttribute("aria-busy", String(value && step === "read"));
    $("decision-form").setAttribute("aria-busy", String(value && step === "decide"));
  }
  function clearRecommendation() {
    $("recommendation-panel").hidden = true;
  }
  function invalidate() {
    revision += 1;
    current = null;
    clearRecommendation();
    $("situation-draft").hidden = true;
    $("situation-help-panel").hidden = true;
    $("situation-empty").hidden = false;
    $("review-heading").textContent = "Your next step starts here.";
    $("decide-button").disabled = true;
    showError();
  }
  async function api(path, body) {
    let response;
    try {
      response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch {
      throw new Error("Could not reach the local app. Check that Winward is running on this Mac, then try again.");
    }
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "";
      if (response.status === 429) throw new Error(detail || "The local model is working on another request. Wait for it to finish, then try again.");
      if (response.status === 503) throw new Error(detail || "The local model is not ready yet. Check that its download and setup have finished, then try again.");
      if (response.status === 422) throw new Error(detail || "Please check your situation and desired win, then read the situation again.");
      throw new Error(detail || "The local request could not be completed. Please try again.");
    }
    return data;
  }
  function populateList(id, values) {
    $(id).replaceChildren();
    list(values).forEach(value => { if (typeof value === "string" && value) $(id).append(node("li", value)); });
  }
  function actionList(actions, facts) {
    const labels = new Map(list(facts).map(fact => [fact.id, fact.label]));
    const label = value => labels.get(value) || text(value).replaceAll("_", " ");
    $("draft-actions").replaceChildren();
    list(actions).forEach(action => {
      const item = node("li");
      const blocked = action.allowed === false;
      const available = !blocked && action.available === true;
      const status = blocked ? "Blocked by your constraints" : available ? "Available now" : "Needs earlier steps or already completed";
      item.append(node("strong", action.name), node("span", status, `action-status ${blocked ? "action-blocked" : available ? "action-available" : "action-waiting"}`), node("p", action.description));
      if (action.cost_label) item.append(node("p", action.cost_label, "action-cost"));
      if (list(action.requires).length) item.append(node("p", `Needs: ${action.requires.map(label).join("; ")}.`));
      if (list(action.effects).length) item.append(node("p", `Helps establish: ${action.effects.map(label).join("; ")}.`));
      $("draft-actions").append(item);
    });
  }
  function interpretationTime(data) {
    const timing = data.interpretation || {};
    const details = [];
    if (Number.isFinite(timing.latency_ms)) details.push(`Read in ${number.format(timing.latency_ms / 1000)} seconds`);
    if (Number.isFinite(timing.input_tokens) && Number.isFinite(timing.output_tokens)) details.push(`${number.format(timing.input_tokens + timing.output_tokens)} language tokens`);
    $("interpretation-time").textContent = details.join(" · ");
  }
  function showClarification(data) {
    current = null;
    $("situation-empty").hidden = true;
    $("situation-draft").hidden = true;
    $("situation-help-panel").hidden = false;
    $("review-heading").textContent = "Let’s make the situation clearer.";
    $("clarification-heading").textContent = data.status === "unsupported" ? "This situation is outside the current model’s scope." : "A little more information is needed.";
    $("clarification-summary").textContent = text(data.draft?.summary) || (data.status === "unsupported" ? "This version can suggest next actions for software-agent tasks." : "Add the missing details to your description so the decision can use them.");
    populateList("clarification-questions", data.questions);
    $("clarification-limitation").textContent = text(data.limitation);
    $("clarification-heading").focus();
  }
  function showDraft(data, input) {
    current = { data, input };
    $("situation-empty").hidden = true;
    $("situation-help-panel").hidden = true;
    $("situation-draft").hidden = false;
    $("review-heading").textContent = "Check the facts before deciding.";
    $("draft-summary").textContent = text(data.draft?.summary);
    $("draft-goal").textContent = text(data.goal_label) || input.goal;
    $("draft-limitation").textContent = text(data.limitation);
    $("task-kind").value = text(data.draft?.task_kind);
    $("action-preview").hidden = false;
    $("actions-stale-note").hidden = true;
    $("draft-restrictions").replaceChildren();
    const restrictionNames = { no_edits: "Do not edit files", no_running_tests: "Do not run tests" };
    list(data.draft?.restrictions).forEach((restriction, index) => {
      const label = node("label", undefined, "fact-label restriction-label");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.id = `confirmed-restriction-${index}`;
      checkbox.value = String(index);
      checkbox.checked = true;
      const wording = node("span", undefined, "fact-text");
      wording.append(node("span", restrictionNames[restriction.kind] || "A constraint from your description"));
      if (restriction.evidence) wording.append(node("q", text(restriction.evidence)));
      label.append(checkbox, wording);
      $("draft-restrictions").append(label);
    });
    $("restrictions-section").hidden = !list(data.draft?.restrictions).length;
    $("draft-facts").replaceChildren();
    list(data.facts).forEach((fact, index) => {
      const label = node("label", undefined, "fact-label");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.id = `confirmed-fact-${index}`;
      checkbox.value = text(fact.id);
      checkbox.checked = fact.confirmed === true;
      const wording = node("span", undefined, "fact-text");
      wording.append(node("span", text(fact.label)));
      if (fact.evidence) wording.append(node("q", text(fact.evidence)));
      label.append(checkbox, wording);
      $("draft-facts").append(label);
    });
    if (!list(data.facts).length) $("draft-facts").append(node("p", "No completed steps have been established yet.", "field-help"));
    populateList("draft-unknowns", data.draft?.unknowns);
    $("unknowns-section").hidden = !list(data.draft?.unknowns).length;
    actionList(data.actions, data.facts);
    interpretationTime(data);
    $("draft-heading").focus();
  }
  function showRecommendation(data) {
    if (data.status !== "recommended" || !data.choice) {
      showClarification(data);
      return;
    }
    $("recommendation-name").textContent = text(data.choice.name);
    $("recommendation-description").textContent = text(data.choice.description);
    $("recommendation-reason").textContent = text(data.reason);
    $("recommendation-check").textContent = text(data.next_check);
    populateList("recommendation-assumptions", data.assumptions);
    $("assumptions-section").hidden = !list(data.assumptions).length;
    const sources = [];
    if (data.source?.reader) sources.push(`Description read by ${data.source.reader}`);
    if (data.source?.decision) sources.push(`Action chosen by ${data.source.decision}`);
    if (Number.isFinite(data.timing?.decision_ms)) sources.push(`Decision: ${number.format(data.timing.decision_ms)} ms`);
    $("recommendation-source").textContent = sources.join(" · ");
    $("recommendation-limitation").textContent = text(data.limitation);
    if (Array.isArray(data.actions) && current) {
      actionList(data.actions, current.data.facts);
      $("action-preview").hidden = false;
      $("actions-stale-note").hidden = true;
    }
    $("recommendation-panel").hidden = false;
    $("recommendation-name").focus();
  }
  $("situation-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !$("situation-form").reportValidity()) return;
    const input = { situation: $("situation-input").value.trim(), goal: $("goal-input").value.trim() };
    if (!input.situation || !input.goal) { showError("Add both a situation and the win you want to achieve."); return; }
    if (input.situation.length < 10 || input.goal.length < 3) { showError("Use at least 10 characters for the situation and 3 characters for the desired win."); return; }
    invalidate();
    const startedRevision = revision;
    lock(true, "read");
    try {
      const data = await api("/api/situation/interpret", input);
      if (revision !== startedRevision) return;
      if (data.status === "ready" && data.draft) showDraft(data, input);
      else showClarification(data);
    } catch (error) {
      if (revision === startedRevision) showError(error.message);
    } finally { lock(false); }
  });
  $("decision-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !current) return;
    showError();
    clearRecommendation();
    const startedRevision = revision;
    const confirmed = Array.from($("draft-facts").querySelectorAll("input:checked"), input => input.value);
    const keptRestrictions = new Set(Array.from($("draft-restrictions").querySelectorAll("input:checked"), input => Number(input.value)));
    const draft = { ...current.data.draft, restrictions: list(current.data.draft.restrictions).filter((restriction, index) => keptRestrictions.has(index)) };
    const body = { ...current.input, draft, confirmed_facts: confirmed, max_depth: 5 };
    lock(true, "decide");
    try {
      const data = await api("/api/situation/decide", body);
      if (revision === startedRevision) showRecommendation(data);
    } catch (error) {
      if (revision === startedRevision) showError(error.message);
    } finally { lock(false); }
  });
  $("situation-input").addEventListener("input", invalidate);
  $("goal-input").addEventListener("input", invalidate);
  $("draft-facts").addEventListener("change", () => {
    revision += 1;
    clearRecommendation();
    showError();
    $("action-preview").hidden = true;
    $("actions-stale-note").hidden = false;
  });
  $("draft-restrictions").addEventListener("change", () => {
    revision += 1;
    clearRecommendation();
    showError();
    $("action-preview").hidden = true;
    $("actions-stale-note").hidden = false;
  });
  $("task-kind").addEventListener("change", () => {
    if (!current || busy) return;
    current.data.draft.task_kind = $("task-kind").value;
    revision += 1;
    clearRecommendation();
    showError();
    $("action-preview").hidden = true;
    $("actions-stale-note").hidden = false;
  });
  document.querySelectorAll("[data-sample]").forEach(button => {
    button.addEventListener("click", () => {
      if (busy) return;
      const sample = samples[button.dataset.sample];
      if (!sample) return;
      $("situation-input").value = sample.situation;
      $("goal-input").value = sample.goal;
      invalidate();
      $("situation-input").focus();
    });
  });
  lock(false);
})();
