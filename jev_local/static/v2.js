(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
  const integer = new Intl.NumberFormat();
  const pct = x => Number.isFinite(x) ? `${number.format(x * 100)}%` : "Not measured";
  const ms = x => Number.isFinite(x) ? `${number.format(x)} ms` : "Not measured";
  let examples = [], busy = false;
  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "The request could not be completed.");
    return value;
  }
  function error(message = "") { $("lab-error").textContent = message; $("lab-error").hidden = !message; }
  function stat(label, value) {
    const item = el("div", undefined, "v2-stat");
    item.append(el("strong", value), el("span", label));
    $("training-stats").append(item);
  }
  function displayStatus(data) {
    const report = data.training;
    $("model-label").textContent = report?.model_name || "Checkpoint not trained yet";
    $("checkpoint-status").textContent = data.status === "ready" ? "EXPERIMENTAL CHECKPOINT READY" : "TRAINING REQUIRED";
    $("lab-limits").textContent = data.limitations;
    $("training-record").textContent = JSON.stringify(data, null, 2);
    $("training-stats").replaceChildren();
    if (report) {
      stat("Trainable parameters", integer.format(report.parameter_count));
      stat("Training examples", integer.format(report.training_cases));
    }
    const audit = data.evaluation;
    const baseline = audit?.matched_baseline;
    if (audit) {
      stat("Fresh paired-goal decisions · candidate", pct(audit.fresh_goal_changes?.neural_accuracy));
      stat("Same paired goals · v1", pct(baseline?.fresh_goal_changes?.neural_accuracy));
      stat("New graph tasks completed · candidate", pct(audit.closed_loop?.methods?.neural?.success_rate));
      stat("Same graph tasks completed · v1", pct(baseline?.closed_loop?.methods?.neural?.success_rate));
      const current = audit.closed_loop?.methods?.neural?.success_rate;
      const old = baseline?.closed_loop?.methods?.neural?.success_rate;
      const legacy = audit.legacy_fresh?.closed_loop?.methods?.neural?.success_rate;
      const legacyOld = baseline?.legacy_fresh?.closed_loop?.methods?.neural?.success_rate;
      const regressed = current < old || legacy < legacyOld;
      $("audit-summary").textContent = `This candidate remains experimental; v1 is retained in the first lab. ${regressed ? "At least one task-completion measure regressed against v1. " : "These results do not establish general agent reliability. "}On fresh instances of the earlier task families, completion was ${pct(legacy)} versus v1’s ${pct(legacyOld)}. Tests were generated after checkpoint selection; the synthetic task templates were inspected during development.`;
      $("audit-summary").className = `interpretation-note${regressed ? " v2-warning" : ""}`;
    }
    const taskAudit = data.sandbox_audit;
    $("coding-metrics").replaceChildren();
    const labels = { neural: "Trained candidate", evidence_first: "Evidence-first rules", random: "Random eligible action" };
    for (const [name, values] of Object.entries(taskAudit?.methods || {})) {
      const row = el("tr");
      row.append(el("td", labels[name] || name), el("td", `${values.successes}/${values.cases} · ${pct(values.success_rate)}`),
                 el("td", number.format(values.mean_actions)), el("td", ms(values.mean_wall_ms)));
      $("coding-metrics").append(row);
    }
    if (!taskAudit) {
      const row = el("tr"); const cell = el("td", "Coding-task audit has not been recorded yet."); cell.colSpan = 4; row.append(cell); $("coding-metrics").append(row);
    }
    $("coding-note").textContent = taskAudit
      ? `${taskAudit.task_provenance} Lower time or fewer actions can mean giving up early; read cost beside verified wins. Time includes decisions, tools, and temporary-project setup/cleanup. Token costs are estimates; no language tokens are generated.`
      : "The controls below run actual checks on trusted generated files. The neural policy does not write code.";
  }
  function branchList(tree, parent, depth = 0) {
    if (!tree || depth > 5) return;
    for (const branch of tree.branches || []) {
      const item = el("li");
      item.append(el("strong", `If: ${String(branch.observation).replaceAll("_", " ")}`),
                  el("p", `${pct(branch.probability)} modeled probability → ${branch.next?.action_name || "No further action"}`));
      if (branch.next?.branches?.length) { const list = el("ol"); branchList(branch.next, list, depth+1); item.append(list); }
      parent.append(item);
    }
  }
  function displayBelief(data) {
    const plan = data.reference;
    $("belief-choice").textContent = plan.chosen_action_name;
    $("belief-summary").textContent = `${pct(plan.expected_verified_success)} modeled chance of a verified goal within the plan. ${number.format(plan.expected_steps)} expected actions; ${ms(plan.planning_ms)} measured planning time; ${plan.explored_nodes} belief states searched. ${plan.search_complete ? "Search completed within its limit." : "Search budget exhausted; this is a partial plan."}`;
    $("belief-branches").replaceChildren();
    branchList(plan.plan, $("belief-branches"));
    $("belief-json").textContent = JSON.stringify(data, null, 2);
    $("belief-result").hidden = false;
  }
  function displayTask(data) {
    $("task-outcome").textContent = data.success ? "The current goal passed its checks." : "The goal was not verified.";
    $("task-time").textContent = `${data.steps} actions · ${ms(data.wall_ms)} total`;
    $("task-summary").textContent = `${data.edits} supplied patches applied; ${data.checks} real checks; ${data.failed_checks} failed checks. Decisions took ${ms(data.decision_ms)}. ${data.deferred ? "The policy deferred before verifying the goal. " : ""}${data.horizon_exhausted ? "The action budget was reached. " : ""}${data.success && !data.done ? "Checks passed but the policy had no remaining turn to signal finish. " : ""}The model generated zero language tokens. Tool-token estimates are not measured token usage.`;
    $("task-trace").replaceChildren();
    for (const step of data.trace) {
      const item = el("li");
      item.append(el("strong", step.action_name || step.action_id), el("p", step.summary),
                  el("p", `Decision ${ms(step.decision_ms)} · Tool ${ms(step.elapsed_ms)}${step.goal_changed ? " · GOAL CHANGED: previous check is now stale" : ""}`));
      const evidence = [step.diff, step.stdout, step.stderr].filter(Boolean).join("\n");
      if (evidence) { const details = el("details"); details.append(el("summary", "See the actual edit or check output"), el("pre", evidence)); item.append(details); }
      $("task-trace").append(item);
    }
    $("task-json").textContent = JSON.stringify(data, null, 2);
    $("task-result").hidden = false;
  }
  function lock(value) {
    busy = value;
    $("task-fields").disabled = value;
    $("belief-run").disabled = value;
    $("task-run").textContent = value ? "Running local actions…" : "Run the controlled task →";
  }
  $("belief-example").addEventListener("change", () => {
    $("belief-description").textContent = examples.find(x => x.id === $("belief-example").value)?.description || "";
    $("belief-result").hidden = true;
  });
  $("belief-form").addEventListener("submit", async event => {
    event.preventDefault(); if (busy) return; error(); lock(true);
    try { displayBelief(await request("/api/v2/uncertainty", { example_id: $("belief-example").value, max_depth: 5 })); }
    catch (err) { error(err.message); } finally { lock(false); }
  });
  $("task-form").addEventListener("submit", async event => {
    event.preventDefault(); if (busy) return; error(); lock(true);
    try {
      displayTask(await request("/api/v2/sandbox", { kind: $("task-kind").value, seed: Number($("task-seed").value),
        changed_goal: $("task-changed").checked, uncertain: $("task-uncertain").checked, policy: $("task-policy").value }));
    } catch (err) { error(err.message); } finally { lock(false); }
  });
  async function init() {
    try {
      const [status, options] = await Promise.all([request("/api/v2/status"), request("/api/v2/examples")]);
      displayStatus(status); examples = options.uncertainty;
      for (const item of examples) { const option = el("option", item.title); option.value = item.id; $("belief-example").append(option); }
      $("belief-description").textContent = examples[0]?.description || "";
      for (const kind of options.task_kinds) { const option = el("option", kind.replaceAll("_", " ")); option.value = kind; $("task-kind").append(option); }
    } catch (err) { error(err.message); }
  }
  init();
})();
