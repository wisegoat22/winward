"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const states = {planned: "Planned", preparing: "Preparing", initialized: "Initialized", weights_updated: "Weights updated", unavailable: "No readable report"};
  const purposes = {capacity_probe: "Capacity check", pilot: "Learning pilot", training: "Training run", unspecified: "Purpose unspecified"};
  const labels = {"32m": "32M", "135m": "135M", "500m": "500M", "1b": "1B", "4b": "4.03B", "4.86m": "4.86M", "19m": "19.16M", "171m": "170.73M", "4.25b": "4.25B"};
  const trackNames = {byte: "Scratch workflow reader", structured: "Trained policy growth"};
  const number = (value) => Number.isFinite(value) ? value.toLocaleString("en-US") : "Not recorded";
  const percent = (value, digits = 1) => Number.isFinite(value) ? `${(100 * value).toFixed(digits)}%` : "Not measured";
  const decimal = (value, suffix, digits = 1) => Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "Not recorded";
  function node(tag, text, className) { const item = document.createElement(tag); if (text !== undefined) item.textContent = text; if (className) item.className = className; return item; }
  function age(seconds) { if (!Number.isFinite(seconds)) return "Time not recorded"; if (seconds < 60) return "Less than a minute ago"; if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`; return `${Math.floor(seconds / 3600)} hours ago`; }
  function stateText(run) { const recorded = {training: "Training reported", completed: "Run completed", stopped: "Run stopped", failed: "Run failed", preparing: "Preparing", initialized: "Initialized", initializing: "Preparing"}[run.reported_status]; return recorded || states[run.state] || "Unknown"; }
  function list(id, values) { const target = $(id); target.replaceChildren(...values.map(value => node("li", value))); target.hidden = !values.length; }
  function metric(value, label) { const item = node("div", undefined, "v2-stat"); item.append(node("strong", value), node("span", label)); return item; }
  function renderAudit(audit) {
    const complete = audit?.status === "completed";
    $("audit-title").textContent = complete ? "4.25B versus the original v4" : audit?.status === "invalid" ? "Final audit unavailable" : "Final audit pending";
    $("audit-note").textContent = audit?.note || "The final audit summary has not been published yet.";
    $("audit-results").hidden = !complete;
    if (!complete) return;
    $("audit-candidate-score").textContent = percent(audit.next_action.candidate_accuracy, 2);
    $("audit-baseline-score").textContent = percent(audit.next_action.baseline_accuracy, 2);
    $("audit-retention-score").textContent = `${number(audit.retention.passed_checks)} / ${number(audit.retention.total_checks)}`;
    $("audit-action-note").textContent = `Correct next-action choices on ${number(audit.cases)} held-out cases: our 4.25B ${number(audit.next_action.candidate_correct)}; original v4 ${number(audit.next_action.baseline_correct)}. ${audit.retention.all_measured_retention_checks_passed ? "All measured retention checks passed." : "Some measured retention checks failed."} This does not automatically promote the model.`;
    $("audit-candidate-completion").textContent = percent(audit.rollouts.candidate_success, 2);
    $("audit-baseline-completion").textContent = percent(audit.rollouts.baseline_success, 2);
    $("audit-reference-completion").textContent = percent(audit.rollouts.reference_success, 2);
    $("audit-rollout-note").textContent = `Mean expected verified completion on ${number(audit.rollouts.denominator)} initially unfinished, reference-reachable cases from ${number(audit.rollout_cases)} rollout cases. Probabilistic branches are integrated; these percentages are not observed real-tool success counts.`;
    const timingScope = audit.timing.scope.trim().replace(/\.$/, "");
    $("audit-timing").textContent = `Decision time: our 4.25B ${decimal(audit.timing.candidate_ms, " ms", 2)}; original v4 ${decimal(audit.timing.baseline_ms, " ms", 2)}. ${timingScope}. These measurements are not cold-start loading time.`;
    $("audit-source").textContent = `Frozen checkpoint ${audit.model_sha256.slice(0, 12)}… · audit protocol ${audit.protocol_sha256.slice(0, 12)}…`;
    $("audit-scope").textContent = audit.scope;
  }
  function render(data) {
    renderAudit(data.final_audit);
    for (const track of data.tracks || []) {
      $(`${track.id}-target-count`).textContent = number(track.target_parameters);
      $(`${track.id}-ladder`).replaceChildren(...track.stages.map(stage => {
        const item = node("li"); item.dataset.state = stage.state;
        const state = stage.source_reference && stage.state === "planned" ? "Source reference" : states[stage.state] || "Unknown";
        item.append(node("strong", labels[stage.preset] || stage.preset), node("span", `${number(stage.parameters)} parameters`, "exact-count"), node("span", state, "scale-badge"));
        return item;
      }));
    }
    list("scale-limitations", data.limitations || []);
    $("history-empty").hidden = data.runs.length > 0;
    $("run-rows").replaceChildren(...data.runs.map(run => {
      const row = node("tr"), name = node("td", run.name);
      name.append(node("span", `${number(run.parameters)} parameters · ${age(run.report_age_seconds)}`, "run-subtext"));
      row.append(name, node("td", trackNames[run.track] || "Unspecified"), node("td", purposes[run.purpose] || "Unspecified"), node("td", stateText(run)), node("td", number(run.total_optimizer_updates ?? run.optimizer_steps)), node("td", percent(run.validation?.accuracy)), node("td", decimal(run.peak_mlx_gib, " GiB")));
      return row;
    }));
    const latest = data.runs.find(run => run.name === data.latest_run) || data.runs[0];
    if (!latest) {
      $("latest-title").textContent = "Waiting for the first report.";
      $("latest-purpose").hidden = true;
      $("latest-summary").textContent = "No scaling run reports are currently available. This page refreshes automatically.";
      $("latest-metrics").replaceChildren();
      $("step-progress").hidden = true; $("validation-panel").hidden = true;
      $("checkpoint-note").textContent = "";
      list("latest-warnings", data.warnings || []); return;
    }
    $("latest-title").textContent = latest.name;
    $("latest-purpose").hidden = false;
    $("latest-purpose").textContent = `${trackNames[latest.track] || "Unspecified track"} · ${purposes[latest.purpose] || "Purpose unspecified"}`;
    const descriptions = {
      capacity_probe: "This checks whether the model can be initialized and updated on this Mac. It does not establish useful decision-making skill.",
      pilot: "This small learning experiment checks whether the training setup improves next-action choices.",
      training: "This run updates our model using synthetic next-action examples. Its validation results guide development."
    };
    const stale = latest.report_stale && latest.reported_status === "training" ? " No recent update has arrived; the report alone cannot confirm the process is still running." : "";
    const growth = latest.track === "structured" ? " Width growth starts from our trained policy. Expanding it does not add skills by itself." : " This model reads the compact byte-workflow language.";
    $("latest-summary").textContent = `${stateText(latest)} · ${age(latest.report_age_seconds)}. ${descriptions[latest.purpose] || "Waiting for a declared experiment purpose."}${growth}${stale}`;
    $("latest-metrics").replaceChildren(
      metric(number(latest.parameters), "Parameters in this run"),
      metric(number(latest.total_optimizer_updates ?? latest.optimizer_steps), "Total recorded optimizer updates"),
      metric(decimal(latest.peak_mlx_gib, " GiB"), "Peak MLX allocation, not total Mac memory"),
      metric(decimal(latest.step_seconds, " s"), "Time for the last training step"),
      latest.track === "structured" ? metric(decimal(latest.training_seconds, " s"), "Recorded training time") : metric(decimal(latest.padded_tokens_per_second, "/s", 0), "Training tokens per second, including padding")
    );
    $("step-progress").hidden = !Number.isFinite(latest.planned_steps);
    if (Number.isFinite(latest.planned_steps)) {
      $("step-meter").max = latest.planned_steps; $("step-meter").value = latest.optimizer_steps;
      $("step-caption").textContent = `${number(latest.optimizer_steps)} of ${number(latest.planned_steps)} updates in this run`;
      $("step-percent").textContent = percent(Math.min(1, latest.optimizer_steps / latest.planned_steps));
    }
    $("validation-panel").hidden = !latest.validation && !latest.initial_validation;
    $("validation-score").textContent = percent(latest.validation?.accuracy);
    $("baseline-score").textContent = percent(latest.majority_baseline);
    $("initial-score").textContent = percent(latest.initial_validation?.accuracy);
    const best = latest.best_validation ? `Best recorded during this run: ${percent(latest.best_validation.accuracy)}${Number.isFinite(latest.best_validation.step) ? ` at update ${number(latest.best_validation.step)}` : ""}. ` : "";
    const validationScope = latest.track === "structured" ? "This checks the selected candidate against the supplied teacher." : "This checks the first action byte with all outputs competing.";
    $("validation-note").textContent = `${latest.validation ? `${number(latest.validation.examples)} development examples. ` : "No validation after updates has been recorded yet. "}${best}${validationScope} It is not a final audit, a real task completion score, or proof of general language understanding.`;
    $("checkpoint-note").textContent = latest.checkpoint_reported ? `A checkpoint was reported at update ${number(latest.checkpoint_step)}. Saving weights does not promote this model or replace the model on the Try page.` : "No completed checkpoint has been reported for this run yet.";
    const warnings = [...(data.warnings || []), ...(latest.warnings || [])];
    if (latest.parameter_count_matches_plan === false) warnings.push("This run’s parameter count differs from the planned architecture, so it does not mark that ladder stage as reached.");
    list("latest-warnings", warnings);
  }
  let timer, busy = false;
  async function refresh() {
    if (busy || document.hidden) return;
    busy = true; clearTimeout(timer);
    try {
      const response = await fetch("/api/scale/status", {cache: "no-store"});
      if (!response.ok) throw new Error("The local progress reports are unavailable.");
      const data = await response.json();
      render(data); $("scale-error").hidden = true;
      $("refresh-status").textContent = `Refreshed at ${new Date().toLocaleTimeString()}. Checks every 15 seconds while this page is visible.`;
    } catch (error) { $("scale-error").textContent = "Could not refresh local progress. Any values shown are from the last successful refresh. The page will try again."; $("scale-error").hidden = false; }
    finally { busy = false; timer = setTimeout(refresh, 15000); }
  }
  document.addEventListener("visibilitychange", () => { clearTimeout(timer); if (!document.hidden) refresh(); });
  refresh();
})();
