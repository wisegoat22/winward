(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const num = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
  const ms = value => Number.isFinite(value) ? `${num.format(value)} ms` : "Not reported";
  let busy = false, ready = false, examples = [];
  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const human = value => {
    if (value === null || value === undefined) return "Not reported";
    if (typeof value === "number") return num.format(value);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };
  const checkNames = {
    validation_retention:"Earlier skills during training validation",
    legacy_retention:"Earlier tasks · goals completed",
    curriculum_retention:"New task combinations · goals completed",
    legacy_action_cost_retention:"Earlier tasks · cost when both succeed",
    changed_goal_action_retention:"Changed goals · correct next action",
    changed_goal_pair_retention:"Changed goals · both choices correct (of 200 pairs)",
    actual_tool_retention:"Real coding fixtures · verified and finished",
    uncertainty_expected_vs_myopic_progress:"Uncertainty · expected completion vs progress rules",
    uncertainty_sampled_vs_myopic_progress:"Uncertainty · sampled completion vs progress rules",
    uncertainty_expected_vs_information_first:"Uncertainty · expected completion vs information rules",
    uncertainty_sampled_vs_information_first:"Uncertainty · sampled completion vs information rules",
    useful_efficiency:"Uncertainty · cost when both succeed"
  };
  function checkValue(check, value) {
    if (value === null || value === undefined) return "Not reported";
    if (check.name === "validation_retention") return value ? "All passed" : "Failed";
    if (check.name.includes("cost") || check.name === "useful_efficiency" || check.name === "changed_goal_pair_retention") return human(value);
    return typeof value === "number" ? `${num.format(value * 100)}%` : human(value);
  }
  async function api(path, body) {
    const response = await fetch(path, body === undefined ? {} : { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The local request failed.");
    return data;
  }
  function showError(message = "") { $("v4-error").textContent = message; $("v4-error").hidden = !message; }
  function lock(value) {
    busy = value; $("v4-fields").disabled = value || !ready;
    $("v4-decide").textContent = value ? "Evaluating locally…" : "Compare the next decision →";
  }
  function stat(label, value) { const item = el("div", undefined, "v2-stat"); item.append(el("strong",value),el("span",label)); $("v4-stats").append(item); }
  function status(data) {
    ready = data.status === "ready";
    $("v4-model").textContent = data.training?.model_name || "No completed v4 checkpoint";
    $("v4-status").textContent = ready ? "TRAINED CANDIDATE" : "CHECKPOINT NOT READY";
    $("v4-limits").textContent = data.limitation;
    $("v4-record").textContent = JSON.stringify(data,null,2);
    $("v4-stats").replaceChildren();
    if (data.training?.parameter_count) stat("Trainable model size · parameters",new Intl.NumberFormat().format(data.training.parameter_count));
    if (data.training?.training_cases) stat("Training examples",new Intl.NumberFormat().format(data.training.training_cases));
    const sampled = data.evaluation?.belief?.sampled;
    for (const [key, label] of [["v4", "V4"], ["v3", "Frozen v3"]]) {
      const result = sampled?.[key];
      if (result) stat(`${label} · same fresh uncertainty episodes completed`, `${result.successes}/${result.cases} (${num.format(result.success_rate * 100)}%)`);
    }
    const promotion = data.evaluation?.promotion;
    const promoted = promotion?.promoted;
    $("promotion-title").textContent = !promotion ? "Promotion pending" : promoted ? "Passed this audit’s promotion checks" : "Not promoted";
    $("promotion-note").textContent = !promotion ? "Training and an independent evaluation must finish before a promotion decision is available. Earlier versions remain available." : promoted ? "Every predeclared check passed on this audit. This is bounded experimental evidence, not general real-world reliability." : "At least one predeclared check failed. Earlier checkpoints remain available unchanged. Inspect the failed checks below before interpreting improvements.";
    $("gate-rows").replaceChildren();
    for (const check of promotion?.checks || []) {
      const row = el("tr");
      const label = el("td",checkNames[check.name] || check.name.replaceAll("_"," "));
      if (check.requirement) label.append(el("p",check.requirement,"field-help"));
      row.append(label,el("td",checkValue(check,check.candidate)),el("td",checkValue(check,check.baseline)),el("td",check.passed ? "PASS" : "FAIL",check.passed ? "" : "v2-warning"));
      $("gate-rows").append(row);
    }
    if (!promotion) { const row=el("tr"),cell=el("td","The frozen-candidate audit is not available yet."); cell.colSpan=4;row.append(cell);$("gate-rows").append(row); }
    lock(false);
  }
  function displayDecision(data) {
    $("v4-choice").textContent=data.learned_choice.action_name;
    $("v4-choice-time").textContent=`Measured decision time: ${ms(data.learned_choice.decision_ms)}. No language tokens generated.`;
    $("v4-reference").textContent=data.reference.chosen_action_name;
    $("v4-agreement").textContent=data.agrees_with_reference ? "The learned choice matches one of the reference’s optimal actions." : "The model and reference disagree. The model’s answer has not been replaced or corrected.";
    $("v4-reference-note").textContent=`Reference search took ${ms(data.reference.planning_ms)}. Its success and cost predictions apply only to these supplied possible worlds.`;
    $("v4-decision-json").textContent=JSON.stringify(data,null,2);
    $("v4-empty").hidden=true;$("v4-decision").hidden=false;
  }
  function displayEpisode(data) {
    $("v4-outcome").textContent=data.success ? "Goal verified in this sampled world." : "The goal was not verified.";
    $("v4-episode-note").textContent=`The model chose each action from the updated belief. Search did not correct it. ${data.steps ?? data.trace?.length ?? 0} actions observed. Probabilities and action effects are supplied simulation assumptions.`;
    $("v4-trace").replaceChildren();
    for (const step of data.trace || []) {
      const item=el("li");item.append(el("strong",step.action_name || step.action_id),
        el("p",`Observed: ${step.observation ?? "No further observation"}. Decision time: ${ms(step.decision_ms)}.`));
      if (step.verified !== undefined) item.append(el("p",step.verified ? "Every remaining hypothesis satisfies the goal." : "The goal is still unverified."));
      $("v4-trace").append(item);
    }
    $("v4-episode-json").textContent=JSON.stringify(data,null,2);$("v4-episode-panel").hidden=false;
  }
  async function run(episode) {
    if (busy || !ready) return; showError();lock(true);
    try {
      const body={example_id:$("v4-example").value,max_depth:Number($("v4-depth").value)};
      const data=await api(episode ? "/api/v4/episode" : "/api/v4/decide",body);
      if (episode) displayEpisode(data); else displayDecision(data);
    } catch(error) { showError(error.message); } finally { lock(false); }
  }
  $("v4-form").addEventListener("submit",event=>{event.preventDefault();run(false);});
  $("v4-episode").addEventListener("click",()=>run(true));
  function changed() { $("v4-description").textContent=examples.find(e=>e.id===$("v4-example").value)?.description || ""; $("v4-decision").hidden=true;$("v4-empty").hidden=false;$("v4-episode-panel").hidden=true; }
  $("v4-example").addEventListener("change",changed);$("v4-depth").addEventListener("change",changed);
  async function init() {
    lock(true);
    try { const [data,items]=await Promise.all([api("/api/v4/status"),api("/api/v4/examples")]);status(data);examples=items.examples;
      for (const entry of examples) {const option=el("option",entry.title);option.value=entry.id;$("v4-example").append(option);}changed();
    } catch(error) {showError(error.message);lock(false);}
  }
  init();
})();
