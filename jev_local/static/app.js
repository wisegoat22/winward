"use strict";

const $ = (id) => document.getElementById(id);
const state = { examples: [], busy: false, benchmarking: false, stop: false, lastResponses: {}, benchmarkRows: [] };
const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
const duration = (value) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const ms = Number(value);
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
};
const readableError = (error) => error instanceof Error ? error.message : String(error);

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? { cache: "no-store" } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch {
    throw new Error("The local server could not be reached. Check that JEV Local is running, then try again.");
  }
  let result;
  try { result = await response.json(); } catch { throw new Error(`The local server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = result.detail || result.error || result.message;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Request failed (${response.status}).`);
  }
  return result;
}

function showError(message) {
  $("global-error").textContent = message;
  $("global-error").hidden = !message;
}

function setBusy(busy, benchmark = false) {
  state.busy = busy;
  state.benchmarking = busy && benchmark;
  $("decision-fields").disabled = busy;
  $("benchmark-button").disabled = busy;
  $("stop-benchmark").hidden = !state.benchmarking;
  $("stop-benchmark").disabled = false;
  $("stop-benchmark").textContent = "Stop after request";
  document.querySelector(".results-panel").setAttribute("aria-busy", String(busy && !benchmark));
  updateChoiceButtons();
}

function updateChoiceButtons() {
  const rows = Array.from($("choices").children);
  rows.forEach((row, index) => {
    row.querySelector(".choice-letter").textContent = String.fromCharCode(65 + index);
    row.querySelector(".choice-name").setAttribute("aria-label", `Choice ${index + 1} name`);
    row.querySelector(".choice-description").setAttribute("aria-label", `Choice ${index + 1} description (optional)`);
    const remove = row.querySelector(".remove-choice");
    remove.disabled = rows.length <= 2 || state.busy;
    remove.setAttribute("aria-label", `Remove choice ${index + 1}`);
  });
  $("add-choice").disabled = rows.length >= 12 || state.busy;
}

function addChoice(choice = {}, focus = false) {
  if ($("choices").children.length >= 12) return;
  const row = element("div", "choice-row");
  const letter = element("span", "choice-letter");
  letter.setAttribute("aria-hidden", "true");
  const inputs = element("div", "choice-inputs");
  const name = element("input", "choice-name");
  Object.assign(name, { type: "text", placeholder: "Choice name", value: choice.name || "", required: true, maxLength: 80,
    pattern: "[A-Za-z0-9_ \\-]+", title: "Use up to 80 letters, numbers, spaces, underscores, or hyphens." });
  const description = element("input", "choice-description");
  Object.assign(description, { type: "text", placeholder: "Description (optional)", value: choice.description || "", maxLength: 500 });
  const remove = element("button", "remove-choice", "×");
  remove.type = "button";
  remove.addEventListener("click", () => {
    if (state.busy || $("choices").children.length <= 2) return;
    const next = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    updateChoiceButtons();
    markCustom();
    next?.querySelector(".choice-name").focus();
  });
  inputs.append(name, description);
  row.append(letter, inputs, remove);
  $("choices").append(row);
  updateChoiceButtons();
  if (focus) name.focus();
}

function markCustom() { $("example-select").value = ""; }

function applyExample(example) {
  $("state").value = typeof example.state === "string" ? example.state : JSON.stringify(example.state, null, 2);
  $("question").value = example.question || "";
  $("choices").replaceChildren();
  for (const choice of example.choices || []) addChoice(choice);
  while ($("choices").children.length < 2) addChoice();
}

function payloadFromForm() {
  const choices = Array.from($("choices").children).map((row) => ({
    name: row.querySelector(".choice-name").value.trim(), description: row.querySelector(".choice-description").value.trim(),
  }));
  if (!$("state").value.trim() || !$("question").value.trim()) throw new Error("Add a situation and a question before scoring the decision.");
  if (choices.some((choice) => !choice.name)) throw new Error("Give every choice a name.");
  if (new Set(choices.map((choice) => choice.name.toLowerCase())).size !== choices.length) throw new Error("Give each choice a distinct name so its score is unambiguous.");
  return { state: $("state").value.trim(), question: $("question").value.trim(), choices,
    threshold: Number($("threshold").value), margin: Number($("margin").value) };
}

function addMetric(parent, value, label) {
  const metric = element("div", "metric");
  metric.append(element("span", "metric-value", value), element("span", "metric-label", label));
  parent.append(metric);
}

function resultHeader(title, tag) {
  const header = element("div", "result-kicker");
  header.append(element("span", "", title), element("span", "method-label", tag));
  return header;
}

function renderScore(result) {
  const container = $("score-result");
  container.replaceChildren();
  container.hidden = false;
  container.append(resultHeader("CHOICE SCORING", "0 GENERATED TOKENS"), element("h3", "result-choice", result.choice || "No choice returned"));
  const review = Boolean(result.needs_review);
  container.append(element("span", `review-badge${review ? " review" : ""}`, review ? "Flagged for review" : "Meets your review settings"));
  container.append(element("p", "review-explanation", `Top score ${percent(result.confidence)} · Lead over next choice ${percent(result.margin)}.`));
  const probabilities = element("div", "probabilities");
  Object.entries(result.probabilities || {}).sort((a, b) => Number(b[1]) - Number(a[1])).forEach(([name, probability]) => {
    const row = element("div", `probability-row${name === result.choice ? " winner" : ""}`);
    const header = element("div", "probability-header");
    header.append(element("span", "probability-name", name), element("span", "probability-value", percent(probability)));
    const bar = element("div", "probability-bar");
    bar.setAttribute("aria-hidden", "true");
    const fill = element("div", "probability-fill");
    fill.style.width = `${Math.min(100, Math.max(0, Number(probability) * 100 || 0))}%`;
    bar.append(fill);
    row.append(header, bar);
    probabilities.append(row);
  });
  const metrics = element("div", "metric-strip");
  addMetric(metrics, duration(result.latency_ms), "Model time");
  addMetric(metrics, String(result.input_tokens ?? "—"), "Input tokens");
  addMetric(metrics, String(result.output_tokens ?? 0), "Output tokens");
  container.append(probabilities, metrics);
}

function renderGeneration(result) {
  const container = $("generation-result");
  container.replaceChildren();
  container.hidden = false;
  container.append(resultHeader("TEXT GENERATION", "SAME MODEL"), element("h3", "result-choice", result.choice || "No valid choice parsed"));
  if (!result.choice) container.append(element("p", "review-explanation", "The generated response did not identify one of the available choices."));
  const text = element("div", "generation-text", result.text || "(No text returned)");
  text.setAttribute("aria-label", "Generated response");
  container.append(text);
  const metrics = element("div", "metric-strip");
  addMetric(metrics, duration(result.latency_ms), "Model time");
  addMetric(metrics, String(result.input_tokens ?? "—"), "Input tokens");
  addMetric(metrics, String(result.output_tokens ?? "—"), "Output tokens");
  container.append(metrics);
  if (result.finish_reason === "length" || result.finish_reason === "max_tokens") container.append(element("p", "review-explanation", "Generation reached the 80-token output limit."));
}

function refreshJSON() { $("json-output").textContent = JSON.stringify(state.lastResponses, null, 2); }

async function runDecision(compare) {
  if (state.busy) return;
  if (!$("decision-form").reportValidity()) return;
  let body;
  try { body = payloadFromForm(); } catch (error) { showError(readableError(error)); return; }
  showError("");
  setBusy(true);
  state.lastResponses = {};
  $("empty-state").hidden = true;
  $("results").hidden = true;
  $("score-result").hidden = true;
  $("generation-result").hidden = true;
  $("comparison-note").hidden = true;
  $("run-status").hidden = false;
  $("run-status").textContent = "Scoring your choices on the local model…";
  $("results-mode").textContent = compare ? "SCORING + GENERATION" : "CHOICE SCORING";
  try {
    const scored = await api("/api/decide", body);
    state.lastResponses.scoring = scored;
    renderScore(scored);
    $("results").hidden = false;
    refreshJSON();
    if (compare) {
      $("run-status").textContent = "Scoring finished. Generating an answer with the same model…";
      const generated = await api("/api/generate", { ...body, max_tokens: 80 });
      state.lastResponses.generation = generated;
      renderGeneration(generated);
      refreshJSON();
      const scoreTime = Number(scored.latency_ms);
      const generationTime = Number(generated.latency_ms);
      let timing = "";
      if (scoreTime > 0 && generationTime > 0) {
        timing = generationTime >= scoreTime ? `Scoring was ${(generationTime / scoreTime).toFixed(1)}× faster in this run. ` : `Generation was ${(scoreTime / generationTime).toFixed(1)}× faster in this run. `;
      }
      $("comparison-note").textContent = `${timing}${scored.choice === generated.choice ? "Both approaches selected the same choice." : "The two approaches returned different choices."} Runs are sequential; timing can vary with warm-up, prompt length, and output length.`;
      $("comparison-note").hidden = false;
    }
  } catch (error) {
    const message = readableError(error);
    showError(message);
    state.lastResponses.error = message;
    refreshJSON();
    if (!state.lastResponses.scoring) $("empty-state").hidden = false;
  } finally {
    $("run-status").hidden = true;
    setBusy(false);
  }
}

function mean(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function normalized(value) { return String(value ?? "").trim().toLowerCase(); }

function renderBenchmark() {
  const table = $("benchmark-rows");
  table.replaceChildren();
  for (const row of state.benchmarkRows) {
    const tr = element("tr");
    const name = element("td", "", row.example.title);
    name.append(element("span", "expected-label", `Expected: ${row.example.expected ?? "Not provided"}`));
    tr.append(name);
    for (const key of ["scoring", "generation"]) {
      const result = row[key];
      const error = row[`${key}Error`];
      const knownExpected = row.example.expected !== undefined && row.example.expected !== null;
      const correct = knownExpected && normalized(result?.choice) === normalized(row.example.expected);
      const cell = element("td", error ? "answer-error" : result && knownExpected ? (correct ? "answer-correct" : "answer-incorrect") : "", error ? "Request failed" : result ? result.choice || "No valid choice" : "—");
      if (error) cell.title = error;
      tr.append(cell);
    }
    tr.append(element("td", "", duration(row.scoring?.latency_ms)), element("td", "", duration(row.generation?.latency_ms)));
    table.append(tr);
  }
  const summary = $("benchmark-summary");
  summary.replaceChildren();
  for (const [key, label] of [["scoring", "Choice scoring"], ["generation", "Text generation"]]) {
    const successful = state.benchmarkRows.filter((row) => row[key]);
    const evaluated = successful.filter((row) => row.example.expected !== undefined && row.example.expected !== null);
    const correct = evaluated.filter((row) => normalized(row[key].choice) === normalized(row.example.expected)).length;
    const times = successful.map((row) => Number(row[key].latency_ms)).filter(Number.isFinite);
    const card = element("div", "benchmark-stat");
    card.append(element("h3", "", label));
    const metrics = element("div", "metric-strip");
    addMetric(metrics, evaluated.length ? `${correct}/${evaluated.length}` : "—", "Expected answers matched");
    addMetric(metrics, times.length ? duration(mean(times)) : "—", "Mean model time");
    addMetric(metrics, times.length ? duration(median(times)) : "—", "Median model time");
    card.append(metrics);
    summary.append(card);
  }
}

async function runBenchmark() {
  if (state.busy) return;
  showError("");
  setBusy(true, true);
  state.stop = false;
  state.benchmarkRows = [];
  $("benchmark-output").hidden = false;
  $("benchmark-status").textContent = "Loading the demo examples…";
  $("benchmark-progress").value = 0;
  renderBenchmark();
  let completed = 0;
  let failed = 0;
  let total = 0;
  try {
    const response = await api("/api/examples");
    const examples = response.examples || [];
    if (!examples.length) throw new Error("No demo examples are available from the local server.");
    total = examples.length * 2;
    $("benchmark-progress").max = total;
    const warmup = { state: examples[0].state, question: examples[0].question, choices: examples[0].choices,
      threshold: Number($("threshold").value), margin: Number($("margin").value) };
    for (const method of ["scoring", "generation"]) {
      if (state.stop) break;
      $("benchmark-status").textContent = `Warming up ${method === "scoring" ? "choice scoring" : "text generation"}… This request is excluded from measured results.`;
      await api(method === "scoring" ? "/api/decide" : "/api/generate", method === "scoring" ? warmup : { ...warmup, max_tokens: 80 });
    }
    for (let index = 0; index < examples.length; index += 1) {
      if (state.stop) break;
      const example = examples[index];
      const row = { example };
      state.benchmarkRows.push(row);
      const body = { state: example.state, question: example.question, choices: example.choices,
        threshold: Number($("threshold").value), margin: Number($("margin").value) };
      for (const method of index % 2 === 0 ? ["scoring", "generation"] : ["generation", "scoring"]) {
        if (state.stop) break;
        $("benchmark-status").textContent = `Example ${index + 1} of ${examples.length}: ${example.title} · ${method === "scoring" ? "scoring choices" : "generating an answer"}…`;
        renderBenchmark();
        try {
          row[method] = await api(method === "scoring" ? "/api/decide" : "/api/generate", method === "scoring" ? body : { ...body, max_tokens: 80 });
        } catch (error) {
          row[`${method}Error`] = readableError(error);
          failed += 1;
        }
        completed += 1;
        $("benchmark-progress").value = completed;
        renderBenchmark();
      }
    }
    $("benchmark-status").textContent = `${state.stop ? "Stopped" : "Complete"} · ${completed} of ${total} measured requests finished${failed ? ` · ${failed} failed (hover the failed result for details)` : ""}. Warm-up is excluded. Results include successful requests only.`;
  } catch (error) {
    showError(readableError(error));
    $("benchmark-status").textContent = "Benchmark could not finish. Any completed results remain below.";
  } finally {
    setBusy(false);
  }
}

async function initialize() {
  addChoice();
  addChoice();
  const initial = await Promise.allSettled([api("/api/health"), api("/api/examples")]);
  const [healthResult, examplesResult] = initial;
  if (healthResult.status === "fulfilled") {
    const health = healthResult.value;
    const ready = health.status === "ready" || health.status === "ok";
    $("connection").className = `connection ${ready ? "ready" : "unavailable"}`;
    $("connection-label").textContent = ready ? "Local model connected" : "Local server connected";
    $("model-name").textContent = health.model || "Model name unavailable";
    $("runtime-name").textContent = health.runtime || "Local inference";
  } else {
    $("connection").className = "connection unavailable";
    $("connection-label").textContent = "Local server unavailable";
    $("model-name").textContent = "Start the local server to connect.";
    showError(readableError(healthResult.reason));
  }
  if (examplesResult.status === "fulfilled") {
    state.examples = examplesResult.value.examples || [];
    state.examples.forEach((example, index) => {
      const option = element("option", "", example.title);
      option.value = String(index);
      $("example-select").append(option);
    });
    if (state.examples.length && !$("state").value && !$("question").value && !Array.from($("choices").querySelectorAll("input")).some((input) => input.value)) {
      applyExample(state.examples[0]);
      $("example-select").value = "0";
    }
  } else if (healthResult.status === "fulfilled") {
    showError(`The example list could not be loaded. You can still write your own decision. ${readableError(examplesResult.reason)}`);
  }
}

$("decision-form").addEventListener("submit", (event) => { event.preventDefault(); runDecision(false); });
$("compare-button").addEventListener("click", () => runDecision(true));
$("add-choice").addEventListener("click", () => { if (!state.busy) { addChoice({}, true); markCustom(); } });
$("example-select").addEventListener("change", (event) => {
  if (event.target.value !== "" && !state.busy) applyExample(state.examples[Number(event.target.value)]);
});
for (const id of ["state", "question", "choices"]) $(id).addEventListener("input", markCustom);
for (const id of ["threshold", "margin"]) $(id).addEventListener("input", () => { $(`${id}-value`).value = `${Math.round(Number($(id).value) * 100)}%`; });
$("benchmark-button").addEventListener("click", runBenchmark);
$("stop-benchmark").addEventListener("click", () => {
  state.stop = true;
  $("stop-benchmark").disabled = true;
  $("stop-benchmark").textContent = "Finishing current request…";
  $("benchmark-status").textContent = "Stopping after the current request finishes. Completed results will be kept.";
});
$("copy-json").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.lastResponses, null, 2));
    $("copy-json").textContent = "Copied";
    setTimeout(() => { $("copy-json").textContent = "Copy JSON"; }, 1800);
  } catch {
    $("copy-json").textContent = "Select text to copy";
    const range = document.createRange();
    range.selectNodeContents($("json-output"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $("json-output").focus();
  }
});
initialize();
