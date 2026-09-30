// Execute the real form script against a tiny DOM and mocked local responses.
// No browser, model, network request, or software-agent tool is used.
const fs = require("fs");
const vm = require("vm");
const assert = require("assert/strict");
const path = require("path");

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this.listeners = {};
    this.dataset = {}; this.hidden = false; this.disabled = false;
    this.value = ""; this.checked = false; this.textContent = "";
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  setAttribute() {}
  focus() {}
  reportValidity() { return true; }
  querySelectorAll(selector) {
    assert.equal(selector, "input:checked");
    const found = [];
    const visit = node => { if (node.tagName === "INPUT" && node.checked) found.push(node); node.children.forEach(visit); };
    this.children.forEach(visit); return found;
  }
  set innerHTML(_) { throw Error("Untrusted UI values must use textContent"); }
}

const source = fs.readFileSync(path.join(__dirname, "../jev_local/static/situation.js"), "utf8");
const flush = () => new Promise(setImmediate);
const readyDraft = {
  status: "ready", draft: {domain: "software", task_kind: "investigate", summary: "Investigate.", confirmed: [], restrictions: [], unknowns: []},
  facts: [{id: "context_available", label: "Code is available", confirmed: true}], actions: [], interpretation: {},
};
const recommendation = model => ({status: "recommended", choice: {id: "understand", name: `${model} next step`, description: "Inspect the evidence."},
  source: {reader: "Qwen local", decision: model}, assumptions: [], actions: [], timing: {decision_ms: 1}, limitation: "Suggestions only."});

function setup(ready, query = "") {
  const elements = {}, calls = [];
  let pendingDecision, failDecision = false;
  const get = id => elements[id] ||= new Element();
  get("decision-model").value = "v4";
  get("model-4b-option").disabled = true;
  get("situation-input").value = "I have source code for a failing test.";
  get("goal-input").value = "Identify the cause.";
  get("recommendation-panel").hidden = true;
  const document = {getElementById: get, createElement: tag => new Element(tag), querySelectorAll: () => []};
  const context = {document, Intl, URLSearchParams, location: {search: query}, fetch: async (url, options) => {
    calls.push({url, options});
    if (url === "/api/scale/model") return {ok: true, json: async () => ({ready, label: "Our 4.25B <safe literal>"})};
    assert.equal(options.method, "POST");
    if (url === "/api/situation/interpret") return {ok: true, json: async () => readyDraft};
    assert.ok(["/api/situation/decide", "/api/situation/decide4b"].includes(url));
    if (pendingDecision) return pendingDecision;
    if (failDecision) return {ok: false, status: 503, json: async () => ({detail: "The model is unavailable."})};
    return {ok: true, json: async () => recommendation(url.endsWith("4b") ? "Our 4.25B" : "V4")};
  }};
  vm.runInNewContext(source, context);
  return {get, calls, setPending(value) { pendingDecision = value; }, fail() { failDecision = true; },
    async event(id, type) { await get(id).listeners[type]({preventDefault() {}}); }};
}

(async () => {
  const view = setup(true); await flush();
  assert.equal(view.get("decision-model").value, "v4", "availability must not switch the default");
  assert.equal(view.get("model-4b-option").disabled, false);
  assert.equal(view.get("model-4b-option").textContent, "Our 4.25B <safe literal>");
  await view.event("situation-form", "submit");
  await view.event("decision-form", "submit");
  assert.equal(view.calls.at(-1).url, "/api/situation/decide");
  assert.match(view.get("recommendation-source").textContent, /Action chosen by V4/);
  assert.equal(view.get("recommendation-panel").hidden, false);
  view.get("decision-model").value = "4b";
  await view.event("decision-model", "change");
  assert.equal(view.get("recommendation-panel").hidden, true);
  await view.event("decision-form", "submit");
  assert.equal(view.calls.at(-1).url, "/api/situation/decide4b");
  assert.deepEqual(JSON.parse(view.calls.at(-1).options.body).confirmed_facts, ["context_available"]);
  assert.match(view.get("recommendation-source").textContent, /Action chosen by Our 4.25B/);

  let finish;
  view.setPending(new Promise(resolve => { finish = resolve; }));
  const inFlight = view.event("decision-form", "submit");
  assert.equal(view.get("decision-model").disabled, true, "selection locks during inference");
  // A programmatic revision still cannot display a late response under another selection.
  view.get("decision-model").value = "v4";
  await view.event("decision-model", "change");
  finish({ok: true, json: async () => recommendation("Late 4B")});
  await inFlight;
  assert.equal(view.get("recommendation-panel").hidden, true);
  view.setPending(undefined); view.fail();
  view.get("decision-model").value = "4b";
  await view.event("decision-model", "change");
  const before = view.calls.length;
  await view.event("decision-form", "submit");
  assert.equal(view.calls.length, before + 1, "failed 4B must not silently request V4");
  assert.equal(view.calls.at(-1).url, "/api/situation/decide4b");
  assert.equal(view.get("situation-error").textContent, "The model is unavailable.");
  assert.equal(view.get("recommendation-panel").hidden, true);

  const unavailable = setup(false); await flush();
  assert.equal(unavailable.get("model-4b-option").disabled, true);
  assert.equal(unavailable.get("decision-model").value, "v4");
  await unavailable.event("situation-form", "submit");
  await unavailable.event("decision-form", "submit");
  assert.equal(unavailable.calls.at(-1).url, "/api/situation/decide");
  const linked = setup(true, "?model=4b"); await flush();
  assert.equal(linked.get("decision-model").value, "4b");
  await linked.event("situation-form", "submit");
  await linked.event("decision-form", "submit");
  assert.equal(linked.calls.at(-1).url, "/api/situation/decide4b");
  const notReadyLinked = setup(false, "?model=4b"); await flush();
  assert.equal(notReadyLinked.get("decision-model").value, "v4");
  assert.equal(notReadyLinked.get("model-4b-option").disabled, true);
  const manualChoice = setup(true, "?model=4b");
  manualChoice.get("decision-model").value = "v4";
  await manualChoice.event("decision-model", "change");
  await flush();
  assert.equal(manualChoice.get("decision-model").value, "v4", "late availability must preserve an explicit user choice");
  console.log("Model-selector DOM checks passed: default V4, ready-only 4B deep link, explicit endpoint, disabled unavailable option, no fallback, source attribution and stale-response protection.");
})().catch(error => { console.error(error); process.exitCode = 1; });
