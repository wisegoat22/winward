// Pure DOM logic test; no browser, GPU, external network or model loading.
const fs = require("fs"), vm = require("vm"), assert = require("assert/strict"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "../jev_local/static/scale.html"), "utf8");
const ids = new Set(Array.from(html.matchAll(/id="([^"]+)"/g), match => match[1]));
class Element {
  constructor() { this.children = []; this.dataset = {}; this.hidden = false; this.textContent = ""; }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children = values; }
  set innerHTML(_) { throw Error("Audit content must be literal text"); }
}
const elements = {};
const document = {hidden: false, getElementById: id => { assert.ok(ids.has(id), `Missing actual HTML id: ${id}`); return elements[id] ||= new Element(); }, createElement: () => new Element(), addEventListener() {}};
let data = {tracks: [], runs: [], latest_run: null, limitations: [], warnings: [], final_audit: {status: "pending", note: "Final audit not published."}};
let tick;
const context = {document, Date, fetch: async url => { assert.equal(url, "/api/scale/status"); return {ok: true, json: async () => data}; },
  clearTimeout() {}, setTimeout(callback) { tick = callback; return 1; }};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../jev_local/static/scale.js"), "utf8"), context);
const flush = () => new Promise(setImmediate);
(async () => {
  await flush();
  assert.equal(elements["audit-title"].textContent, "Final audit pending");
  assert.equal(elements["audit-results"].hidden, true);
  data.final_audit = {status: "completed", note: "No automatic promotion.", cases: 512, rollout_cases: 40,
    next_action: {candidate_accuracy: 480 / 512, baseline_accuracy: 470 / 512, candidate_correct: 480, baseline_correct: 470},
    retention: {passed_checks: 7, total_checks: 8, all_measured_retention_checks_passed: false},
    rollouts: {candidate_success: .87, baseline_success: .85, reference_success: .97, denominator: 28},
    timing: {candidate_ms: 6.4, baseline_ms: .12, scope: "Warmed batch8, amortized ms per decision"},
    model_sha256: "a".repeat(64), protocol_sha256: "b".repeat(64), scope: "Controlled synthetic <literal scope>."};
  tick(); await flush();
  assert.equal(elements["audit-results"].hidden, false);
  assert.equal(elements["audit-candidate-score"].textContent, "93.75%");
  assert.equal(elements["audit-baseline-score"].textContent, "91.80%");
  assert.equal(elements["audit-retention-score"].textContent, "7 / 8");
  assert.match(elements["audit-action-note"].textContent, /4\.25B 480; original v4 470/);
  assert.match(elements["audit-action-note"].textContent, /checks failed/);
  assert.equal(elements["audit-candidate-completion"].textContent, "87.00%");
  assert.match(elements["audit-rollout-note"].textContent, /28 initially unfinished/);
  assert.match(elements["audit-rollout-note"].textContent, /not observed real-tool success counts/);
  assert.match(elements["audit-timing"].textContent, /6\.40 ms; original v4 0\.12 ms/);
  assert.match(elements["audit-timing"].textContent, /Warmed batch8, amortized ms per decision/);
  assert.match(elements["audit-source"].textContent, /aaaaaaaaaaaa… · audit protocol bbbbbbbbbbbb…/);
  assert.equal(elements["audit-scope"].textContent, "Controlled synthetic <literal scope>.");
  assert.ok(html.includes('href="/try?model=4b"'));
  data.final_audit = {status: "invalid", note: "The audit cannot be validated."};
  tick(); await flush();
  assert.equal(elements["audit-title"].textContent, "Final audit unavailable");
  assert.equal(elements["audit-results"].hidden, true, "A bad replacement must hide previous scores");
  assert.equal(elements["scale-error"].hidden, true);
  console.log("Audit DOM checks passed: pending/completed/invalid, exact held-out counts, expected-probability semantics, timing scope, frozen hashes and no promotion.");
})().catch(error => { console.error(error); process.exitCode = 1; });
