# Try a situation on your Mac

With the server running, open **http://127.0.0.1:8765/try**. This interface supports software-agent tasks: fixing bugs, adding features, refactoring, investigating, reviewing code, updating documentation, and running existing checks.

1. Enter what is happening, what evidence you have, and what you already tried. Add a concrete **desired win**.
2. Select **Read my situation**. Local Qwen reads the text. Review the task type, quoted facts, and constraints. Check only facts that are already true; future intentions do not count. Uncheck a constraint if the reader misunderstood you.
3. Select **Get my next step**. Winward chooses one action from the supplied workflow. Read the assumptions and what to check next.
4. Take the action yourself. Add its observed result to your situation, then repeat. A suggestion does not mean the action succeeded.

For example:

> Situation: The checkout test fails on an empty cart. I reproduced the failure and have the source code and test output, but I do not know the cause.
>
> Desired win: Make the empty-cart test pass while preserving the expected behavior.

Try the three example buttons as well. Watch how the checked facts change between an unknown bug, an applied fix awaiting validation, and completed work. You can correct the same draft and ask again to see how a different confirmed state changes the model's choice.

## What the result means

Qwen3 4B is an optional, pretrained language reader. Winward v4 is our own trained **4,862,721-parameter** decision policy. Qwen extracts task type, quoted evidence, constraints, and missing information; it does not generate workflow transitions or select the action. Code supplies a small fixed workflow, and the frozen Winward model ranks its permitted actions. Its choice is returned directly, including a deferral. No search answer is silently substituted.

The workflow assumes that an action establishes its intended result. It does not model hidden causes or failures in this interface. The five-action budget is supplied to the policy; this is not a fresh five-level search through consequences of arbitrary prose. Costs are relative defaults, not measured token bills or predictions of how long your real task will take. Displayed reading and decision times measure the local requests.

The app does not inspect a repository or run tools. Completion means **your confirmed facts** say the workflow milestone is satisfied. Check that the milestone actually covers your win. A model deferral can reflect missing information, constraints, or a model limitation; it does not prove the task is impossible.

The free-text adapter is new and outside the frozen v4 benchmark. V4 failed five of its twelve promotion checks. Results on synthetic or controlled coding tasks do not establish reliability on your situation or on arbitrary decisions.

## Setup and privacy

The interface needs installed Qwen weights and a trained v4 checkpoint. On a fresh checkout, follow [the README](../README.md) for dependencies and [the v4 guide](V4.md#reproduce-on-mac) to train its checkpoint. Download the optional text reader with `.venv/bin/python scripts/download_model.py`, then start the server with `.venv/bin/python scripts/server.py start`.

Dependencies and the initial model download require internet. Inference uses installed local weights. This interface does not save your situation, create a history, use browser local storage, or send prompts to an external service. Refreshing the page clears the form. If another model request is in progress, wait for it to finish before trying again.

The text reader allows one repair attempt when its draft fails validation, then returns a visible error. Correctly quoted evidence can still be misunderstood, so review the form before requesting a decision. If interpretation fails, shorten or rephrase the situation. Unsupported domains do not receive a Winward decision.

Live checks found both omitted facts and an incorrect inference that a failure had been reproduced merely because source code was available. The quoted evidence and editable checkboxes expose these mistakes for correction; passing format validation does not establish that the interpretation is correct.
