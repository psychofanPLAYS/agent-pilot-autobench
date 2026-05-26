from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message


@task
def json_repair():
    return Task(
        dataset=[
            Sample(
                input=('Return exactly this JSON object and no prose: {"ok":true,"items":[1,2,3]}'),
                target='{"ok":true,"items":[1,2,3]}',
            ),
            Sample(
                input='Repair this JSON and return only valid compact JSON: {"tool":"search","args":["qwen",]}',
                target='{"tool":"search","args":["qwen"]}',
            ),
            Sample(
                input=(
                    "Return a compact JSON command object with action set to summarize "
                    "and safe set to true. No Markdown."
                ),
                target='{"action":"summarize","safe":true}',
            ),
        ],
        solver=[
            system_message("You repair or produce compact JSON. Return JSON only."),
            generate(),
        ],
        scorer=match(location="exact", ignore_case=False),
    )
