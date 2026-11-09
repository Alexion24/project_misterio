import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> tuple[str, dict[tuple[str, int], str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    results = {
        (case["brief"], case["repetition"]): case["artifact"]["markdown"]
        for case in report["cases"]
        if "artifact" in case
    }
    return report["model"], results


def build_review(
    left_path: Path,
    right_path: Path,
    seed: int,
    left_label: str | None = None,
    right_label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_model, left = _load_results(left_path)
    right_model, right = _load_results(right_path)
    left_label = left_label or left_model
    right_label = right_label or right_model
    if left.keys() != right.keys():
        raise ValueError("reports must contain the same successful cases")

    rng = random.Random(seed)
    review = []
    answers = {}
    for index, case_key in enumerate(sorted(left), start=1):
        brief, repetition = case_key
        variants = [(left_label, left[case_key]), (right_label, right[case_key])]
        rng.shuffle(variants)
        case_id = f"case-{index:02d}"
        review.append(
            {
                "id": case_id,
                "brief": brief,
                "repetition": repetition,
                "a": variants[0][1],
                "b": variants[1][1],
                "choice": None,
                "notes": "",
            }
        )
        answers[case_id] = {
            "a_model": variants[0][0],
            "b_model": variants[1][0],
        }
    return review, {"seed": seed, "answers": answers}


def render_markdown(review: list[dict[str, Any]]) -> str:
    sections = [
        "# Слепое сравнение вариантов",
        "",
        "Для каждого кейса отметьте один вариант: `A`, `B`, `Ничья` или "
        "`Оба плохие`. Ключ конфигураций находится в отдельном файле.",
    ]
    for case in review:
        sections.extend(
            [
                "",
                f"## {case['id']} — {case['brief']}",
                "",
                "### Вариант A",
                "",
                case["a"],
                "",
                "### Вариант B",
                "",
                case["b"],
                "",
                "**Выбор:** [ ] A  [ ] B  [ ] Ничья  [ ] Оба плохие",
                "",
                "**Заметка:** ",
                "",
                "---",
            ]
        )
    return "\n".join(sections) + "\n"


def render_html(review: list[dict[str, Any]]) -> str:
    data = json.dumps(review, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Слепое сравнение вариантов</title>
<style>
  :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1180px; margin: auto; padding: 24px; }}
  header {{ display:flex; justify-content:space-between; gap:16px; align-items:end; }}
  nav {{ display:flex; gap:8px; flex-wrap:wrap; margin:20px 0; }}
  button {{ padding:8px 12px; cursor:pointer; }}
  button.active {{ outline:2px solid Highlight; }}
  .variants {{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
    gap:16px;
  }}
  article {{ border:1px solid GrayText; border-radius:8px; padding:16px; }}
  pre {{ white-space:pre-wrap; font:15px/1.55 ui-monospace,monospace; }}
  fieldset {{ margin:20px 0; padding:16px; }}
  label {{ margin-right:18px; white-space:nowrap; }}
  textarea {{ box-sizing:border-box; width:100%; min-height:72px; padding:8px; }}
  footer {{ display:flex; justify-content:space-between; gap:12px; margin-top:16px; }}
  .done {{ color:green; }}
</style>
<body>
<header>
  <div><h1>Слепое сравнение вариантов</h1>
  <p>Скрыто сравниваются 2 и 4 кандидата.</p></div>
  <strong id="progress"></strong>
</header>
<nav id="nav"></nav>
<main>
  <p id="topic"></p>
  <div class="variants">
    <article><h2>Вариант A</h2><pre id="a"></pre></article>
    <article><h2>Вариант B</h2><pre id="b"></pre></article>
  </div>
  <fieldset><legend>Ваш выбор</legend><div id="choices"></div></fieldset>
  <label for="note">Заметка</label>
  <textarea id="note" placeholder="Необязательно"></textarea>
</main>
<footer>
  <button id="prev">Назад</button>
  <button id="export">Скачать ответы JSON</button>
  <button id="next">Далее</button>
</footer>
<script>
const cases={data};
const labels={{a:'Вариант A',b:'Вариант B',tie:'Ничья',neither:'Оба плохие'}};
const byId=id=>document.getElementById(id);
const topic=byId('topic'),a=byId('a'),b=byId('b');
const choices=byId('choices'),note=byId('note'),progress=byId('progress');
const nav=byId('nav'),prev=byId('prev'),next=byId('next');
const exportButton=byId('export');
let index=Number(localStorage.getItem('candidate-review-index')||0);
let reviews=JSON.parse(localStorage.getItem('candidate-review-answers')||'{{}}');
const save=()=>localStorage.setItem('candidate-review-answers',JSON.stringify(reviews));
function render(){{
 const item=cases[index], answer=reviews[item.id]||{{note:''}};
 topic.textContent=`${{item.id}} — ${{item.brief}}`;
 a.textContent=item.a;
 b.textContent=item.b;
 choices.innerHTML=Object.entries(labels).map(([v,l])=>
  `<label><input type="radio" name="choice" value="${{v}}" `+
  `${{answer.choice===v?'checked':''}}> ${{l}}</label>`
 ).join('');
 choices.querySelectorAll('input').forEach(el=>el.onchange=()=>{{reviews[item.id]={{...answer,choice:el.value}};save();render();}});
 note.value=answer.note||'';
 note.oninput=()=>{{
  reviews[item.id]={{...(reviews[item.id]||{{}}),note:note.value}};
  save();
 }};
 const count=cases.filter(c=>reviews[c.id]?.choice).length;
 progress.textContent=`${{count}} / ${{cases.length}} оценено`;
 progress.className=count===cases.length?'done':'';
 nav.innerHTML=cases.map((c,i)=>
  `<button data-i="${{i}}" class="${{i===index?'active':''}}">`+
  `${{i+1}}${{reviews[c.id]?.choice?' ✓':''}}</button>`
 ).join('');
 nav.querySelectorAll('button').forEach(el=>el.onclick=()=>go(Number(el.dataset.i)));
 prev.disabled=index===0; next.disabled=index===cases.length-1;
}}
function go(i){{
 index=i;
 localStorage.setItem('candidate-review-index',String(i));
 render();
}}
prev.onclick=()=>go(index-1); next.onclick=()=>go(index+1);
exportButton.onclick=()=>{{
 const blob=new Blob(
  [JSON.stringify(reviews,null,2)],
  {{type:'application/json'}},
 );
 const link=document.createElement('a');
 link.href=URL.createObjectURL(blob);
 link.download='candidate-count-review-results.json';
 link.click();
 URL.revokeObjectURL(link.href);
}};
render();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("key", type=Path)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    args = parser.parse_args()

    review, key = build_review(
        args.left,
        args.right,
        args.seed,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    if args.review.suffix.casefold() == ".md":
        args.review.write_text(render_markdown(review), encoding="utf-8")
    elif args.review.suffix.casefold() == ".html":
        args.review.write_text(render_html(review), encoding="utf-8")
    else:
        args.review.write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    args.key.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
