"""The application's icons, held in the application.

`dash_iconify` draws an icon by name and fetches its SVG from the Iconify web
API the first time. The application operates on your computer only, so that
fetch fails when the computer has no internet, and every icon is then blank.
`assets/icons.js` gives Iconify the icons before it asks: it sets
`window.IconifyPreload`, which Iconify reads when it loads, to the Tabler
icons the application uses and no others.

`used()` finds those names in the application's code: every `"tabler:..."`
string, the first argument of every `icon(...)` call, and the values of every
`ICONS` table. `test_app.py` fails, by name, for an icon that the code uses
and `icons.js` does not hold. To add icons, write them again from the Tabler
set of Iconify (the npm package `@iconify-json/tabler`, MIT licence):

    npm pack @iconify-json/tabler && tar xzf iconify-json-tabler-*.tgz
    python -m atisim.apps.icons package/icons.json
"""

import ast
import json
import re
import sys
from pathlib import Path

APPS = Path(__file__).resolve().parent
ICONS_JS = APPS / "assets" / "icons.js"
_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def used(directory: Path = APPS, known: set | None = None) -> set[str]:
    """Every Tabler icon name the application's code draws.

    With `known` (every name in the icon set), also every string in the code
    that is an icon name, which finds the icons named in tables of tuples, such
    as Setup's tree. `test_app.py` also collects the icons of the drawn pages."""
    names = set()
    for path in sorted(Path(directory).glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and known is not None and node.value in known:
                names.add(node.value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.startswith("tabler:"):
                names.add(node.value.split(":", 1)[1])
            elif isinstance(node, ast.Call) and _call_name(node) == "icon" and node.args \
                    and isinstance(node.args[0], ast.Constant):
                names.add(str(node.args[0].value).removeprefix("tabler:"))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(
                    isinstance(t, ast.Name) and t.id == "ICONS" for t in node.targets):
                names.update(str(v.value).removeprefix("tabler:") for v in node.value.values
                             if isinstance(v, ast.Constant))
    return {n for n in names if _NAME.fullmatch(n)}


def held(path: Path = ICONS_JS) -> set[str]:
    """The icon names `icons.js` holds."""
    text = Path(path).read_text()
    body = text[text.index("=") + 1:].strip().rstrip(";")
    return set(json.loads(body)["icons"])


def write(icon_set: dict, names, path: Path = ICONS_JS) -> None:
    """Write `icons.js` with `names` from an Iconify JSON icon set."""
    icons, missing = {}, []
    for name in sorted(names):
        entry = icon_set["icons"].get(name)
        if entry is None and name in icon_set.get("aliases", {}):
            alias = icon_set["aliases"][name]
            entry = {**icon_set["icons"][alias["parent"]],
                     **{k: v for k, v in alias.items() if k != "parent"}}
        if entry is None:
            missing.append(name)
        else:
            icons[name] = entry
    if missing:
        raise KeyError(f"not in the {icon_set['prefix']} set: {', '.join(missing)}")
    data = {"prefix": icon_set["prefix"], "icons": icons,
            "width": icon_set.get("width", 24), "height": icon_set.get("height", 24)}
    Path(path).write_text(
        "// The Tabler icons the application draws, so that it draws them with no\n"
        "// internet. Tabler Icons, MIT licence, https://tabler.io/icons, from the\n"
        "// Iconify set @iconify-json/tabler. Written by `python -m atisim.apps.icons`;\n"
        "// do not edit by hand.\n"
        "window.IconifyPreload = " + json.dumps(data, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    icon_set = json.loads(Path(sys.argv[1]).read_text())
    names = used(known=set(icon_set["icons"]) | set(icon_set.get("aliases", {})))
    write(icon_set, names)
    print(f"wrote {len(names)} icons to {ICONS_JS}")
