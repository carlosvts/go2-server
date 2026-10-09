"""Gera `catalog.yaml` a partir da descoberta nos dois repositórios.

Entradas (pasta `discovery/`):

    capabilities.json   resposta de `GET /capabilities` da go2-api
    commands.json       `config/commands.json` do go2-tvbox
    phrases.json        `config/phrases.json` do go2-tvbox
    inferred.yaml       a parte escrita à mão (durações, sinônimos, IDs)

Para atualizar a descoberta a partir dos repositórios e da API no ar:

    uv run python scripts/build_catalog.py \
        --tvbox ../go2-tvbox --capabilities http://127.0.0.1:8000/capabilities

Sem argumentos, só regenera o catálogo com o que já está em `discovery/`. O
script falha (código 1) se o edge usar um comando que a API não tem ou que
diverge dela, e imprime o resumo da reconciliação.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DISCOVERY = ROOT / "discovery"
INFERRED_TAG = "# INFERIDO"
REVIEW_TAG = "# INFERIDO, REVISAR"


def refresh(tvbox: Path | None, capabilities: str | None) -> None:
    if tvbox is not None:
        for name in ("commands.json", "phrases.json"):
            shutil.copy(tvbox / "config" / name, DISCOVERY / name)
    if capabilities is not None:
        if capabilities.startswith("http"):
            with urllib.request.urlopen(capabilities, timeout=5) as response:
                data = json.load(response)
        else:
            data = json.loads(Path(capabilities).read_text(encoding="utf-8"))
        (DISCOVERY / "capabilities.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def flow(value: object) -> str:
    """Valor em YAML de uma linha."""
    text = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=10_000, sort_keys=False
    )
    # Para um escalar solto o PyYAML acrescenta o marcador de fim de documento.
    return text.removesuffix("...\n").strip()


def call_of(capability: dict) -> dict:
    call = {"method": capability["method"], "endpoint": capability["endpoint"]}
    if capability["params"] == ["cmd"]:
        call["body"] = {"cmd": capability["name"]}
    return call


def build() -> tuple[str, list[str], list[str]]:
    capabilities = json.loads((DISCOVERY / "capabilities.json").read_text(encoding="utf-8"))
    edge = json.loads((DISCOVERY / "commands.json").read_text(encoding="utf-8"))
    phrases = json.loads((DISCOVERY / "phrases.json").read_text(encoding="utf-8"))
    inferred = yaml.safe_load((DISCOVERY / "inferred.yaml").read_text(encoding="utf-8"))

    api = {entry["name"]: entry for entry in capabilities["commands"]}
    problems: list[str] = []
    notes: list[str] = []

    # 1. Todo comando que o edge usa tem de existir na API, igual.
    for name, entry in edge["commands"].items():
        if name not in api:
            problems.append(f"edge usa {name!r}, que a go2-api não tem")
        elif any(entry[key] != api[name][key] for key in ("method", "endpoint", "params")):
            problems.append(f"{name!r} diverge entre o edge e a go2-api")

    stop_name = phrases["parada"]["comando"]
    move_name = phrases["movimentos"]["comando"]
    spoken = set(phrases["acoes"].values()) | {stop_name, move_name}
    unspoken = [name for name in api if name not in spoken]
    notes.append(f"go2-api: {len(api)} comandos (versão {capabilities['version']})")
    notes.append(f"com frase no edge: {sorted(spoken)}")
    notes.append(f"sem frase no edge: {unspoken}")

    # 2. Ações de verbo único: frases do edge + exemplos inferidos.
    edge_examples: dict[str, list[str]] = {}
    for phrase, name in phrases["acoes"].items():
        edge_examples.setdefault(name, []).append(phrase)

    out: list[str] = [
        "# GERADO por scripts/build_catalog.py. Não edite à mão: o que é inferido",
        "# mora em discovery/inferred.yaml, e o resto vem da go2-api e do go2-tvbox.",
        "#",
        "# Marcas:",
        "#   # edge               frase que o usuário já fala (go2-tvbox, phrases.json)",
        "#   # INFERIDO           escolha deste servidor, sem respaldo nos repositórios",
        "#   # INFERIDO, REVISAR  idem, e precisa de conferência com o robô",
        "",
        "version: 1",
        f"go2_api_capabilities_version: {flow(capabilities['version'])}",
        "",
        "stop:",
        f"  id: {stop_name}",
        f"  call: {flow(call_of(api[stop_name]))}",
        f"  words: {flow(phrases['parada']['palavras'])}  # edge",
        "",
        "commands:",
    ]

    def emit_command(name: str, examples: list[tuple[str, str]], enabled: bool) -> None:
        if name not in api:
            problems.append(f"{name!r} está no inferred.yaml e não existe na go2-api")
            return
        if name not in inferred["durations"]:
            problems.append(f"{name!r} sem duração em inferred.yaml")
            return
        match = "lexical_only" if name in inferred["lexical_only"] else "embedding"
        out.append(f"  {name}:")
        out.append(f"    enabled: {flow(enabled)}" + ("" if enabled else f"  {REVIEW_TAG}"))
        out.append(f"    match: {match}" + (f"  {INFERRED_TAG}" if match != "embedding" else ""))
        out.append(f"    sport_cmd: {api[name]['sport_cmd']}")
        out.append(f"    call: {flow(call_of(api[name]))}")
        out.append(f"    max_duration_s: {inferred['durations'][name]}  {REVIEW_TAG}")
        out.append("    examples:")
        for text, origin in examples:
            out.append(f"      - {flow(text)}  {origin}")

    for name, texts in edge_examples.items():
        examples = [(text, "# edge") for text in texts]
        examples += [(text, INFERRED_TAG) for text in inferred["extra_examples"].get(name, [])]
        emit_command(name, examples, enabled=True)
    for name, texts in inferred["disabled"].items():
        if name in spoken:
            problems.append(f"{name!r} tem frase no edge e está em `disabled`")
            continue
        emit_command(name, [(text, INFERRED_TAG) for text in texts], enabled=False)

    not_in_catalog = [n for n in unspoken if n not in inferred["disabled"]]
    notes.append(f"fora do catálogo (exigem parâmetro que a voz não dá): {not_in_catalog}")

    # 3. Movimento por slots: verbo × direção → args do `move`.
    movement = inferred["movement"]
    moves = phrases["movimentos"]
    defaults = moves["padrao"]
    if sorted(defaults) != sorted(api[move_name]["params"]):
        problems.append("os campos do `move` no edge não batem com os da go2-api")

    out += [
        "",
        "movement:",
        f"  call: {flow(call_of(api[move_name]))}",
        f"  params: {flow(api[move_name]['params'])}",
        f"  prepare: {flow(movement['prepare'])}  {REVIEW_TAG}",
        f"  settle_s: {movement['settle_s']}  {REVIEW_TAG}",
        f"  ignore: {flow(movement['ignore'])}  {INFERRED_TAG}",
        "  verbs:",
    ]
    for verb in moves["verbos"]:
        words = [verb, *movement["verb_synonyms"].get(verb, [])]
        out.append(f"    {verb}: {flow(words)}  # 1ª do edge, demais INFERIDAS")
    out.append("  directions:")
    for direction in phrases["direcoes"]:
        words = [direction, *movement["direction_synonyms"].get(direction, [])]
        out.append(f"    {direction}: {flow(words)}  # 1ª do edge, demais INFERIDAS")
    out.append("  # Velocidades e duração copiadas do edge. O `id` é INFERIDO.")
    out.append("  rules:")
    for verb, directions in moves["verbos"].items():
        for direction, overrides in directions.items():
            rule_id = movement["ids"].get(verb, {}).get(direction)
            if rule_id is None:
                problems.append(f"sem id para {verb} + {direction} em inferred.yaml")
                continue
            if rule_id in api:
                problems.append(f"id de movimento {rule_id!r} colide com comando da API")
            args = {**defaults, **overrides}
            # O commands.json é gerado do phrases.json: os dois têm de concordar.
            for phrase, target in edge["phrases"].items():
                words = phrase.split()
                if not isinstance(target, dict):
                    continue
                if words[0] == verb and words[-1] == direction and target["args"] != args:
                    problems.append(f"{phrase!r}: args diferentes no commands.json")
            rule = {"id": rule_id, "verb": verb, "direction": direction, "args": args}
            out.append(f"    - {flow(rule)}")

    language = inferred["language"]
    out += [
        "",
        f"language:  {INFERRED_TAG}",
        f"  negation: {flow(language['negation'])}",
        f"  separators: {flow(language['separators'])}",
        f"  replacements: {flow(language['replacements'])}",
        f"  fillers: {flow(language['fillers'])}",
        "",
    ]
    return "\n".join(out), problems, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tvbox", type=Path, help="clone do go2-tvbox, para recopiar os JSON")
    parser.add_argument("--capabilities", help="URL ou arquivo com o GET /capabilities")
    parser.add_argument("--output", type=Path, default=ROOT / "catalog.yaml")
    args = parser.parse_args()

    refresh(args.tvbox, args.capabilities)
    text, problems, notes = build()
    for note in notes:
        print(note)
    if problems:
        print("\nDIVERGÊNCIAS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    args.output.write_text(text, encoding="utf-8")
    print(f"\nOK: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
