import pprint
import re
from pathlib import Path


def read_file(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines


def write_file(file_path: Path, content: str):
    with file_path.open("w", encoding="utf-8") as f:
        f.write(content)


def merge(lines: list[str]) -> dict[str, dict[str, list[str]]]:
    sections = ["Book", "Paper", "Doc", "Blog", "Code", "OpenReview", "Wiki"]
    section_re = re.compile(r"^\\\[\w+\\\]")
    items = []
    item_re = re.compile(r"^- \[\[.+\]\]:?")
    code_item_re = re.compile(r"^- .+")

    lines_tree = {section: {} for section in sections}

    for line in lines:
        # match section header
        if m := re.match(section_re, line):
            # strip '\[' and '\]'
            curr_section = m.group()[2:-2]
        # match item header
        elif m := re.match(item_re, line):
            # strip '- ' and '[' and ']' and '[[' and ']]'
            item = m.group().strip().strip("-").strip(":").strip()
            # item = item.strip('[').strip(']')
            if item not in lines_tree[curr_section]:
                lines_tree[curr_section][item] = []
                items.append(item)
            curr_item = item
        # match code item header
        elif m := re.match(code_item_re, line):
            item = m.group().strip("-").strip()
            if item not in lines_tree[curr_section]:
                lines_tree[curr_section][item] = []
                items.append(item)
            curr_item = item
        # match date line
        elif m := re.match(r"^Date: \d+", line):
            continue
        # match empty line
        elif m := re.match(r"^\s+$", line):
            continue
        # match heading line
        elif m := re.match(r"^#+ \w+", line):
            continue
        # match content line
        else:
            lines_tree[curr_section][curr_item].append(line)

    return lines_tree


def generate_changelog(lines_tree: dict[str, dict[str, list[str]]]) -> str:
    changelog = ""

    changelog += "# Overview\n"
    for section, items in lines_tree.items():
        changelog += f"\n## {section}\n"
        for item in items.keys():
            changelog += f"- {item}\n"

    changelog += "\n# Detail\n"
    for section, items in lines_tree.items():
        changelog += f"\n## {section}\n"
        for item, lines in items.items():
            changelog += f"- {item}\n"
            for line in lines:
                changelog += f"  {line}"

    return changelog


if __name__ == "__main__":
    personal_log_path = Path("../logs/Personal log.md")
    changelog_path = Path("../ChangeLog.md")

    lines = read_file(personal_log_path)
    lines_tree = merge(lines)

    changelog = generate_changelog(lines_tree)
    write_file(changelog_path, changelog)
    print("ChangeLog.md generated")
