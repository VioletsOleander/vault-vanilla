import re
import pprint


def read_file(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def write_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def merge(lines: list[str]) -> dict[str, dict[str, list[str]]]:
    sections = ['Book', 'Paper', 'Doc', 'Blog', 'Code']
    section_re = re.compile(r'^\\\[\w+\\\]')
    items = []
    item_re = re.compile(r'^- .+:\s')
    code_item_re = re.compile(r'^- .+')

    lines_tree = {section: {} for section in sections}

    curr_section = None
    curr_item = None

    for line in lines:
        # match section header
        if m := re.match(section_re, line):
            # strip '\[' and '\]'
            curr_section = m.group()[2:-2]
        # match item header
        elif m := re.match(item_re, line):
            # strip '- ' and '[' and ']' and '[[' and ']]'
            item = m.group().strip().strip('-').strip(':').strip()
            # item = item.strip('[').strip(']')
            if item not in lines_tree[curr_section]:
                lines_tree[curr_section][item] = []
                items.append(item)
            curr_item = item
        # match code item header
        elif m := re.match(code_item_re, line):
            item = m.group().strip('-').strip()
            if item not in lines_tree[curr_section]:
                lines_tree[curr_section][item] = []
                items.append(item)
            curr_item = item
        # match date line
        elif m := re.match(r'^Date: \d+', line):
            continue
        # match empty line
        elif m := re.match(r'^\s+$', line):
            continue
        # match heading line
        elif m := re.match(r'^#+ \w+', line):
            continue
        # match content line
        else:
            lines_tree[curr_section][curr_item].append(line)

    return lines_tree


def generate_changlog(lines_tree: dict[str, dict[str, list[str]]]) -> str:
    changelog = ''

    changelog += '# Overview\n'
    for section, items in lines_tree.items():
        changelog += f'## {section}\n'
        for item in items.keys():
            changelog += f'- {item}\n'

    changelog += '# Detail\n'
    for section, items in lines_tree.items():
        changelog += f'## {section}\n'
        for item, lines in items.items():
            changelog += f'- {item}\n'
            for line in lines:
                changelog += f'  {line}'

    return changelog


if __name__ == "__main__":
    lines = read_file('logs/Personal log.md')
    lines_tree = merge(lines)
    changlog = generate_changlog(lines_tree)
    write_file('ChangeLog.md', changlog)
    print('ChangeLog.md generated')
