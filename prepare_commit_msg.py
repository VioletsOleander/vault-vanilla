def get_content_after_heading(path, heading):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    content_lines = []

    for line in reversed(lines):
        if line.strip() == heading:
            break
        else:
            content_lines.append('# ' + line.replace('**', ''))

    return reversed(content_lines)


if __name__ == "__main__":
    path = 'casual-notes/Commit Message Guide.md'
    heading = '## Commit Message Guide for Obsidian Vault'
    content_lines = get_content_after_heading(path, heading)

    with open('.git/COMMIT_EDITMSG', 'a', encoding='utf-8', newline='\n') as file:
        file.writelines(content_lines)
