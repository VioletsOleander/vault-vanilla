import argparse


def setup_parser():
    parser = argparse.ArgumentParser(description="Prepare commit message.")
    parser.add_argument(
        "source", type=str, help="Prespecified commit message source.", default=""
    )
    return parser


def get_content_after_heading(path, heading):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    content_lines = []

    for line in reversed(lines):
        if line.strip() == heading:
            break
        else:
            content_lines.append("# " + line.replace("**", ""))

    return reversed(content_lines)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.source:
        exit(0)

    path = "D:\\data\\vault-vanilla\\casual-notes\\Commit Message Guide.md"
    heading = "## Commit Message Guide for Obsidian Vault"
    content_lines = get_content_after_heading(path, heading)

    with open(".git/COMMIT_EDITMSG", "a", encoding="utf-8", newline="\n") as file:
        file.writelines(content_lines)
