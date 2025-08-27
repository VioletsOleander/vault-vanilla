import argparse
import re


def setup_parser():
    parser = argparse.ArgumentParser(description="Format log items.")
    parser.add_argument(
        "--no_confirm",
        action="store_true",
        help="Do not ask for confirmation before formatting each item",
    )
    parser.add_argument(
        "--inplace", action="store_true", help="Format the log items in place"
    )

    return parser


def open_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(content)


def ask_confirmation():
    confirm = input(
        "Do you want to format this item? (press enter to confirm, press any other key to refuse): "
    )
    if confirm == "":
        return True
    else:
        return False


def judge_formatted(path_part, display_part, sec_type):
    if sec_type == "Doc":
        return path_part[path_part.find("/") + 1 :] == display_part
    elif sec_type == "Paper":
        return format_item(path_part, sec_type) == display_part


def format_item(path_part, sec_type):
    if sec_type == "Doc":
        return path_part[path_part.find("/") + 1 :]
    elif sec_type == "Paper":
        paper_note_title = path_part.split("/")[-1]
        items = paper_note_title.split("-")

        if items[-1].isdigit():
            year = items[-1]
            display_part = f"{year}-{paper_note_title[:paper_note_title.find(year)]}"
        else:
            year = items[-2]
            publisher = items[-1]
            display_part = (
                f"{year}-{publisher}-{paper_note_title[:paper_note_title.find(year)]}"
            )

        return display_part.rstrip("-")

    else:
        raise ValueError(
            f'Unknown section type: {sec_type}. Expected "Doc" or "Paper".'
        )


def format_line(line, no_confirm, sec_type):
    formatted_line = line

    part1, part2 = line.split("|")
    prefix, path_part = part1.split("[[")
    display_part, suffix = part2.split("]]")

    if judge_formatted(path_part, display_part, sec_type):
        print(f"Log item is already formatted")
    else:
        print(f"Log item is not formatted")
        if no_confirm or ask_confirmation():
            display_part = format_item(path_part, sec_type)
            formatted_line = f"{prefix}[[{path_part}|{display_part}]]{suffix}"
        print(f"The Formatted item is:\n{formatted_line.strip()}")

    return formatted_line


def format_log_item(contents, no_confirm):
    doc_sec_re = re.compile(r"\\\[Doc\\\]\s")
    paper_sec_re = re.compile(r"\\\[Paper\\\]\s")
    item_re = re.compile(r"^- \[\[.+\]\]:?")
    item_content_re = re.compile(r"^    .*")

    line_idx = 0
    while line_idx < len(contents):
        line = contents[line_idx]

        if doc_sec_re.match(line):
            print(f"In Doc section, line {line_idx+1}")
            line_idx += 1

            while line_idx < len(contents):
                if item_re.match(contents[line_idx]):
                    print(f"In line {line_idx+1}")
                    line = contents[line_idx]
                    print(line.strip())

                    contents[line_idx] = format_line(line, no_confirm, "Doc")

                    line_idx += 1
                elif item_content_re.match(contents[line_idx]):
                    line_idx += 1
                else:
                    break

        elif paper_sec_re.match(line):
            print(f"In Paper section, line {line_idx+1}")
            line_idx += 1

            while line_idx < len(contents):
                if item_re.match(contents[line_idx]):
                    print(f"In line {line_idx+1}")
                    print(contents[line_idx].strip())

                    contents[line_idx] = format_line(
                        contents[line_idx], no_confirm, "Paper"
                    )

                    line_idx += 1

                elif item_content_re.match(contents[line_idx]):
                    line_idx += 1
                else:
                    break
        else:
            pass

        line_idx += 1

    return contents


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    log_path = "../logs/Personal Log.md"
    contents = open_file(log_path)

    formatted_contents = format_log_item(contents, args.no_confirm)

    store_path = log_path if args.inplace else "../logs/Formatted Personal Log.md"
    write_file(store_path, formatted_contents)
    print(f"Formatted log items saved to {store_path}")
