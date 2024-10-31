import datetime
import sys
import os


def parse_arg() -> bool:
    print(sys.argv)
    if len(sys.argv) > 2:
        print("Too many arguments, abort!")
        sys.exit(1)
    elif len(sys.argv) == 1:
        return False
    else:
        arg = sys.argv[-1]
        if arg == 'late':
            return True
        else:
            print("Illeagle argument, abort!")
            sys.exit(1)


def generate_commit_message(late: bool):
    day = datetime.datetime.today()
    if late:
        day = day - datetime.timedelta(days=1)

    date_str = day.strftime("%Y-%m-%d")
    weekday_str = day.strftime("%A")

    headline = f"log(daily): {date_str} {weekday_str}\n\n"

    return headline


if __name__ == "__main__":
    late = parse_arg()
    headline = generate_commit_message(late)
    os.system(f'git commit -e -m "{headline}"')
