import subprocess
import sys


class GitLogDays:
    """The 'git log-days' command"""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.arg_val = 0
        self.hash_list = []

    def parse_arguments(self):
        if len(sys.argv) > 2:
            print("Too many arguments, abort!")
            sys.exit(1)
        elif len(sys.argv) == 2:
            try:
                self.arg_val = int(sys.argv[1])
            except ValueError:
                print("Illegal argument, abort!")
                sys.exit(1)

    def get_commit_hashes(self):
        """get hash list for commits whose message satisfy specified pattern"""

        try:
            result = subprocess.run(
                ['git', 'log', f'--grep={self.pattern}', '--oneline'],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.splitlines()
            if lines:
                self.hash_list = [line.split()[0] for line in lines]
            else:
                print(f"Can not find commit with pattern: {self.pattern}")
        except subprocess.CalledProcessError as e:
            print(f"Git error: {e}")

    def show_recent_commits(self):
        if not self.hash_list:
            print("No commits found to display.")
            return

        try:
            index = self.arg_val if self.arg_val else 0
            subprocess.run(
                ['git', 'log', f'{self.hash_list[index]}..HEAD', '--oneline'])
        except IndexError:
            print(f"Argument too large, abort!")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Git error: {e}")


if __name__ == "__main__":
    pattern = '^log(daily)'
    command = GitLogDays(pattern)
    command.parse_arguments()
    command.get_commit_hashes()
    command.show_recent_commits()
