import argparse
import os
import re
import subprocess
import sys
import tempfile


class GitCommitItem:
    def __init__(self):
        self.repo_root = self._get_repo_root()
        if not self.repo_root:
            raise EnvironmentError("Not a git repository or git not found.")

    def _run_cmd(self, cmd, check=True):
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout.strip()

    def _get_repo_root(self):
        return subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _get_staged_files(self):
        output = self._run_cmd(["git", "diff", "--cached", "--name-only"])
        return [f.strip() for f in output.splitlines() if f.strip()]

    def _is_note_file(self, file_path):
        return re.match(r"^.+-notes/.+\.md$", file_path) is not None

    def _parse_note_info(self, file_path) -> dict:
        """
        Parse note type, multi-level category, and title.
        Supports:
        *-notes/category/title.md
        *-notes/cat1/cat2/.../title.md
        *-notes/title.md
        *-notes/.../title-2022.md
        *-notes/.../title-2022-OSDI.md

        Examples:
        paper-notes/mlsys/distributed/gpu/Orca Paper-2022-OSDI.md
            -> type: 'paper'
            -> category: 'mlsys/distributed/gpu'
            -> title: 'Orca Paper'

        paper-notes/Orca Paper-2022-OSDI.md
            -> type: 'paper'
            -> category: ''
            -> title: 'Orca Paper'
        """
        # Pattern breakdown:
        # ^(.+)-notes/          --> note type (e.g., paper, tech, etc.)
        # (.+?)/                --> optional category path (with / at end), non-greedy
        # ([^/]+?)              --> title (without path separators)
        # (?:-\d{4}(?:-[A-Z]+)?)?  --> optional -YYYY or -YYYY-CONF
        # \.md$                 --> ends with .md
        #
        # We make the category part fully optional using (?:.+/)?
        match = re.match(
            r"^(.+)-notes/(?:(.+)/)?([^/]+?)(?:-\d{4}(?:-[A-Za-z]+)?)?\.md$", file_path
        )
        if not match:
            raise ValueError(f"Invalid note file path: {file_path}")

        note_type, full_category_path, title = match.groups()

        category = full_category_path if full_category_path is not None else ""

        return {
            "type": note_type,
            "category": category,
            "title": title,
            "full_title": (f"{category}/{title}" if category else title),
        }

    def _get_all_committed_files(self):
        """
        Acquire all committed file paths in the current HEAD commit.
        """
        result = self._run_cmd(["git", "ls-tree", "-r", "HEAD", "--name-only"])

        files = set()
        for line in result.splitlines():
            if not line:
                continue
            files.add(line)

        return files

    def _is_file_committed_before(self, file_path: str) -> bool:
        committed_files = self._get_all_committed_files()
        return file_path in committed_files

    def _generate_commit_message(self, note_file):
        """Generate commit message from the note file."""
        info = self._parse_note_info(note_file)
        print(info)

        if self.args.status is None:
            status = "update" if self._is_file_committed_before(note_file) else "add"
        else:
            status = self.args.status

        return f"note({info['type']}): {status} '{info['full_title']}'"

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Git commit item")
        parser.add_argument(
            "--status",
            "-s",
            type=str,
            help="add or update or let the script decide",
            default=None,
        )
        self.args = parser.parse_args()
        assert self.args.status in (
            None,
            "add",
            "update",
        ), "status must be one of these: add, update"

    def run(self):
        staged_files = self._get_staged_files()
        if not staged_files:
            print("No staged files found.")
            return

        note_file = next((f for f in staged_files if self._is_note_file(f)), None)
        if not note_file:
            print("No note files (e.g., *-notes/) staged. Nothing to do.")
            return

        commit_msg = self._generate_commit_message(note_file)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmpfile:
            tmpfile.write(commit_msg + "\n")
            temp_path = tmpfile.name

        try:
            print("Generated commit message:")
            print("-" * 60)
            print(commit_msg)
            print("-" * 60)

            cmd = ["git", "commit", "-F", temp_path, "-e"]
            subprocess.run(cmd, cwd=self.repo_root, check=True)
        except subprocess.CalledProcessError:
            print("Git commit failed or was aborted by the user.")
        except Exception as e:
            print(f"Error during commit: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass


if __name__ == "__main__":
    try:
        commit_item = GitCommitItem()
        commit_item.parse_args()
        commit_item.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
