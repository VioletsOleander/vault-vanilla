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
            cmd, shell=True, cwd=self.repo_root, capture_output=True, text=True
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout.strip()

    def _get_repo_root(self):
        try:
            return subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except:
            return None

    def _get_staged_files(self):
        output = self._run_cmd(["git", "diff", "--cached", "--name-only"])
        return [f.strip() for f in output.splitlines() if f.strip()]

    def _is_note_file(self, file_path):
        return re.match(r"^.+-notes/.+\.md$", file_path) is not None

    def _parse_note_info(self, file_path):
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
            r"^(.+)-notes/(?:(.+)/)?([^/]+?)(?:-\d{4}(?:-[A-Z]+)?)?\.md$", file_path
        )
        if not match:
            return None

        note_type, full_category_path, title = match.groups()

        category = full_category_path if full_category_path is not None else ""

        return {
            "type": note_type,
            "category": category,
            "title": title,
            "full_title": (
                f"{category}/{'/' if category else ''}{title}" if category else title
            ),
        }

    def _is_file_committed_before(self, file_path):
        print(file_path)
        try:
            self._run_cmd(["git", "ls-files", "--error-unmatch", file_path], check=True)
            print("return true")
            return True
        except:
            print("return false")
            return False

    def _generate_commit_message(self, note_files):
        """Generate commit message from the first note file."""
        for file_path in note_files:
            info = self._parse_note_info(file_path)
            print(info)
            if not info:
                continue

            status = "update" if self._is_file_committed_before(file_path) else "add"
            return f"note({info['type']}): {status} '{info['full_title']}'"

        return None  # No valid note file found

    def run(self):
        staged_files = self._get_staged_files()
        if not staged_files:
            print("No staged files found.")
            return

        note_files = [f for f in staged_files if self._is_note_file(f)]
        if not note_files:
            print("No note files (e.g., *-notes/) staged. Nothing to do.")
            return

        commit_msg = self._generate_commit_message(note_files)
        if not commit_msg.strip():
            print("Could not generate commit message.")
            return

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
        commit_item.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
