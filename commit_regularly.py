import datetime
import sys
import subprocess
import argparse


class GitCommitRegularly:
    def __init__(self):
        self.status_list = ['daily', 'weekly', 'monthly']

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Generate daily, weekly, or montly commit message headline")
        parser.add_argument('status', choices=self.status_list,
                            help="specify 'daily' for a daily commit, 'weekly' for a weekly commit, or 'monthly' for a monthly commit.")
        parser.add_argument('-l', '--late', action="store_true",
                            help="for daily commit, if specified, generate headline for the previous day")
        args = parser.parse_args()

        self.args = args
        self.status = args.status
        self.late = args.late

    def generate_headline(self) -> str:
        today = datetime.datetime.today()
        day = today - datetime.timedelta(days=1) if self.late else today

        if self.status == 'daily':
            date_str = day.strftime("%Y-%m-%d")
            weekday_str = day.strftime("%A")

            self.headline = f"log(daily): {date_str} {weekday_str}"
            return

        elif self.status == 'weekly':
            day_of_month = int(day.strftime("%d"))
            week_of_month = (day_of_month) // 7 + 1

            month = day.strftime("%B")
            year = day.strftime("%Y")

            self.headline = f"log(weekly): Week{week_of_month}-of-{
                month} {year}"
            return
        elif self.status == 'monthly':
            today = datetime.datetime.today()
            start_of_month = today.replace(day=1)
            end_of_month = (start_of_month + datetime.timedelta(days=32)
                            ).replace(day=1) - datetime.timedelta(days=1)

            start_date_str = start_of_month.strftime("%Y-%m-%d")
            end_date_str = end_of_month.strftime("%Y-%m-%d")

            self.headline = f"log(monthly): Month {today.month}({
                start_date_str} to {end_date_str}"
            return
        else:
            print("Illegal Argument, abort!")
            sys.exit(1)

    def execute(self):
        self.parse_args()
        self.generate_headline()
        subprocess.run(['git', 'commit', '--allow-empty',
                           '-e', '-m', self.headline])


if __name__ == "__main__":
    command=GitCommitRegularly()
    command.execute()
