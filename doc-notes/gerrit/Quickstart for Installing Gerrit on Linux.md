>  version v3.11.2

This content explains how to install a basic instance of Gerrit on a Linux machine.

|   |   |
|---|---|
|Note|This quickstart is provided for demonstration purposes only. The Gerrit instance they install must not be used in a production environment.<br><br>Instead, to install a Gerrit production environment, see [Standalone Daemon Installation Guide](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/install.html).|

## Before you start
Be sure you have:

1. A Unix-based server, including any Linux flavor, MacOS, or Berkeley Software Distribution (BSD).
2. Java SE Runtime Environment version 11 and up.

## Download Gerrit
From the Linux machine on which you want to install Gerrit:

1. Open a terminal window.
2. Download the desired Gerrit archive.

To view previous archives, see [Gerrit Code Review: Releases](https://gerrit-releases.storage.googleapis.com/index.html). The steps below install Gerrit 3.9.4:

```
wget https://gerrit-releases.storage.googleapis.com/gerrit-3.9.4.war
```

|   |   |
|---|---|
|Note|To build and install Gerrit from the source files, see [Gerrit Code Review: Developer Setup](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/dev-readme.html).|

## Install and initialize Gerrit
From the command line, enter:

```
export GERRIT_SITE=~/gerrit_testsite
java -jar gerrit*.war init --batch --dev -d $GERRIT_SITE
```

This command takes two parameters:

- `--batch` assigns default values to several Gerrit configuration options. To learn more about these options, see [Configuration](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/config-gerrit.html).
- `--dev` configures the Gerrit server to use the authentication option, `DEVELOPMENT_BECOME_ANY_ACCOUNT`, which enables you to switch between different users to explore how Gerrit works. To learn more about setting up Gerrit for development, see [Gerrit Code Review: Developer Setup](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/dev-readme.html).

While this command executes, status messages are displayed in the terminal window. For example:

```
Generating SSH host key ... rsa(simple)... done
Initialized /home/gerrit/gerrit_testsite
Executing /home/gerrit/gerrit_testsite/bin/gerrit.sh start
Starting Gerrit Code Review: OK
```

The last message confirms that the Gerrit service is running:

`Starting Gerrit Code Review: OK`.

## Update the listen URL
To prevent outside connections from contacting your new Gerrit instance (strongly recommended), change the URL on which Gerrit listens from `*` to `localhost`. For example:

```
git config --file $GERRIT_SITE/etc/gerrit.config httpd.listenUrl 'http://localhost:8080'
```

## Restart the Gerrit service
You must restart the Gerrit service for your authentication type and listen URL changes to take effect:

```
$GERRIT_SITE/bin/gerrit.sh restart
```

## Viewing Gerrit
To view your new basic installation of Gerrit, go to:

```
http://localhost:8080
```

## Next steps
Now that you have a simple version of Gerrit running, use the installation to explore the user interface and learn about Gerrit. For more detailed installation instructions, see [Standalone Daemon Installation Guide](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/install.html).

---

Part of [Gerrit Code Review](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/index.html)