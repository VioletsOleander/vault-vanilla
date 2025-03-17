---
version: "1.3"
---
# Basic Tutorial
## Your first Hydra app
### A simple command-line application
This is a simple Hydra application that prints your configuration. The `my_app` function is a placeholder for your code. We will slowly evolve this example to showcase more Hydra features.

The examples in this tutorial are available [here](https://github.com/facebookresearch/hydra/blob/main/examples/tutorials/basic).

my_app.py

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == "__main__":    
    my_app()
```

In this example, Hydra creates an empty `cfg` object and passes it to the function annotated with `@hydra.main`.

>  `@hydra.main` 修饰的函数接受一个 `DictConfig` 对象，Hydra 会创建一个空的 `DictConfig` 然后传递给它

You can add config values via the command line. The `+` indicates that the field is new.

```
$ python my_app.py +db.driver=mysql +db.user=omry +db.password=secret
db:  
    driver: mysql  
    user: omry  
    password: secret
```

>  这样的程序将可以接受命令行参数，可以通过 `+` 传递新的 field 和值

info
See the [version_base page](https://hydra.cc/docs/upgrades/version_base/) for details on the `version_base` parameter.

See [Hydra's command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/) and [Basic Override Syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information about the command line.

Last updated on **Nov 19, 2022** by **Bálint Mucsányi**

### Specifying a config file
It can get tedious to type all those command line arguments. You can solve it by creating a configuration file next to my_app.py. Hydra configuration files are yaml files and should have the .yaml file extension.

config.yaml

```yaml
db:   
    driver: mysql  
    user: omry  
    password: secret
```

Specify the config name by passing a `config_name` parameter to the @hydra.main() decorator. Note that you should omit the **.yaml** extension. Hydra also needs to know where to find your config. Specify the directory containing it relative to the application by passing `config_path`:

my_app.py

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

`config.yaml` is loaded automatically when you run your application.

```
$ python my_app.py
db:
  driver: mysql
  user: omry
  password: secret
```

>  在 `@hydra.main()` 中，可以直接指定配置文件的路径和名称，此时直接运行程序，Hydra 会自动加载配置文件

You can override values in the loaded config from the command line.  
Note the lack of the `+` prefix.

```
$ python my_app.py db.user=root db.password=1234
db:
  driver: mysql
  user: root
  password: 1234
```

>  配置文件中的 field 的值会被在命令行中传递的新的值覆盖

Use `++` to override a config value if it's already in the config, or add it otherwise. e.g:

```
# Override an existing item
$ python my_app.py ++db.password=1234

# Add a new item
$ python my_app.py ++db.timeout=5
```

>  `++` 可以用于将覆盖值写入配置文件，或者为配置文件添加新的 field 和值

You can enable [tab completion](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for your Hydra applications.

Last updated on **Mar 28, 2022** by **Pádraig Brady**

### Using the config object
Here are some basic features of the Hydra Configuration Object:

config.yaml

```yaml
node:                         # Config is hierarchical
  loompa: 10                  # Simple value
  zippity: ${node.loompa}     # Value interpolation
  do: "oompa ${node.loompa}"  # String interpolation
  waldo: ???                  # Missing value, must be populated prior to access
```

main.py

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig):
    assert cfg.node.loompa == 10          # attribute style access
    assert cfg["node"]["loompa"] == 10    # dictionary style access

    assert cfg.node.zippity == 10         # Value interpolation
    assert isinstance(cfg.node.zippity, int)  # Value interpolation type
    assert cfg.node.do == "oompa 10"      # string interpolation

    cfg.node.waldo                        # raises an exception

if __name__ == "__main__":
    my_app()
```

>  `cfg: DictConfig` 可以用 attribute style 访问其属性，也可以用 dictionary style 访问其属性

Outputs:

```
$ python my_app.py 
Traceback (most recent call last):
  File "my_app.py", line 32, in my_app
    cfg.node.waldo
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: node.waldo
    full_key: node.waldo
    object_type=dict
```

Hydra's configuration object is an instance of OmegaConf's `DictConfig`. You can learn more about OmegaConf [here](https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation).

Last updated on **Mar 28, 2022** by **Pádraig Brady**

### Grouping config files
Suppose you want to benchmark your application on each of PostgreSQL and MySQL. To do this, use config groups.

A _**Config Group**_ is a named group with a set of valid options. Selecting a non-existent config option generates an error message with the valid options.
>  配置组指一个带有一组有效配置选项的命名组
>  选择不存在的配置选项会生成一条包含有效选项的错误消息

#### Creating config groups
To create a config group, create a directory, e.g. `db`, to hold a file for each database configuration option. Since we are expecting to have multiple config groups, we will proactively move all the configuration files into a `conf` directory.
>  我们在配置目录 `conf` 中创建目录 `db` ，用于存放数据库配置选项文件，每个配置文件对应一个配置选项

Directory layout

```
├─ conf
│  └─ db
│      ├─ mysql.yaml
│      └─ postgresql.yaml
└── my_app.py
```

db/mysql.yaml

```
driver: mysql
user: omry
password: secret
```

db/postgresql.yaml

```
driver: postgresql
user: postgres_user
password: drowssap
timeout: 10
```

#### Using config groups
Since we moved all the configs into the `conf` directory, we need to tell Hydra where to find them using the `config_path` parameter. **`config_path` is a directory relative to `my_app.py`**.

my_app.py

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

>  在 `@hydra.main()` 中指定 `config_path="conf"`，即存放配置组目录 `db` 的目录名称

Running `my_app.py` without requesting a configuration will print an empty config.
>  此时，不显式请求配置时，运行程序将打印空配置

```
$ python my_app.py
{}
```

Select an item from a config group with `+GROUP=OPTION`, e.g:

```
$ python my_app.py +db=postgresql
db:
  driver: postgresql
  pass: drowssap
  timeout: 10
  user: postgres_user
```

>  通过 `+GROUP=OPTION` 为指定配置组指定配置选项

By default, the config group determines where the config content is placed inside the final config object. In Hydra, the path to the config content is referred to as the config `package`. 
>  默认情况下，配置组决定了配置内容在最终配置对象 `cfg` 中的存放位置
>  Hydra 中，配置内容的存放路径称为配置的包

The package of `db/postgresql.yaml` is `db`:
>  例如，配置内容 `db/postgresql.yaml` 的包就是 `db`

Like before, you can still override individual values in the resulting config:

```
$ python my_app.py +db=postgresql db.timeout=20
db:
  driver: postgresql
  pass: drowssap
  timeout: 20
  user: postgres_user
```

#### Advanced topics
- Config content can be relocated via package overrides. See [Reference Manual/Packages](https://hydra.cc/docs/advanced/overriding_packages/).
- Multiple options can be selected from the same Config Group by specifying them as a list.  
    See [Common Patterns/Selecting multiple configs from a Config Group](https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/)

Last updated on **Nov 19, 2022** by **Bálint Mucsányi**

### Selecting default configs
After office politics, you decide that you want to use MySQL by default. You no longer want to type `+db=mysql` every time you run your application.

You can add a **Default List** to your config file. A **Defaults List** is a list telling Hydra how to compose the final config object. By convention, it is the first item in the config.
>  可以在 `config_name` (注意不是 `config_path` )指定的文件中添加默认列表，Hydra 根据该列表选择为配置组选择默认的配置选项

#### Config group defaults

config.yaml

```
defaults:
  - db: mysql
```

Remember to specify the `config_name`:

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

When you run the updated application, MySQL is loaded by default.

```
$ python my_app.py
db:
  driver: mysql
  pass: secret
  user: omry
```

You can have multiple items in the defaults list, e.g.

```
defaults:
 - db: mysql
 - db/mysql/engine: innodb
```

The defaults are ordered:

- If multiple configs define the same value, the last one wins.
- If multiple configs contribute to the same dictionary, the result is the combined dictionary.

##### Overriding a config group default
You can still load PostgreSQL, and override individual values.

```
$ python my_app.py db=postgresql db.timeout=20
db:
  driver: postgresql
  pass: drowssap
  timeout: 20
  user: postgres_user
```

You can remove a default entry from the defaults list by prefixing it with ~:
>  `~` 可以用于移除默认配置列表

```
$ python my_app.py ~db
{}
```

#### Composition order of primary config
Your primary config can contain both config values and a Defaults List. In such cases, you should add the `_self_` keyword to your defaults list to specify the composition order of the config file relative to the items in the defaults list.

- If you want your primary config to override the values of configs from the Defaults List, append `_self_` to the end of the Defaults List.
- If you want the configs from the Defaults List to override the values in your primary config, insert `_self_` as the first item in your Defaults List.

>  主配置文件中 (`config_name` 指定的文件) 既可以包括配置值也可以包括默认配置列表，当二者同时存在时，需要在默认配置列表中添加一个 `_self_` 关键字，用以指定配置文件相对于默认配置列表中项目的组成顺序
>  - 如果希望主配置覆盖默认配置列表中的配置值，则将 `_self_` 追加在默认列表尾部 (默认列表中，多个配置定义相同值时，the last one wins)
>  - 如果希望默认列表中的配置值覆盖主配置文件中的配置值，将 `_self_` 追加在默认列表头部

config.yaml

```
defaults:
  - db: mysql
  - _self_

db:
  user: root
```

Result config: `db.user` from config.yaml

```
db:
  driver: mysql  # db/mysql.yaml
  pass: secret   # db/mysql.yaml 
  user: root     # config.yaml
```

config.yaml

```
defaults:
  - _self_
  - db: mysql

db:
  user: root
```

Result config: All values from db/mysql

```
db:
  driver: mysql # db/mysql.yaml
  pass: secret  # db/mysql.yaml
  user: omry    # db/mysql.yaml
```

See [Composition Order](https://hydra.cc/docs/advanced/defaults_list/#composition-order) for more information.

info
The default composition order changed between Hydra 1.0 and Hydra 1.1.

- **Hydra 1.0**: Configs from the defaults list are overriding the primary config
- **Hydra 1.1**: A config is overriding the configs from the defaults list.

To mitigate confusion, Hydra 1.1 issue a warning if the primary config contains both Default List and Config values, and `_self_` is not specified in the Defaults List.  
The warning will disappear if you add `_self_` to the Defaults List based on the desired behavior.

#### Non-config group defaults
Sometimes a config file does not belong in any config group. You can still load it by default. Here is an example for `some_file.yaml`.

```
defaults:
  - some_file
```

Config files that are not part of a config group will always be loaded. They cannot be overridden.  

>  如果一个配置文件不属于任何的配置组，我们也可以在默认列表中加入它，以默认加载它
>  不属于任何配置组的配置文件不会被覆盖

Prefer using a config group.

info
For more information about the Defaults List see [Reference Manual/The Defaults List](https://hydra.cc/docs/advanced/defaults_list/).

Last updated on **Mar 28, 2022** by **Pádraig Brady**

### Putting it all together
As software gets more complex, we resort to modularity and composition to keep it manageable. We can do the same with configs. Suppose we want our working example to support multiple databases, with multiple schemas per database, and different UIs. We wouldn't write a separate class for each permutation of db, schema and UI, so we shouldn't write separate configs either. We use the same solution in configuration as in writing the underlying software: composition.
>  随着软件变得复杂，我们将使用模块化和组合来保持其可管理，配置文件也可以这样做
>  假设我们希望我们的示例可以支持多个数据库，每个数据库有多个模式和不同的 UI。我们不会为每个数据库、模式和 UI 的排列组合编写一个单独的类，故我们也不应该编写分离的配置
>  我们在配置中使用的解决方案和编写底层软件时相同：组合

To do this in Hydra, we first add a `schema` and a `ui` config group:

Directory layout

```
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   ├── schema
│   │   ├── school.yaml
│   │   ├── support.yaml
│   │   └── warehouse.yaml
│   └── ui
│       ├── full.yaml
│       └── view.yaml
└── my_app.py
```

With these configs, we already have 12 possible combinations. Without composition, we would need 12 separate configs. A single change, such as renaming `db.user` to `db.username`, requires editing all 12 of them. This is a maintenance nightmare.
>  我们在 `conf` 目录中添加了三个配置组 `db, schema, ui` ，这三个配置组已经有了 12 个可能组合

Composition can come to the rescue. Instead of creating 12 different config files, that fully specify each config, create a single config that specifies the different configuration dimensions, and the default for each.

config.yaml

```
defaults:
  - db: mysql
  - ui: full
  - schema: school
```

The resulting configuration is a composition of the _mysql_ database, the _full_ ui, and the _school_ schema (which we are seeing for the first time here):

```
$ python my_app.py
db:
  driver: mysql
  user: omry
  pass: secret
ui:
  windows:
    create_db: true
    view: true
schema:
  database: school
  tables:
  - name: students
    fields:
    - name: string
    - class: int
  - name: exams
    fields:
    - profession: string
    - time: data
    - class: int
```

Stay tuned to see how to run all of the combinations automatically ([Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)).

#### Summary
- The addition of each new db, schema, or ui only requires a single file.
- Each config group can have a default specified in the Defaults List.
- Any combination can be composed by selecting the desired option from each config group in the Defaults List or the command line.

> - 每当添加新的数据库（db）、模式（schema）或用户界面（ui）时，只需要一个文件即可完成。
> - 每个配置组都可以在默认列表中指定默认值。
> - 通过从默认列表或命令行中选择每个配置组的所需选项，可以组合成任何组合。

## Running your Hydra app
### Multi-run
Sometimes you want to run the same application with multiple different configurations.  
E.g. running a performance test on each of the databases with each of the schemas.
>  有时，我们希望以多种不同的配置运行同一个应用程序，例如，在每个数据库的各个模式下运行性能测试

You can multirun a Hydra application via either command-line or configuration:
>  可以通过命令行或配置文件来多实例运行 Hydra 应用程序

#### Configure `hydra.mode` (new in Hydra 1.2)
You can configure `hydra.mode` in any supported way. The legal values are `RUN` and `MULTIRUN`. The following shows how to override from the command-line and sweep over all combinations of the dbs and schemas. Setting `hydra.mode=MULTIRUN` in your input config would make your application multi-run by default.
>  `hydra.mode` 的合法值为 `RUN` 和 `MULTIRUN` ，以下展示了如何用命令行进行覆盖，并遍历所有数据库和模式的组合
>  在命令行中传入 `hydra.mode=MULTIRUN` 将使得应用程序默认以多运行模式运行

```
$ python my_app.py hydra.mode=MULTIRUN db=mysql,postgresql schema=warehouse,support,school

[2021-01-20 17:25:03,317][HYDRA] Launching 6 jobs locally[2021-01-20 17:25:03,318][HYDRA]        #0 : db=mysql schema=warehouse[2021-01-20 17:25:03,458][HYDRA]        #1 : db=mysql schema=support[2021-01-20 17:25:03,602][HYDRA]        #2 : db=mysql schema=school[2021-01-20 17:25:03,755][HYDRA]        #3 : db=postgresql schema=warehouse[2021-01-20 17:25:03,895][HYDRA]        #4 : db=postgresql schema=support[2021-01-20 17:25:04,040][HYDRA]        #5 : db=postgresql schema=school
```

The printed configurations have been omitted for brevity.

#### `--multirun (-m)` from the command-line
You can achieve the above from command-line as well:

```
python my_app.py --multirun db=mysql,postgresql schema=warehouse,support,school
```

or

```
python my_app.py -m db=mysql,postgresql schema=warehouse,support,school
```

You can access `hydra.mode` at runtime to determine whether the application is in RUN or MULTIRUN mode. Check [here](https://hydra.cc/docs/configure_hydra/intro/) on how to access Hydra config at run time.

>  或者传入 `--multirun` 或 `-m` 也可以达到相同的效果

If conflicts arise (e.g., `hydra.mode=RUN` and the application was run with `--multirun`), Hydra will determine the value of `hydra.mode` at run time. The following table shows what runtime `hydra.mode` value you'd get with different input configs and command-line combinations.
>  如果出现冲突，例如设置了 `hydar.mode=RUN` 但应用通过 `--multirun` 启动时，Hydra 会在运行时决定 `hydra.mode` 的具体值，依据以下的表格

|                           | No multirun command-line flag | --multirun ( -m)                    |
| ------------------------- | ----------------------------- | ----------------------------------- |
| hydra.mode=RUN            | RunMode.RUN                   | RunMode.MULTIRUN (with UserWarning) |
| hydra.mode=MULTIRUN       | RunMode.MULTIRUN              | RunMode.MULTIRUN                    |
| hydra.mode=None (default) | RunMode.RUN                   | RunMode.MULTIRUN                    |

info
Hydra composes configs lazily at job launching time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

#### Sweeping via `hydra.sweeper.params`
You can also define sweeping in the input configs by overriding `hydra.sweeper.params`. Using the above example, the same multirun could be achieved via the following config.

```
hydra:
  sweeper:
    params:
      db: mysql,postgresql
      schema: warehouse,support,school
```

The syntax are consistent for both input configs and command-line overrides. If a sweep is specified in both an input config and at the command line, then the command-line sweep will take precedence over the sweep defined in the input config. If we run the same application with the above input config and a new command-line override:

```
$ python my_app.py -m db=mysql

[2021-01-20 17:25:03,317][HYDRA] Launching 3 jobs locally[2021-01-20 17:25:03,318][HYDRA]        #0 : db=mysql schema=warehouse[2021-01-20 17:25:03,458][HYDRA]        #1 : db=mysql schema=support[2021-01-20 17:25:03,602][HYDRA]        #2 : db=mysql schema=school
```

info
The above configuration methods only apply to Hydra's default `BasicSweeper` for now. For other sweepers, please check out the corresponding documentations.

#### Additional sweep types
Hydra supports other kinds of sweeps, e.g.:

```
x=range(1,10)                  # 1-9schema=glob(*)                 # warehouse,support,schoolschema=glob(*,exclude=w*)      # support,school
```

See the [Extended Override syntax](https://hydra.cc/docs/advanced/override_grammar/extended/) for details.

#### Sweeper
The default sweeping logic is built into Hydra. Additional sweepers are available as plugins. For example, the [Ax Sweeper](https://hydra.cc/docs/plugins/ax_sweeper/) can automatically find the best parameter combination!

#### Launcher
By default, Hydra runs your multi-run jobs locally and serially. Other launchers are available as plugins for launching in parallel and on different clusters. For example, the [JobLib Launcher](https://hydra.cc/docs/plugins/joblib_launcher/) can execute the different parameter combinations in parallel on your local machine using multi-processing.