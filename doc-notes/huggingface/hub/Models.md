# The Model Hub

## What is the Model Hub?
The Model Hub is where the members of the Hugging Face community can host all of their model checkpoints for simple storage, discovery, and sharing. Download pre-trained models with the [`huggingface_hub` client library](https://huggingface.co/docs/huggingface_hub/index), with 🤗 [`Transformers`](https://huggingface.co/docs/transformers/index) for fine-tuning and other usages or with any of the over [15 integrated libraries](https://huggingface.co/docs/hub/models-libraries). You can even leverage the [Serverless Inference API](https://huggingface.co/docs/hub/models-inference) or [Inference Endpoints](https://huggingface.co/docs/inference-endpoints). to use models in production settings.

# Model Cards

## What are Model Cards?
Model cards are files that accompany the models and provide handy information. Under the hood, model cards are simple Markdown files with additional metadata. Model cards are essential for discoverability, reproducibility, and sharing! You can find a model card as the `README.md` file in any model repo.

The model card should describe:

- the model
- its intended uses & potential limitations, including biases and ethical considerations as detailed in [Mitchell, 2018](https://arxiv.org/abs/1810.03993)
- the training params and experimental info (you can embed or link to an experiment tracking platform for reference)
- which datasets were used to train your model
- the model’s evaluation results

> model card 即 model 仓库的 `README.md` 文件
> model card 应该描述：
> 模型本身
> 用途、潜在的限制 (包括偏置和道德上的考虑)
> 训练参数和试验信息
> 使用的训练数据集
> 模型的评估结果

The model card template is available [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md).

How to fill out each section of the model card is described in [the Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated).

Model Cards on the Hub have two key parts, with overlapping information:

- [Metadata](https://huggingface.co/docs/hub/model-cards#model-card-metadata)
- [Text descriptions](https://huggingface.co/docs/hub/model-cards#model-card-text)

> model card 的两个关键部分为：元数据、文本描述

## Model card metadata
A model repo will render its `README.md` as a model card. The model card is a [Markdown](https://en.wikipedia.org/wiki/Markdown) file, with a [YAML](https://en.wikipedia.org/wiki/YAML) section at the top that contains metadata about the model.
> model card 为 markdown 文件，它的顶部应该有 yaml 部分描述 model 的元数据

The metadata you add to the model card supports discovery and easier use of your model. For example:

- Allowing users to filter models at [https://huggingface.co/models](https://huggingface.co/models).
- Displaying the model’s license.
- Adding datasets to the metadata will add a message reading `Datasets used to train:` to your model page and link the relevant datasets, if they’re available on the Hub.

> model card 的元数据用于帮助用户查找和使用模型，例如：
> 用于过滤式搜索特定类型的模型
> 展示模型的许可证
> 如果元数据中的数据集可以在 huggingface hub 中存在，model card 中会直接包含指向该数据集的 URL

Dataset, metric, and language identifiers are those listed on the [Datasets](https://huggingface.co/datasets), [Metrics](https://huggingface.co/metrics) and [Languages](https://huggingface.co/languages) pages.

### Adding metadata to your model card
There are a few different ways to add metadata to your model card including:

- Using the metadata UI
- Directly editing the YAML section of the `README.md` file
- Via the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) Python library, see the [docs](https://huggingface.co/docs/huggingface_hub/guides/model-cards#update-metadata) for more details.

Many libraries with [Hub integration](https://huggingface.co/docs/hub/models-libraries) will automatically add metadata to the model card when you upload a model.
> 许多集成了 huggingface hub 的库会自动在我们上传模型时添加元数据

#### Using the metadata UI
You can add metadata to your model card using the metadata UI. To access the metadata UI, go to the model page and click on the `Edit model card` button in the top right corner of the model card. This will open an editor showing the model card `README.md` file, as well as a UI for editing the metadata.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/metadata-ui-editor.png)

This UI will allow you to add key metadata to your model card and many of the fields will autocomplete based on the information you provide. Using the UI is the easiest way to add metadata to your model card, but it doesn’t support all of the metadata fields. If you want to add metadata that isn’t supported by the UI, you can edit the YAML section of the `README.md` file directly.

#### Editing the YAML section of the README.md file
You can also directly edit the YAML section of the `README.md` file. If the model card doesn’t already have a YAML section, you can add one by adding three `---` at the top of the file, then include all of the relevant metadata, and close the section with another group of `---` like the example below:

```
---
language: 
  - "List of ISO 639-1 code for your language"
  - lang1
  - lang2
thumbnail: "url to a thumbnail used in social sharing"
tags:
- tag1
- tag2
license: "any valid license identifier"
datasets:
- dataset1
- dataset2
metrics:
- metric1
- metric2
base_model: "base model Hub identifier"
---
```

You can find the detailed model card metadata specification [here](https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1).

### Specifying a library
You can specify the supported libraries in the model card metadata section. Find more about our supported libraries [here](https://huggingface.co/docs/hub/models-libraries). The library will be specified in the following order of priority:

1. Specifying `library_name` in the model card (recommended if your model is not a `transformers` model). This information can be added via the metadata UI or directly in the model card YAML section:

```
library_name: flair
```

2. Having a tag with the name of a library that is supported

```
tags:
- flair
```

> model card 的元数据区域可以用于指定模型中支持的库，指定的方式有： 1. `library_name` (如果模型不是 `transformers` 模型，都推荐使用该方式) 2. 在 `tags` 中指定支持库名

If it’s not specified, the Hub will try to automatically detect the library type. However, this approach is discouraged, and repo creators should use the explicit `library_name` as much as possible.

1. By looking into the presence of files such as `*.nemo` or `*.mlmodel`, the Hub can determine if a model is from NeMo or CoreML.
2. In the past, if nothing was detected and there was a `config.json` file, it was assumed the library was `transformers`. For model repos created after August 2024, this is not the case anymore – so you need to `library_name: transformers` explicitly.

> 如果没有指定，huggingface hub 会自动检测库的类型
> 推荐尽量显式用 `library_name` ，包括 `transformers` 模型 (`libraray_name: transformers`)

### Specifying a base model
If your model is a fine-tune, an adapter, or a quantized version of a base model, you can specify the base model in the model card metadata section. This information can also be used to indicate if your model is a merge of multiple existing models. Hence, the `base_model` field can either be a single model ID, or a list of one or more base_models (specified by their Hub identifiers).
> 如果我们的模型是基础模型的微调/量化，或者融合了多个模型，可以在 `base_model` 中指定基础模型

```
base_model: HuggingFaceH4/zephyr-7b-beta
```

This metadata will be used to display the base model on the model page. Users can also use this information to filter models by base model or find models that are derived from a specific base model:

For a fine-tuned model:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/base-model-ui.png)

For an adapter (LoRA, PEFT, etc):

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/base_model_adapter.png)

For a quantized version of another model:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/base_model_quantized.png)

For a merge of two or more models:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/base_model_merge.png)

In the merge case, you specify a list of two or more base_models:

```
base_model:
- Endevor/InfinityRP-v1-7B
- l3utterfly/mistral-7b-v0.1-layla-v4
```

The Hub will infer the type of relationship from the current model to the base model (`"adapter", "merge", "quantized", "finetune"`) but you can also set it explicitly if needed: `base_model_relation: quantized` for instance.
> huggingface hub 会自动推理本模型和基础模型的关系 (`adapter, merge, quantized, finetune` )，我们也可以在 `base_model_relation` 中指定

### Specifying a new version
If a new version of your model is available in the Hub, you can specify it in a `new_version` field.
> `new_version` 指定模型的新版本

For example, on `l3utterfly/mistral-7b-v0.1-layla-v3`:

```
new_version: l3utterfly/mistral-7b-v0.1-layla-v4
```

This metadata will be used to display a link to the latest version of a model on the model page. If the model linked in `new_version` also has a `new_version` field, the very latest version will always be displayed.
> 如果 `new_version` 指向的也有 `new_version` ，就展示最新的

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/new_version.png)

### Specifying a dataset
You can specify the datasets used to train your model in the model card metadata section. The datasets will be displayed on the model page and users will be able to filter models by dataset. You should use the Hub dataset identifier, which is the same as the dataset’s repo name as the identifier:

```
datasets:
- imdb
- HuggingFaceH4/no_robots
```

### Specifying a task ( pipeline_tag )
You can specify the `pipeline_tag` in the model card metadata. The `pipeline_tag` indicates the type of task the model is intended for. This tag will be displayed on the model page and users can filter models on the Hub by task. This tag is also used to determine which [widget](https://huggingface.co/docs/hub/models-widgets#enabling-a-widget) to use for the model and which APIs to use under the hood.
> `pipeline_tag` 指定模型用于哪种类型的任务，该 tag 也会被用于决定哪些 widget 和 API 会被用于模型

For `transformers` models, the pipeline tag is automatically inferred from the model’s `config.json` file but you can override it in the model card metadata if required. Editing this field in the metadata UI will ensure that the pipeline tag is valid. Some other libraries with Hub integration will also automatically add the pipeline tag to the model card metadata.
> `transformers` 模型的 `pipeline_tag` 自动从模型的 `cofig.json` 文件中推导，推导的结果可以手动覆盖

### Specifying a license
You can specify the license in the model card metadata section. The license will be displayed on the model page and users will be able to filter models by license. Using the metadata UI, you will see a dropdown of the most common licenses.

If required, you can also specify a custom license by adding `other` as the license value and specifying the name and a link to the license in the metadata.

```
# Example from https://huggingface.co/coqui/XTTS-v1
---
license: other
license_name: coqui-public-model-license
license_link: https://coqui.ai/cpml
---
```

If the license is not available via a URL you can link to a LICENSE stored in the model repo.

### Evaluation Results
You can specify your **model’s evaluation results** in a structured way in the model card metadata. Results are parsed by the Hub and displayed in a widget on the model page. Here is an example on how it looks like for the [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) model:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/eval-results-v2.png)

The metadata spec was based on Papers with code’s [model-index specification](https://github.com/paperswithcode/model-index). This allow us to directly index the results into Papers with code’s leaderboards when appropriate. You can also link the source from where the eval results has been computed.

Here is a partial example to describe [01-ai/Yi-34B](https://huggingface.co/01-ai/Yi-34B)’s score on the ARC benchmark. The result comes from the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) which is defined as the `source`:

```
---
model-index:
  - name: Yi-34B
    results:
      - task:
          type: text-generation
        dataset:
          name: ai2_arc
          type: ai2_arc
        metrics:
          - name: AI2 Reasoning Challenge (25-Shot)
            type: AI2 Reasoning Challenge (25-Shot)
            value: 64.59
        source:
          name: Open LLM Leaderboard
          url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
---
```

For more details on how to format this data, check out the [Model Card specifications](https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1).

### CO2 Emissions
The model card is also a great place to show information about the CO2 impact of your model. Visit our [guide on tracking and reporting CO2 emissions](https://huggingface.co/docs/hub/model-cards-co2) to learn more.

### Linking a Paper
If the model card includes a link to a paper on arXiv, the Hugging Face Hub will extract the arXiv ID and include it in the model tags with the format `arxiv:<PAPER ID>`. Clicking on the tag will let you:

- Visit the Paper page
- Filter for other models on the Hub that cite the same paper.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/models-arxiv.png)

Read more about Paper pages [here](https://huggingface.co/docs/hub/paper-pages).

## Model Card text
Details on how to fill out a human-readable model card without Hub-specific metadata (so that it may be printed out, cut+pasted, etc.) is available in the [Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated).

## FAQ

### How are model tags determined?
Each model page lists all the model’s tags in the page header, below the model name. These are primarily computed from the model card metadata, although some are added automatically, as described in [Enabling a Widget](https://huggingface.co/docs/hub/models-widgets#enabling-a-widget).

### Can I add custom tags to my model?
Yes, you can add custom tags to your model by adding them to the `tags` field in the model card metadata. The metadata UI will suggest some popular tags, but you can add any tag you want. For example, you could indicate that your model is focused on finance by adding a `finance` tag.

### How can I indicate that my model is not suitable for all audiences
You can add a `not-for-all-audience` tag to your model card metadata. When this tag is present, a message will be displayed on the model page indicating that the model is not for all audiences. Users can click through this message to view the model card.

### Can I write LaTeX in my model card?
Yes! The Hub uses the [KaTeX](https://katex.org/) math typesetting library to render math formulas server-side before parsing the Markdown.

You have to use the following delimiters:

- `$$ ... $$` for display mode
- `\\(...\\)` for inline mode (no space between the slashes and the parenthesis).
