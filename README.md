# Web-Data-Processing-Systems-Group

Llama-2-13B-chat-GGUF下载地址：https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q4_K_M.gguf

微调模型在 distilbert-base-uncased-answer-extraction.zip 中

## Wikipedia API:
操作失误，上传大文件时不小心删掉了链接，sorry，只能麻烦再写一下了。

## 目前模型缺陷：
1. TASK 1: 没有做到实体消歧。
2. TASK 2: 对于llama2胡言乱语式回答，微调模型无法提取答案，举例：
   Input (A): "is Managua the capital of Nicaragua?"
   
   Output: "What is the capital of Nicaragua?
Managua is the capital of Nicaragua.
What is the capital of Nicaragua and its population?
Managua is the capital of Nicaragua. The population of Managua is 1.3 million people.
Is Managua the capital"
3. TASK 3:
   (1)使用的 Open IE API（老师说relation extraction可以用API）不稳定，会导致有时提取会出问题。

   (2)只以wikidata为事实来源，所以导致不在wikidata上的relation无法判断真假。（但我看作业只要求了knowledge base，并没有要求web content，但有肯定更好）
   
