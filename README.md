注意：文件是按照所在包路径命名，读者如果使用，可自行拆解文件名，放入指定包下即可。注意：别忘了删除同级目录下的编译文件。

**说明：**

1、graphrag.llm.openai.utils.py
```
发给embeding模型编码之前，转换为str，如：str(chunk)
#add by jidechao 
def remove_json_markers(input_string):
    start_marker = "```json"
    end_marker = "```"
    if input_string.startswith(start_marker) and input_string.endswith(end_marker):
        return input_string[len(start_marker):-len(end_marker)].strip()
    return input_string

def try_parse_json_object(input: str) -> dict:
    """Generate JSON-string output using best-attempt prompting & parsing techniques."""
    try:
        result = json.loads(remove_json_markers(input))
    except json.JSONDecodeError:
        log.exception("error loading json, json=%s", input)
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result
```
2、graphrag.query.llm.oai.embedding.py
```
加载json之前，去掉固化的json字符
def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                #modify by jidechao chunk str function conversion
                embedding, chunk_len = self._embed_with_retry(str(chunk), **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            # TODO: catch a more specific exception
            except Exception as e:  # noqa BLE001
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )

                continue
```
3、graphrag.query.structured_search.global_search.search.py
```
加载json之前，去掉固化的json字符
def parse_search_response(self, search_response: str) -> list[dict[str, Any]]:
        """Parse the search response json and return a list of key points.

        Parameters
        ----------
        search_response: str
            The search response json string

        Returns
        -------
        list[dict[str, Any]]
            A list of key points, each key point is a dictionary with "answer" and "score" keys
        """
        if search_response.startswith("```json") and search_response.endswith("```"):
            search_response =  search_response[len("```json"):-len("```")].strip()

        parsed_elements = json.loads(search_response)["points"]
        return [
            {
                "answer": element["description"],
                "score": int(element["score"]),
            }
            for element in parsed_elements
        ]
```

4、tiktoken.model.py
```
增加deepseek-chat模型与cl100k_base的兼容性
MODEL_TO_ENCODING: dict[str, str] = {
    # chat
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",  # Common shorthand
    "gpt-35-turbo": "cl100k_base",  # Azure deployment name
    "deepseek-chat": "cl100k_base", 
    # base
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # DEPRECATED MODELS
    # text (DEPRECATED)
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # code (DEPRECATED)
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    # edit (DEPRECATED)
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # old embeddings (DEPRECATED)
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    # open source
    "gpt2": "gpt2",
    "gpt-2": "gpt2",  # Maintains consistency with gpt-4
}
```

