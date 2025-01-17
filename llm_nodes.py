import os
import json
import hashlib
import openai
import datetime
import random

from config import config_path, config_key, current_dir_path
from tools.base_utils import load_api_keys
from LLM.llm_utils import Chat

"""
大模型api加载器节点
"""
class LLM_api_loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "deepseepk",
                                          "tooltip": "The name of the model, such as gpt-4o-mini."}),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The base URL of the API, such as https://api.openai.com/v1.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The API key for the API."
                    },
                ),
                "is_ollama": ("BOOLEAN", {"default": False, "tooltip": "Whether to use ollama."}),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The loaded model.",)
    DESCRIPTION = "Load the model in openai format."
    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "LLM_LEGO/模型加载器（model loader）"

    def chatbot(self, model_name, base_url=None, api_key=None, is_ollama=False):
        # 是否使用ollama
        if is_ollama:
            openai.api_key = "ollama"
            openai.base_url = "http://127.0.0.1:11434/v1/"
        else:
            api_keys = load_api_keys(config_path)
            if api_key != "":
                openai.api_key = api_key
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.api_key = api_keys.get("api_key")
            elif api_keys.get("openai_api_key") != "":
                openai.api_key = api_keys.get("openai_api_key")
            if base_url != "":
                openai.base_url = base_url
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.base_url = api_keys.get("base_url")
            elif api_keys.get("base_url") != "":
                openai.base_url = api_keys.get("base_url")
            if openai.api_key == "":
                api_keys = load_api_keys(config_path)
                openai.api_key = api_keys.get("openai_api_key")
                openai.base_url = api_keys.get("base_url")
            if openai.base_url != "":
                if openai.base_url[-1] != "/":
                    openai.base_url = openai.base_url + "/"

        chat = Chat(model_name, openai.api_key, openai.base_url)
        return (chat,)


"""
大模型对话器节点
"""
class LLM:
    def __init__(self):
        current_time = datetime.datetime.now()
        # 以时间戳作为ID，字符串格式 XX年XX月XX日XX时XX分XX秒并加上一个哈希值防止重复
        self.id = current_time.strftime("%Y_%m_%d_%H_%M_%S") + str(hash(random.randint(0, 1000000)))
        global instances
        instances.append(self)
        # 构建prompt.json的绝对路径，如果temp文件夹不存在就创建
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(current_dir_path, "temp"), exist_ok=True)
        self.prompt_path = os.path.join(current_dir_path, "temp", str(self.id) + ".json")
        # 如果文件不存在，创建prompt.json文件，存在就覆盖文件
        if not os.path.exists(self.prompt_path):
            with open(self.prompt_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"role": "system", "content": "你是一个强大的人工智能助手。"}], f, indent=4, ensure_ascii=False
                )
        self.tool_data = {"id": self.id, "system_prompt": "", "type": "api"}
        self.list = []
        self.added_to_list = False
        self.is_locked = "disable"

    @classmethod
    def INPUT_TYPES(s):
        temp_path = os.path.join(current_dir_path, "temp")
        full_paths = [os.path.join(temp_path, f) for f in os.listdir(temp_path)]
        full_paths.sort(key=os.path.getmtime, reverse=True)
        paths = [os.path.basename(f) for f in full_paths]
        paths.insert(0, "")
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": "你一个强大的人工智能助手。","tooltip": "System prompt, used to describe the behavior of the model and the expected output format."}),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好",
                        "tooltip": "User prompt, used to describe the user's request and the expected output format."
                    },
                ),
                "model": ("CUSTOM", {"tooltip": "The model to use for the LLM."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,"tooltip": "The temperature parameter controls the randomness of the model's output. A higher temperature will result in more random and diverse responses, while a lower temperature will result in more focused and deterministic responses."}),
                "is_memory": (["enable", "disable"], {"default": "enable", "tooltip": "Whether to enable memory for the LLM."}),
                "is_tools_in_sys_prompt": (["enable", "disable"], {"default": "disable", "tooltip": "Integrate the tool list into the system prompt, thereby granting some models temporary capability to invoke tools."}),
                "is_locked": (["enable", "disable"], {"default": "disable", "tooltip": "Whether to directly output the result from the last output."}),
                "main_brain": (["enable", "disable"], {"default": "enable", "tooltip": "If this option is disabled, the LLM will become a tool that can be invoked by other LLMs."}),
                "max_length": ("INT", {"default": 1920, "min": 256, "max": 128000, "step": 128, "tooltip": "The maximum length of the output text."}),
            },
            "optional": {
                "system_prompt_input": ("STRING", {"forceInput": True, "tooltip": "System prompt input, used to describe the system's request and the expected output format."}),
                "user_prompt_input": ("STRING", {"forceInput": True, "tooltip": "User prompt input, used to describe the user's request and the expected output format."}),
                "tools": ("STRING", {"forceInput": True, "tooltip": "Tool list, used to describe the tools that the model can invoke."}),
                "file_content": ("STRING", {"forceInput": True, "tooltip": "Input the contents of the file here."}),
                "images": ("IMAGE", {"forceInput": True, "tooltip": "Upload images here."}),
                "imgbb_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional, if not filled out, it will be passed to the LLM in the form of a base64 encoded string. API key for ImgBB, used to upload images to ImgBB and get the image URL."
                    },
                ),
                "conversation_rounds": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1, "tooltip": "The maximum number of dialogue turns that the LLM can see in the history records, where one question and one answer constitute one turn."}),
                "historical_record": (paths, {"default": "", "tooltip": "The dialogue history file is optional; if not selected and left empty, a new dialogue history file will be automatically created."}),
                "is_enable": ("BOOLEAN", {"default": True, "tooltip": "Whether to enable the LLM."}),
                "extra_parameters": ("DICT", {"forceInput": True, "tooltip": "Extra parameters for the LLM."}),
                "user_history": ("STRING", {"forceInput": True, "tooltip": "User history, you can directly input a JSON string containing multiple rounds of dialogue here."}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "assistant_response",
        "history",
        "tool",
        "image",
    )
    OUTPUT_TOOLTIPS = (
        "The assistant's response to the user's request.",
        "The dialogue history",
        "This interface will connect this LLM as a tool to other LLMs.",
        "Images generated or fetched by the LLM.",
    )
    DESCRIPTION = "The API version of the model chain, compatible with all API interfaces."
    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型链（model_chain）"

    def chatbot(
        self,
        user_prompt,
        main_brain,
        system_prompt,
        model,
        temperature,
        is_memory,
        is_tools_in_sys_prompt,
        is_locked,
        max_length,
        system_prompt_input="",
        user_prompt_input="",
        tools=None,
        file_content=None,
        images=None,
        imgbb_api_key=None,
        conversation_rounds=100,
        historical_record="",
        is_enable=True,
        extra_parameters=None,
        user_history=None,
    ):
        if not is_enable:
            return (
                None,
                None,
                None,
                [],
            )
        self.list = [
            main_brain,
            system_prompt,
            model,
            temperature,
            is_memory,
            is_tools_in_sys_prompt,
            is_locked,
            max_length,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            images,
            imgbb_api_key,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
        ]
        if user_prompt is None:
            user_prompt = user_prompt_input
        elif user_prompt_input is not None:
            user_prompt = user_prompt + user_prompt_input
        if historical_record != "":
            temp_path = os.path.join(current_dir_path, "temp")
            self.prompt_path = os.path.join(temp_path, historical_record)
        self.tool_data["system_prompt"] = system_prompt
        if system_prompt_input is not None and system_prompt is not None:
            system_prompt = system_prompt + system_prompt_input
        elif system_prompt is None:
            system_prompt = system_prompt_input
        global llm_tools_list, llm_tools
        if main_brain == "disable":
            if self.added_to_list == False:
                llm_tools_list.append(self.tool_data)
                self.added_to_list = True
        self.is_locked = is_locked
        if self.is_locked == "disable":
            setattr(LLM, "IS_CHANGED", LLM.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(LLM, "IS_CHANGED"):
                delattr(LLM, "IS_CHANGED")
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": "another_llm",
                    "description": "使用llm_tools可以调用其他的智能助手解决你的问题。请根据以下列表中的system_prompt选择你需要的智能助手："
                    + json.dumps(llm_tools_list, ensure_ascii=False, indent=4),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "智能助手的id"},
                            "type": {"type": "string", "description": "智能助手的类型，目前支持api和local两种类型。"},
                            "question": {"type": "string", "description": "问题描述，代表你想要解决的问题。"},
                        },
                        "required": ["id", "type", "question"],
                    },
                },
            }
        ]

        llm_tools_json = json.dumps(llm_tools, ensure_ascii=False, indent=4)
        if (user_prompt is None or user_prompt.strip() == "") and (images is None or images == []) and (user_history is None or user_history == [] or user_history.strip() == ""):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return (
                "",
                str(history),
                llm_tools_json,
                None,
            )
        else:
            try:
                if is_memory == "disable" or "clear party memory" in user_prompt:
                    with open(self.prompt_path, "w", encoding="utf-8") as f:
                        json.dump([{"role": "system", "content": system_prompt}], f, indent=4, ensure_ascii=False)
                    if "clear party memory" in user_prompt:
                        with open(self.prompt_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                        return (
                            "party memory has been cleared, please ask me again, powered by LLM party!",
                            str(history),
                            llm_tools_json,
                            None,
                        )
                api_keys = load_api_keys(config_path)

                with open(self.prompt_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if user_history != "" and user_history is not None:
                    try:
                        history = json.loads(user_history)
                    except:
                        pass
                history_temp = [history[0]]
                elements_to_keep = 2 * conversation_rounds
                if elements_to_keep < len(history) - 1:
                    history_temp += history[-elements_to_keep:]
                    history_copy = history[1:-elements_to_keep]
                else:
                    if len(history) > 1:
                        history_temp += history[1:]
                    history_copy = []
                if len(history_temp) > 1:
                    if history_temp[1]["role"] == "tool":
                        history_temp.insert(1, history[-elements_to_keep - 1])
                        if -elements_to_keep - 1 > 1:
                            history_copy = history[1 : -elements_to_keep - 1]
                        else:
                            history_copy = []
                history = history_temp
                for message in history:
                    if message["role"] == "system":
                        message["content"] = system_prompt
                if is_tools_in_sys_prompt == "enable":
                    tools_list = []
                    GPT_INSTRUCTION = ""
                    if tools is not None:
                        tools_dis = json.loads(tools)
                        tools = None
                        for tool_dis in tools_dis:
                            tools_list.append(tool_dis["function"])
                        tools_instructions = ""
                        tools_instruction_list = []
                        for tool in tools_list:
                            tools_instruction_list.append(tool["name"])
                            tools_instructions += (
                                str(tool["name"])
                                + ":"
                                + "Call this tool to interact with the "
                                + str(tool["name"])
                                + " API. What is the "
                                + str(tool["name"])
                                + " API useful for? "
                                + str(tool["description"])
                                + ". Parameters:"
                                + str(tool["parameters"])
                                + "Required parameters:"
                                + str(tool["parameters"]["required"])
                                + "\n"
                            )
                        REUTRN_FORMAT = '{"tool": "tool name", "parameters": {"parameter name": "parameter value"}}'
                        TOOL_EAXMPLE = 'You will receive a JSON string containing a list of callable tools. Please parse this JSON string and return a JSON object containing the tool name and tool parameters. Here is an example of the tool list:\n\n{"tools": [{"name": "plus_one", "description": "Add one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}},{"name": "minus_one", "description": "Minus one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}}]}\n\nBased on this tool list, generate a JSON object to call a tool. For example, if you need to add one to number 77, return:\n\n{"tool": "plus_one", "parameters": {"number": "77"}}\n\nPlease note that the above is just an example and does not mean that the plus_one and minus_one tools are currently available.'
                        GPT_INSTRUCTION = f"""
        Answer the following questions as best you can. You have access to the following APIs:
        {tools_instructions}

        Use the following format:
        ```tool_json
        {REUTRN_FORMAT}
        ```

        Please choose the appropriate tool according to the user's question. If you don't need to call it, please reply directly to the user's question. When the user communicates with you in a language other than English, you need to communicate with the user in the same language.

        When you have enough information from the tool results, respond directly to the user with a text message without having to call the tool again.
        """

                    for message in history:
                        if message["role"] == "system":
                            message["content"] = system_prompt
                            if tools_list != []:
                                message["content"] += "\n" + TOOL_EAXMPLE + "\n" + GPT_INSTRUCTION + "\n"

                if tools is not None:
                    print(tools)
                    tools = json.loads(tools)

                max_length = int(max_length)

                if file_content is not None:
                    for message in history:
                        if message["role"] == "system":
                            message["content"] += "\n以下是可以参考的已知信息:\n" + file_content
                if extra_parameters is not None and extra_parameters != {}:
                    response, history = model.send(
                        user_prompt, temperature, max_length, history, tools, is_tools_in_sys_prompt,images,imgbb_api_key, **extra_parameters
                    )
                else:
                    response, history = model.send(
                        user_prompt, temperature, max_length, history, tools, is_tools_in_sys_prompt,images,imgbb_api_key
                    )
                print(response)
                # 修改prompt.json文件
                history_get = [history[0]]
                history_get.extend(history_copy)
                history_get.extend(history[1:])
                history = history_get
                with open(self.prompt_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
                history = json.dumps(history, ensure_ascii=False,indent=4)
                global image_buffer
                if image_buffer != [] and image_buffer is not None:
                    image_out = image_buffer.clone()
                else:
                    image_out = None
                image_buffer = []
                return (
                    response,
                    history,
                    llm_tools_json,
                    image_out,
                )
            except Exception as ex:
                print(ex)
                return (
                    str(ex),
                    str(ex),
                    llm_tools_json,
                    None,
                )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value


"""
用于显示文本的节点
"""
class show_text_party:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "LLM_LEGO/文本（text）"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}