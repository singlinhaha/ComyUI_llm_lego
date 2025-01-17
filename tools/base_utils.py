import configparser


def load_api_keys(config_file):
    """
    从指定的配置文件中加载 API 密钥。

    参数:
    config_file (str): 配置文件的路径。

    返回:
    dict: 包含 API 密钥的字典。
    """
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")


    api_keys = {}
    # 检查配置文件中是否存在 "API_KEYS" 部分
    if "API_KEYS" in config:
        # 如果存在，将 "API_KEYS" 部分的内容赋值给 api_keys 字典
        api_keys = config["API_KEYS"]

    # 返回包含 API 密钥的字典
    return api_keys


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if "multi_tool_use." in tool_name:
        tool_name = tool_name.replace("multi_tool_use.", "")
    if '-' in tool_name and tool_name not in _TOOL_HOOKS:
        from .custom_tool.mcp_cli import mcp_client as client

        async def run_client():
            try:
                # Initialize the client (if necessary)
                if client.is_initialized is False:
                    await client.initialize()
                functions = await client.get_openai_functions()
                # Call the tool and get the result
                result = await client.call_tool(tool_name, tool_params)
                return str(result)
            except Exception as e:
                return str(e)

        # Run the async function using asyncio.run
        return asyncio.run(run_client())
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = globals().get(tool_name)
    try:
        ret_out = tool_call(**tool_params)
        if isinstance(ret_out, tuple):
            ret = ret_out[0]
            global image_buffer
            image_buffer = ret_out[1]
            if ret == "" or ret is None:
                ret = "图片已生成。并以展示在本节点的image输出中。可以使用preview_image节点查看图片。"
        else:
            ret = ret_out
    except:
        ret = traceback.format_exc()
    return str(ret)


if __name__ == "__main__":
    print(dict(load_api_keys("../config.ini")))