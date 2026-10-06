import asyncio

from mcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main() -> None:
    async with Client(MCP_SERVER_URL) as client:
        print(f"Protocol version: {client.protocol_version}")

        # 1. Verify capabilities
        tools = await client.list_tools()
        print("Tools:")
        for tool in tools.tools:
            print(f"  - {tool.name}")

        # 2. Execute our real RAG tool
        result = await client.call_tool(
            "search_documents",
            {
                "query": "What is this document about?",
                "top_k": 3,
            },
        )

        print("\nsearch_documents result:")
        print(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
