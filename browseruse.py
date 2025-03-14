from browser_use import Agent, Browser, BrowserConfig
from langchain_anthropic import ChatAnthropic
import asyncio
# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Initialize the model
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.3,
    timeout=100, # Increase for complex tasks
)

# Create agent with the model
agent = Agent(
    task="please say hello world in google",
    llm=llm,
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())