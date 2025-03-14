"""
Enhanced Claude Local Shell with improved task management, prompt handling,
and robust error recovery capabilities.
"""

import asyncio
import cmd
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anthropic
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Configure logging with more detailed format
logging.basicConfig(
    filename='claude_shell.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TaskState:
    """Enhanced task state tracking."""
    in_progress: bool = False
    completed: bool = False
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    
    def should_retry(self) -> bool:
        """Determine if task should be retried."""
        return (
            self.in_progress and 
            not self.completed and 
            self.attempts < self.max_attempts
        )
    
    def check_in(self) -> str:
        """Get current task status."""
        duration = datetime.now() - self.start_time
        status = (
            f"Attempt {self.attempts}/{self.max_attempts}\n"
            f"Duration: {duration.seconds} seconds\n"
            f"Status: {'Completed' if self.completed else 'In Progress'}\n"
        )
        if self.last_error:
            status += f"Last Error: {self.last_error}"
        return status

def get_api_key() -> str:
    """Get Anthropic API key from file with validation."""
    try:
        api_key_path = Path('./API_KEY.txt')
        if not api_key_path.exists():
            raise FileNotFoundError("API_KEY.txt file not found")
            
        key_content = api_key_path.read_text().strip()
        match = re.search(r"API_KEY\s*=\s*['\"]([^'\"]+)['\"]", key_content)
        
        if not match:
            raise ValueError("Could not parse API key from file. Expected format: API_KEY='your-key-here'")
            
        api_key = match.group(1)
        if not api_key or len(api_key) < 10:  # Basic validation
            raise ValueError("Invalid API key format")
            
        return api_key
        
    except Exception as key_error:
        logger.error(f"API key error: {str(key_error)}")
        print(f"Error reading API key: {str(key_error)}")
        print("Please ensure API_KEY.txt exists with format: API_KEY='your-key-here'")
        sys.exit(1)

class Config(BaseSettings):
    """Configuration management using Pydantic."""
    ANTHROPIC_API_KEY: str = Field(default_factory=get_api_key)
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1024
    history_size: int = 10
    command_timeout: int = 30
    max_retries: int = 3
    log_level: str = "DEBUG"
    prompt_file: Optional[str] = None
    
    @property
    def api_key(self):
        return self.ANTHROPIC_API_KEY
    
    class Config:
        env_file_encoding = 'utf-8'

class MessageType(Enum):
    """Enumeration for different types of messages in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    COMMAND = "command"
    SYSTEM = "system"

@dataclass
class CommandOutput:
    """Structure for command execution results."""
    stdout: str
    stderr: str
    returncode: int
    execution_time: float

@dataclass
class ConversationEntry:
    """Structure for conversation history entries."""
    type: MessageType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConversationManager:
    """Manages conversation history and persistence with improved formatting."""
    
    def __init__(self, history_size: int = 10):
        self.history: List[ConversationEntry] = []
        self.history_size = history_size
        self.history_file = Path("conversation_history.json")
        self._load_history()
    
    def add_entry(self, entry: ConversationEntry) -> None:
        """Add a new entry to the conversation history."""
        self.history.append(entry)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        self._save_history()
    
    def get_recent_context(self) -> str:
        """Get formatted recent conversation context with improved readability."""
        formatted_entries = []
        for entry in self.history[-self.history_size:]:
            if entry.type == MessageType.COMMAND:
                cmd_text = f"Command executed: {entry.content}\n"
                if entry.metadata.get('stdout'):
                    cmd_text += f"└─ Output: {entry.metadata['stdout']}\n"
                if entry.metadata.get('stderr'):
                    cmd_text += f"└─ Errors: {entry.metadata['stderr']}\n"
                formatted_entries.append(cmd_text)
            else:
                formatted_entries.append(
                    f"{entry.type.value.capitalize()}: {entry.content}"
                )
        return "\n".join(formatted_entries)
    
    def _save_history(self) -> None:
        """Save conversation history to file."""
        try:
            history_data = [
                {
                    "type": entry.type.value,
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "metadata": entry.metadata
                }
                for entry in self.history
            ]
            self.history_file.write_text(json.dumps(history_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
    
    def _load_history(self) -> None:
        """Load conversation history from file."""
        try:
            if self.history_file.exists():
                history_data = json.loads(self.history_file.read_text())
                self.history = [
                    ConversationEntry(
                        type=MessageType(entry["type"]),
                        content=entry["content"],
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        metadata=entry["metadata"]
                    )
                    for entry in history_data
                ]
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            
class FileAction(BaseModel):
    """Model for file modification actions."""
    file_path: str
    command: str  # 'update', 'rewrite', or 'create'
    content: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None

class FileOperationManager:
    """Handles file operations with safety checks and backup."""
    
    def __init__(self):
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self, file_path: Path) -> None:
        """Create a backup of the file before modification."""
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{file_path.name}.{timestamp}.bak"
            backup_path.write_bytes(file_path.read_bytes())
    
    def update_file(self, file_path: str, old_str: str, new_str: str) -> bool:
        """Update specific content in a file."""
        try:
            path = Path(file_path)
            self.create_backup(path)
            
            content = path.read_text()
            # Count occurrences to ensure unique match
            if content.count(old_str) != 1:
                raise ValueError(f"Found {content.count(old_str)} occurrences of the target string. Must be exactly 1 for safety.")
                
            modified_content = content.replace(old_str, new_str)
            path.write_text(modified_content)
            return True
            
        except Exception as e:
            logger.error(f"File update failed: {e}")
            return False
    
    def rewrite_file(self, file_path: str, content: str) -> bool:
        """Completely rewrite a file with new content."""
        try:
            path = Path(file_path)
            self.create_backup(path)
            path.write_text(content)
            return True
            
        except Exception as e:
            logger.error(f"File rewrite failed: {e}")
            return False
    
    def create_file(self, file_path: str, content: str) -> bool:
        """Create a new file with content."""
        try:
            path = Path(file_path)
            if path.exists():
                raise FileExistsError(f"File {file_path} already exists")
            
            path.write_text(content)
            return True
            
        except Exception as e:
            logger.error(f"File creation failed: {e}")
            return False

class ClaudeShell(cmd.Cmd):
    """Enhanced interactive shell for Claude AI interactions."""
    
    intro = '''
╔═════════════════════════════════════════════════════╗
║            Welcome to Enhanced ClaudeShell          ║
║                                                     ║
║  Available commands:                                ║
║  • help or ? - List available commands              ║
║  • stats     - View conversation statistics         ║
║  • exit      - Exit the shell                       ║
║                                                     ║
║  You can interact naturally with Claude             ║
║  Type your questions or requests in plain language. ║
╚═════════════════════════════════════════════════════╝
'''
    prompt = 'claude> '

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
        self.conversation_manager = ConversationManager(history_size=self.config.history_size)
        self.file_manager = FileOperationManager()
        self.current_task = None
        self.system_prompt = None
        
        # Load custom prompt if specified
        if len(sys.argv) > 1:
            prompt_path = Path(sys.argv[1])
            if prompt_path.suffix == '.md' and prompt_path.exists():
                try:
                    self.system_prompt = prompt_path.read_text(encoding='utf-8')
                    print(f"\nUsing custom prompt: {prompt_path.name}")
                except Exception as e:
                    logger.error(f"Error loading custom prompt: {e}")
                    print(f"\nError loading custom prompt: {e}")
                    print("Falling back to default prompt...")
        
    async def process_actions(self, actions_data: dict) -> None:
        """Process actions with improved error handling and visual feedback."""
        if not isinstance(actions_data, dict) or 'actions' not in actions_data:
            logger.error("Invalid actions data format")
            raise ValueError("Invalid actions data format")

        try:
            print("\n═══════════════════ Executing Actions ═══════════════════")
            
            for action in actions_data['actions']:
                if not isinstance(action, dict) or 'type' not in action:
                    logger.error(f"Invalid action format: {action}")
                    continue

                try:
                    if action['type'] == 'command':
                        print(f"\n┌─ Executing Command")
                        print(f"│ {action['command']}")
                        
                        result = await self.execute_command(
                            action['command'],
                            timeout=action.get('timeout', self.config.command_timeout)
                        )
                        
                        if result.stdout:
                            print("│\n│ Output:")
                            for line in result.stdout.split('\n'):
                                print(f"│ {line}")
                        
                        if result.stderr:
                            print("│\n│ Errors:")
                            for line in result.stderr.split('\n'):
                                print(f"│ {line}")
                        
                        print("└────────────────────")
                        
                        self.conversation_manager.add_entry(
                            ConversationEntry(
                                type=MessageType.COMMAND,
                                content=action['command'],
                                metadata={
                                    "stdout": result.stdout,
                                    "stderr": result.stderr,
                                    "returncode": result.returncode,
                                    "execution_time": result.execution_time
                                }
                            )
                        )
                        
                        if result.returncode != 0:
                            raise Exception(f"Command failed: {result.stderr}")
                    
                    elif action['type'] == 'say':
                        print(f"\nAssistant: {action['message']}")
                        self.conversation_manager.add_entry(
                            ConversationEntry(
                                type=MessageType.ASSISTANT,
                                content=action['message']
                            )
                        )

                    elif action['type'] == 'file':
                        file_action = FileAction(**action)
                        success = False
                        
                        print(f"\n┌─ File Operation")
                        print(f"│ {file_action.command} on {file_action.file_path}")
                        
                        if file_action.command == 'update':
                            success = self.file_manager.update_file(
                                file_action.file_path,
                                file_action.old_str,
                                file_action.new_str
                            )
                        elif file_action.command == 'rewrite':
                            success = self.file_manager.rewrite_file(
                                file_action.file_path,
                                file_action.content
                            )
                        elif file_action.command == 'create':
                            success = self.file_manager.create_file(
                                file_action.file_path,
                                file_action.content
                            )
                        
                        status = "completed successfully" if success else "failed"
                        print(f"└─ Operation {status}")
                        
                        if not success:
                            raise Exception(f"File operation {file_action.command} failed")
                    
                except Exception as action_error:
                    logger.error(f"Error processing action {action}: {action_error}")
                    print(f"\nError processing action: {str(action_error)}")
                    if self.current_task:
                        self.current_task.last_error = str(action_error)
                    continue
            
            print("\n═══════════════════════════════════════════════════════")
        
        except Exception as e:
            logger.error(f"Error processing actions: {e}")
            print(f"\nError executing actions: {e}")
            if self.current_task:
                self.current_task.last_error = str(e)
            raise

    async def execute_command(self, command: str, timeout: int = 30) -> CommandOutput:
        """Execute a command with security checks and timeout."""
        start_time = datetime.now()
        
        # Security check
        blocked_commands = {'rm', 'sudo', 'su', 'dd'}
        if any(cmd in command.split() for cmd in blocked_commands):
            raise ValueError(f"Command contains blocked terms: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Command timed out after {timeout} seconds: {command}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CommandOutput(
                stdout=stdout.decode().strip(),
                stderr=stderr.decode().strip(),
                returncode=process.returncode,
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def send_to_anthropic(self, message: str) -> Optional[str]:
        """Send message to Anthropic API with enhanced context and prompt handling."""
        prompt = self.system_prompt or """You are in an enhanced interactive shell session.
You can respond in three ways:

1. For commands requiring shell interaction:
{
    "actions": [
        {"type": "command", "command": "actual-shell-command", "timeout": optional_timeout},
        {"type": "say", "message": "Here's what I found..."}
    ]
}

2. For file operations:
{
    "actions": [
        {
            "type": "file",
            "command": "update|rewrite|create",
            "file_path": "path/to/file",
            "content": "file content for rewrite/create",
            "old_str": "content to replace for update",
            "new_str": "replacement content for update"
        },
        {"type": "say", "message": "File operation completed..."}
    ]
}

3. For regular conversation, respond naturally.
"""

        context = self.conversation_manager.get_recent_context()
        
        if self.current_task:
            task_status = f"\nCurrent Task Status:\n{self.current_task.check_in()}\n"
        else:
            task_status = ""

        full_prompt = f"""{prompt}

Recent conversation and command history:
{context}
{task_status}
User: {message}"""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            
            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid response from Anthropic API")
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error in send_to_anthropic: {e}")
            if self.current_task:
                self.current_task.last_error = str(e)
            return None

    def default(self, line: str) -> None:
        """Handle user input with improved task management."""
        if not line.strip():
            return
            
        self.conversation_manager.add_entry(
            ConversationEntry(type=MessageType.USER, content=line)
        )
        
        # Initialize task state
        self.current_task = TaskState()
        self.current_task.in_progress = True
        
        asyncio.run(self._process_input(line))

    async def _process_input(self, line: str) -> None:
        """Process user input with retry logic and progress tracking."""
        while self.current_task.should_retry():
            try:
                self.current_task.attempts += 1
                
                # Provide status update on retry attempts
                if self.current_task.attempts > 1:
                    print("\nRetrying task...")
                    print(self.current_task.check_in())
                
                response = await self.send_to_anthropic(line)
                
                if response:
                    try:
                        actions_data = json.loads(response)
                        if isinstance(actions_data, dict) and 'actions' in actions_data:
                            await self.process_actions(actions_data)
                            if not actions_data.get('needs_followup'):
                                self.current_task.completed = True
                                break
                        else:
                            # Regular conversation response
                            print(f"\nAssistant: {response}")
                            self.conversation_manager.add_entry(
                                ConversationEntry(
                                    type=MessageType.ASSISTANT,
                                    content=response
                                )
                            )
                            self.current_task.completed = True
                            break
                    except json.JSONDecodeError:
                        # Handle regular conversation response
                        print(f"\nAssistant: {response}")
                        self.conversation_manager.add_entry(
                            ConversationEntry(
                                type=MessageType.ASSISTANT,
                                content=response
                            )
                        )
                        self.current_task.completed = True
                        break
                else:
                    print("\nNo response received from Claude.")
                    self.current_task.last_error = "No response received"
                    
            except Exception as e:
                logger.error(f"Error in task processing: {e}")
                print(f"\nError: {str(e)}")
                self.current_task.last_error = str(e)
                
                # Wait before retrying
                if self.current_task.should_retry():
                    retry_delay = 2 ** (self.current_task.attempts - 1)  # Exponential backoff
                    print(f"\nWaiting {retry_delay} seconds before retrying...")
                    await asyncio.sleep(retry_delay)
        
        # Final status update
        if not self.current_task.completed:
            print("\nFailed to complete task after maximum attempts.")
            print(self.current_task.check_in())
    
    def do_stats(self, arg: str) -> bool:
        """Display conversation statistics."""
        total_entries = len(self.conversation_manager.history)
        user_messages = sum(1 for entry in self.conversation_manager.history 
                          if entry.type == MessageType.USER)
        assistant_messages = sum(1 for entry in self.conversation_manager.history 
                               if entry.type == MessageType.ASSISTANT)
        commands = sum(1 for entry in self.conversation_manager.history 
                      if entry.type == MessageType.COMMAND)
        
        print("\n═══════════════ Conversation Statistics ═══════════════")
        print(f"Total Entries: {total_entries}")
        print(f"User Messages: {user_messages}")
        print(f"Assistant Messages: {assistant_messages}")
        print(f"Commands Executed: {commands}")
        print("═══════════════════════════════════════════════════════")
        return False
    
    def do_help(self, arg: str) -> bool:
        """Show help information with improved formatting."""
        print("\n═══════════════════ Available Commands ═══════════════════")
        print("Core Commands:")
        print("  help or ?   - Show this help message")
        print("  stats       - Show conversation statistics")
        print("  exit        - Exit the shell")
        print("\nNatural Interaction:")
        print("  You can interact naturally with Claude by typing")
        print("  questions or requests in plain language.")
        print("═══════════════════════════════════════════════════════════")
        return False
    
    def do_exit(self, arg: str) -> bool:
        """Exit the shell."""
        print("\nGoodbye!")
        return True

def main():
    """Main entry point with improved error handling."""
    try:
        # Parse command line arguments for custom prompt
        if len(sys.argv) > 1:
            prompt_path = Path(sys.argv[1])
            if not prompt_path.exists():
                print(f"\nWarning: Prompt file '{prompt_path}' not found.")
                print("Using default prompt instead.")
            elif prompt_path.suffix != '.md':
                print(f"\nWarning: Prompt file must be a markdown file.")
                print("Using default prompt instead.")
        
        shell = ClaudeShell()
        shell.cmdloop()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        print("Please check the log file for more details.")
        sys.exit(1)

if __name__ == '__main__':
    main()
