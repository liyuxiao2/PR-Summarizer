# PR Summarizer

A Python tool that automatically generates concise summaries of GitHub Pull Requests using AI-powered text summarization.

## Features

- Fetches comprehensive PR data from GitHub API (title, description, commits, comments, file diffs)
- Generates 3-5 bullet point summaries of key changes
- Supports both HuggingFace Transformers (BART) and OpenAI API for summarization
- Modular, well-structured code with proper error handling
- Fallback to manual summarization when AI services are unavailable

## Installation

### 1. Clone or Download

Save the `pr_summarizer.py` and `requirements.txt` files to your local directory.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The transformers library and its dependencies (PyTorch) are large (~1-2GB). For a lighter installation, you can install only the basic requirements:

```bash
pip install requests openai
```

### 3. Set Up API Tokens

#### GitHub Token (Required)

1. Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (for private repos) or `public_repo` (for public repos only)
4. Copy the generated token

Set the token as an environment variable:

```bash
# Linux/Mac
export GITHUB_TOKEN="your_github_token_here"

# Windows
set GITHUB_TOKEN=your_github_token_here
```

#### OpenAI API Key (Optional - for better summaries)

1. Sign up at [OpenAI](https://platform.openai.com/)
2. Get your API key from the API keys section
3. Set as environment variable:

```bash
# Linux/Mac
export OPENAI_API_KEY="your_openai_key_here"

# Windows
set OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Basic Usage

```bash
python pr_summarizer.py https://github.com/owner/repo/pull/123
```

### With Command Line Arguments

```bash
# Specify tokens via command line
python pr_summarizer.py https://github.com/owner/repo/pull/123 --github-token YOUR_TOKEN --openai-key YOUR_OPENAI_KEY
```

### Example

```bash
python pr_summarizer.py https://github.com/microsoft/vscode/pull/12345
```

## Output

The program outputs a structured summary like this:

```
============================================================
PULL REQUEST SUMMARY
============================================================
• Add dark mode support to editor interface
• Description: Implements user-requested dark theme with automatic detection...
• Modified 15 files across 3 file types: ts, css, json
• Medium PR: +247/-89 lines changed
• 4 commits with various changes
============================================================
```

## Code Structure

The program is organized into these key functions:

- `fetch_pr_data(repo_owner, repo_name, pr_number)` - Fetches PR data from GitHub API
- `parse_diff(files)` - Analyzes file changes and generates change summaries
- `generate_summary(pr_content)` - Creates AI-powered or manual summaries
- `main()` - Handles command-line interface and orchestrates the process

## Error Handling

- **Invalid GitHub URLs**: Clear error messages for malformed URLs
- **API Rate Limits**: Proper handling of GitHub API limits
- **Missing Dependencies**: Graceful fallback when AI libraries aren't available
- **Network Issues**: Comprehensive error reporting for API failures

## Troubleshooting

### "Module not found" errors
Install missing dependencies:
```bash
pip install -r requirements.txt
```

### GitHub API errors
- Check that your token has appropriate permissions
- Verify the PR URL is correct and accessible
- Check if you've hit API rate limits (5000 requests/hour for authenticated users)

### Large model downloads
The first time you use HuggingFace transformers, it will download the BART model (~1.6GB). This is normal and only happens once.

### OpenAI API errors
- Verify your API key is correct
- Check your OpenAI account has available credits
- The program will fallback to HuggingFace or manual summarization if OpenAI fails

## Requirements

- Python 3.7+
- GitHub API token
- Internet connection
- OpenAI API key (optional, for enhanced summaries)

## Dependencies

- `requests` - GitHub API communication
- `transformers` - HuggingFace BART model (optional)
- `torch` - Required by transformers (optional)
- `openai` - OpenAI API integration (optional)
- `tokenizers` - Text tokenization (optional)