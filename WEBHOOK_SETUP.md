# Webhook Setup Guide

This guide explains how to set up the PR Summarizer as a GitHub webhook to automatically update PR descriptions with AI-generated summaries.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   export OPENAI_API_KEY="your_openai_key_here"  # Optional
   export WEBHOOK_SECRET="your_webhook_secret"   # Optional but recommended
   ```

3. **Run the webhook server:**
   ```bash
   python pr_summarizer.py webhook --port 5000
   ```

4. **Configure GitHub webhook** (see detailed steps below)

## Detailed Setup

### 1. Server Setup

#### Option A: Local Development
```bash
# Run on localhost:5000
python pr_summarizer.py webhook --port 5000 --host 127.0.0.1
```

#### Option B: Production Server
```bash
# Run on all interfaces
python pr_summarizer.py webhook --port 5000 --host 0.0.0.0
```

#### Option C: With Custom Configuration
```bash
python pr_summarizer.py webhook \
  --port 8080 \
  --host 0.0.0.0 \
  --github-token "your_token" \
  --webhook-secret "your_secret"
```

### 2. GitHub Webhook Configuration

1. Go to your GitHub repository
2. Navigate to **Settings** → **Webhooks** → **Add webhook**
3. Configure the webhook:
   - **Payload URL**: `http://your-server.com:5000/webhook`
   - **Content type**: `application/json`
   - **Secret**: (optional but recommended for security)
   - **Events**: Select "Pull requests" only
   - **Active**: ✅ Check this box

### 3. Required Permissions

Your GitHub token needs these permissions:
- `repo` scope (for private repositories)
- `public_repo` scope (for public repositories)
- Ability to read and write pull requests

### 4. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub Personal Access Token |
| `OPENAI_API_KEY` | No | OpenAI API key for better summaries |
| `WEBHOOK_SECRET` | No | Secret for webhook signature verification |

### 5. Testing the Setup

1. **Health Check:**
   ```bash
   curl http://your-server:5000/health
   ```

2. **Create a test PR** in your configured repository
3. Check that the PR description gets updated automatically

## How It Works

1. When a PR is **opened** in your repository, GitHub sends a webhook event
2. The webhook server receives the event and verifies the signature (if configured)
3. The server fetches comprehensive PR data using GitHub API
4. An AI-generated summary is created using OpenAI API or HuggingFace transformers
5. The PR description is updated with the original content plus the auto-generated summary

## Example Updated PR Description

```markdown
[Original PR description content]

---

**Auto-generated Summary:**
• Add dark mode support to editor interface
• Modified 15 files across 3 file types: ts, css, json
• Medium PR: +247/-89 lines changed
• 4 commits with various changes
```

## Troubleshooting

### Webhook Not Receiving Events
- Verify the webhook URL is publicly accessible
- Check GitHub webhook delivery logs in repository settings
- Ensure the webhook is configured for "Pull request" events

### Signature Verification Failures
- Make sure `WEBHOOK_SECRET` matches the secret configured in GitHub
- Verify the secret is properly URL-encoded

### API Rate Limits
- GitHub API: 5,000 requests/hour for authenticated users
- OpenAI API: Varies by plan and model

### Server Errors
- Check server logs for detailed error messages
- Verify GitHub token permissions
- Ensure all required dependencies are installed

## Security Considerations

1. **Use HTTPS** in production
2. **Configure webhook secrets** for signature verification
3. **Restrict access** to your webhook endpoint
4. **Use environment variables** for sensitive data
5. **Monitor logs** for suspicious activity

## CLI Mode (Backward Compatibility)

The original CLI functionality is still available:

```bash
# Old format (still works)
python pr_summarizer.py https://github.com/owner/repo/pull/123

# New explicit format
python pr_summarizer.py cli https://github.com/owner/repo/pull/123
```