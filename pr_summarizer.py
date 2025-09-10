#!/usr/bin/env python3
"""
PR Summarizer - A tool to generate concise summaries of GitHub Pull Requests

This script fetches PR data from GitHub and generates human-readable summaries
using either HuggingFace transformers or OpenAI API for text summarization.
"""

import os
import sys
import argparse
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from transformers import pipeline, BartTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class PRData:
    """Data structure to hold PR information"""
    title: str
    description: str
    commits: List[str]
    comments: List[str]
    files_changed: List[Dict]
    additions: int
    deletions: int


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors"""
    pass


class PRSummarizer:
    """Main class for PR summarization functionality"""
    
    def __init__(self, github_token: str, openai_key: Optional[str] = None):
        self.github_token = github_token
        self.openai_key = openai_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        })
        
        if openai_key:
            openai.api_key = openai_key
    
    def fetch_pr_data(self, repo_owner: str, repo_name: str, pr_number: int) -> PRData:
        """
        Fetch comprehensive PR data from GitHub API
        
        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            pr_number: Pull request number
            
        Returns:
            PRData object containing all PR information
        """
        base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        
        try:
            # Fetch basic PR info
            pr_response = self.session.get(f"{base_url}/pulls/{pr_number}")
            pr_response.raise_for_status()
            pr_info = pr_response.json()
            
            # Fetch commits
            commits_response = self.session.get(f"{base_url}/pulls/{pr_number}/commits")
            commits_response.raise_for_status()
            commits_data = commits_response.json()
            
            # Fetch comments
            comments_response = self.session.get(f"{base_url}/pulls/{pr_number}/comments")
            comments_response.raise_for_status()
            comments_data = comments_response.json()
            
            # Fetch issue comments (general PR comments)
            issue_comments_response = self.session.get(f"{base_url}/issues/{pr_number}/comments")
            issue_comments_response.raise_for_status()
            issue_comments_data = issue_comments_response.json()
            
            # Fetch files changed
            files_response = self.session.get(f"{base_url}/pulls/{pr_number}/files")
            files_response.raise_for_status()
            files_data = files_response.json()
            
            # Process the data
            commits = [commit['commit']['message'] for commit in commits_data]
            
            # Combine review comments and issue comments
            all_comments = []
            for comment in comments_data:
                all_comments.append(comment['body'])
            for comment in issue_comments_data:
                all_comments.append(comment['body'])
            
            return PRData(
                title=pr_info['title'],
                description=pr_info['body'] or "",
                commits=commits,
                comments=all_comments,
                files_changed=files_data,
                additions=pr_info['additions'],
                deletions=pr_info['deletions']
            )
            
        except requests.exceptions.RequestException as e:
            raise GitHubAPIError(f"Failed to fetch PR data: {str(e)}")
        except KeyError as e:
            raise GitHubAPIError(f"Unexpected API response format: {str(e)}")
    
    def parse_diff(self, files: List[Dict]) -> str:
        """
        Parse file diffs to extract meaningful change information
        
        Args:
            files: List of file change objects from GitHub API
            
        Returns:
            Formatted string describing the changes
        """
        if not files:
            return "No files changed."
        
        change_summary = []
        
        # Group files by change type
        added_files = []
        modified_files = []
        deleted_files = []
        renamed_files = []
        
        for file in files:
            filename = file['filename']
            status = file['status']
            
            if status == 'added':
                added_files.append(filename)
            elif status == 'modified':
                modified_files.append(filename)
            elif status == 'removed':
                deleted_files.append(filename)
            elif status == 'renamed':
                renamed_files.append(f"{file.get('previous_filename', 'unknown')} → {filename}")
        
        if added_files:
            change_summary.append(f"Added {len(added_files)} files: {', '.join(added_files[:3])}{'...' if len(added_files) > 3 else ''}")
        
        if modified_files:
            change_summary.append(f"Modified {len(modified_files)} files: {', '.join(modified_files[:3])}{'...' if len(modified_files) > 3 else ''}")
        
        if deleted_files:
            change_summary.append(f"Deleted {len(deleted_files)} files: {', '.join(deleted_files[:3])}{'...' if len(deleted_files) > 3 else ''}")
        
        if renamed_files:
            change_summary.append(f"Renamed {len(renamed_files)} files: {', '.join(renamed_files[:2])}{'...' if len(renamed_files) > 2 else ''}")
        
        return ". ".join(change_summary)
    
    def _summarize_with_transformers(self, text: str) -> str:
        """Summarize text using HuggingFace transformers"""
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            # BART has a max input length, so we truncate if necessary
            max_length = 1024
            if len(text) > max_length:
                text = text[:max_length]
            
            result = summarizer(text, max_length=150, min_length=50, do_sample=False)
            return result[0]['summary_text']
        
        except Exception as e:
            return f"Error in transformer summarization: {str(e)}"
    
    def _summarize_with_openai(self, text: str) -> str:
        """Summarize text using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of code changes and pull requests."},
                    {"role": "user", "content": f"Please provide a concise summary of this pull request in 3-5 bullet points:\n\n{text}"}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error in OpenAI summarization: {str(e)}"
    
    def generate_summary(self, pr_data: PRData) -> str:
        """
        Generate a comprehensive summary of the PR
        
        Args:
            pr_data: PRData object containing PR information
            
        Returns:
            Human-readable summary string
        """
        # Prepare text for summarization
        summary_parts = []
        
        # Add title and description
        summary_parts.append(f"Title: {pr_data.title}")
        if pr_data.description.strip():
            summary_parts.append(f"Description: {pr_data.description}")
        
        # Add commit information
        if pr_data.commits:
            commit_summary = "Commits: " + "; ".join(pr_data.commits[:3])
            if len(pr_data.commits) > 3:
                commit_summary += f" (and {len(pr_data.commits) - 3} more)"
            summary_parts.append(commit_summary)
        
        # Add file changes
        file_changes = self.parse_diff(pr_data.files_changed)
        summary_parts.append(f"File Changes: {file_changes}")
        
        # Add stats
        summary_parts.append(f"Changes: +{pr_data.additions} additions, -{pr_data.deletions} deletions")
        
        # Add significant comments if any
        if pr_data.comments:
            important_comments = [comment for comment in pr_data.comments if len(comment) > 50]
            if important_comments:
                summary_parts.append(f"Key Comments: {important_comments[0][:100]}...")
        
        full_text = "\n".join(summary_parts)
        
        # Try to use AI summarization if available
        if self.openai_key and OPENAI_AVAILABLE:
            ai_summary = self._summarize_with_openai(full_text)
            if not ai_summary.startswith("Error"):
                return ai_summary
        
        if TRANSFORMERS_AVAILABLE:
            ai_summary = self._summarize_with_transformers(full_text)
            if not ai_summary.startswith("Error"):
                return ai_summary
        
        # Fallback to manual summary
        return self._create_manual_summary(pr_data)
    
    def _create_manual_summary(self, pr_data: PRData) -> str:
        """Create a manual summary when AI services are unavailable"""
        summary_points = []
        
        # Extract key information
        summary_points.append(f"• {pr_data.title}")
        
        if pr_data.description and len(pr_data.description.strip()) > 10:
            desc_preview = pr_data.description.strip()[:100]
            if len(pr_data.description) > 100:
                desc_preview += "..."
            summary_points.append(f"• Description: {desc_preview}")
        
        # File change summary
        if pr_data.files_changed:
            file_count = len(pr_data.files_changed)
            file_types = set()
            for file in pr_data.files_changed:
                ext = file['filename'].split('.')[-1] if '.' in file['filename'] else 'no extension'
                file_types.add(ext)
            
            summary_points.append(f"• Modified {file_count} files across {len(file_types)} file types: {', '.join(list(file_types)[:3])}")
        
        # Change magnitude
        total_changes = pr_data.additions + pr_data.deletions
        if total_changes > 500:
            magnitude = "Large"
        elif total_changes > 100:
            magnitude = "Medium"
        else:
            magnitude = "Small"
        
        summary_points.append(f"• {magnitude} PR: +{pr_data.additions}/-{pr_data.deletions} lines changed")
        
        # Commit information
        if pr_data.commits:
            if len(pr_data.commits) == 1:
                summary_points.append(f"• Single commit: {pr_data.commits[0][:60]}...")
            else:
                summary_points.append(f"• {len(pr_data.commits)} commits with various changes")
        
        return "\n".join(summary_points[:5])  # Limit to 5 points


def parse_github_url(url: str) -> Tuple[str, str, int]:
    """
    Parse GitHub PR URL to extract owner, repo, and PR number
    
    Args:
        url: GitHub PR URL
        
    Returns:
        Tuple of (owner, repo_name, pr_number)
    """
    # Match GitHub PR URL pattern
    pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, url)
    
    if not match:
        raise ValueError("Invalid GitHub PR URL format. Expected: https://github.com/owner/repo/pull/number")
    
    return match.group(1), match.group(2), int(match.group(3))


def main():
    """Main function to run the PR summarizer"""
    parser = argparse.ArgumentParser(description="Generate summaries for GitHub Pull Requests")
    parser.add_argument('pr_url', help='GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)')
    parser.add_argument('--github-token', help='GitHub API token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--openai-key', help='OpenAI API key for better summaries (optional)')
    
    args = parser.parse_args()
    
    # Get GitHub token
    github_token = args.github_token or os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Error: GitHub token is required. Set GITHUB_TOKEN environment variable or use --github-token")
        sys.exit(1)
    
    # Get OpenAI key (optional)
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    
    try:
        # Parse the GitHub URL
        owner, repo, pr_number = parse_github_url(args.pr_url)
        print(f"Analyzing PR #{pr_number} in {owner}/{repo}...")
        
        # Initialize summarizer
        summarizer = PRSummarizer(github_token, openai_key)
        
        # Fetch PR data
        print("Fetching PR data from GitHub...")
        pr_data = summarizer.fetch_pr_data(owner, repo, pr_number)
        
        # Generate summary
        print("Generating summary...")
        summary = summarizer.generate_summary(pr_data)
        
        # Output results
        print("\n" + "="*60)
        print("PULL REQUEST SUMMARY")
        print("="*60)
        print(summary)
        print("="*60)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except GitHubAPIError as e:
        print(f"GitHub API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()