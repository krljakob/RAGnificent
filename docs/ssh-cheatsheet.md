# SSH Cheatsheet

A comprehensive guide for SSH commands and configuration with a focus on Git workflows.

## SSH Key Generation and Setup

### Generate a new SSH key

```bash
# Generate an ED25519 key (recommended modern type)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Generate an RSA key with 4096 bits (wider compatibility)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### Start SSH agent and add your key

```bash
# Start the ssh-agent in the background
eval "$(ssh-agent -s)"

# Add your private key to the SSH agent
ssh-add ~/.ssh/id_ed25519   # For ED25519 key
ssh-add ~/.ssh/id_rsa       # For RSA key
```

### Copy SSH public key to clipboard

```bash
# Linux (requires xclip)
xclip -selection clipboard < ~/.ssh/id_ed25519.pub

# macOS
pbcopy < ~/.ssh/id_ed25519.pub

# Windows PowerShell
Get-Content ~/.ssh/id_ed25519.pub | Set-Clipboard
```

### Add SSH key to GitHub/GitLab

1. Copy your public key using one of the commands above
2. Go to GitHub/GitLab settings
3. Navigate to "SSH and GPG keys"
4. Click "New SSH key" or "Add SSH key"
5. Paste your public key and save

## SSH Configuration

### Create/edit SSH config file

```bash
# Create or edit SSH config file
nano ~/.ssh/config
```

### Sample SSH config

```
# Default GitHub account
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

# Personal GitHub account
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_personal
    IdentitiesOnly yes

# Work GitHub account
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_work
    IdentitiesOnly yes

# GitLab instance
Host gitlab.com
    HostName gitlab.com
    User git
    IdentityFile ~/.ssh/id_gitlab
    IdentitiesOnly yes
    
# Custom server with non-standard port
Host myserver
    HostName myserver.example.com
    User username
    Port 2222
    IdentityFile ~/.ssh/id_server
```

## Using SSH with Git

### Clone a repository with SSH

```bash
# Standard clone with SSH
git clone git@github.com:username/repository.git

# Clone using a specific SSH config (e.g. github-work)
git clone git@github-work:username/repository.git
```

### Change remote URL from HTTPS to SSH

```bash
# Check current remote URL
git remote -v

# Change from HTTPS to SSH
git remote set-url origin git@github.com:username/repository.git

# If using a custom SSH config host
git remote set-url origin git@github-personal:username/repository.git
```

### Add a new remote with SSH

```bash
git remote add upstream git@github.com:original-owner/repository.git
```

## SSH Connection Commands

### Basic SSH connection

```bash
# Connect to a server
ssh username@hostname

# Connect to a server with a specific key
ssh -i ~/.ssh/id_ed25519 username@hostname

# Connect to a server with a specific port
ssh -p 2222 username@hostname

# Connect using a host from SSH config
ssh myserver
```

### SSH with port forwarding

```bash
# Local port forwarding (access remote service locally)
ssh -L 8080:localhost:80 username@hostname

# Remote port forwarding (expose local service to remote host)
ssh -R 8080:localhost:80 username@hostname

# Dynamic port forwarding (SOCKS proxy)
ssh -D 1080 username@hostname
```

### File transfer with SCP

```bash
# Copy file to remote host
scp file.txt username@hostname:/path/to/destination/

# Copy file from remote host
scp username@hostname:/path/to/file.txt local-destination/

# Copy entire directory (recursively) to remote host
scp -r directory/ username@hostname:/path/to/destination/
```

### File synchronization with rsync over SSH

```bash
# Sync local directory to remote
rsync -avz -e ssh /local/directory/ username@hostname:/remote/directory/

# Sync remote directory to local
rsync -avz -e ssh username@hostname:/remote/directory/ /local/directory/
```

## SSH Troubleshooting

### Verify the SSH connection

```bash
# Test GitHub SSH connection
ssh -T git@github.com

# Test GitLab SSH connection
ssh -T git@gitlab.com

# Test custom SSH config connection
ssh -T git@github-personal
```

### Debug SSH connections

```bash
# Verbose connection (add more 'v's for more verbosity)
ssh -vv username@hostname

# Check permissions on SSH files
ls -la ~/.ssh

# Correct SSH directory permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### Check SSH agent

```bash
# List keys added to SSH agent
ssh-add -l

# Add keys to SSH agent with timeout (12 hours)
ssh-add -t 43200 ~/.ssh/id_ed25519
```

## SSH Security Best Practices

1. **Use strong key types**: Use ED25519 or RSA with at least 3072 bits.
2. **Add a passphrase**: Always protect your SSH keys with a strong passphrase.
3. **Use specific hosts in your SSH config**: Avoid wildcard hosts.
4. **Rotate keys periodically**: Create new keys once a year or when you suspect compromise.
5. **Use different keys for different services**: Don't reuse the same key everywhere.

## GitHub-Specific SSH Commands

### Configure Git to use SSH instead of HTTPS

```bash
# Set default to SSH for GitHub
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

### Verify your GitHub SSH key

```bash
# This should return "Hi username! You've successfully authenticated..."
ssh -T git@github.com
```

### Clone a repository using a specific SSH key

```bash
# If you have multiple GitHub accounts set up in your SSH config
git clone git@github-personal:username/repository.git
```
