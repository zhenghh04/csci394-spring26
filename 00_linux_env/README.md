# Linux Command Line Basics (Review)

A quick refresher for navigating and working in a Linux terminal.

## Navigation
```bash
pwd            # print current directory
ls             # list files
ls -la         # list with hidden files and details
cd <dir>       # change directory
cd ..          # go up one level
cd ~           # go to home
```

## Files and folders
```bash
touch file.txt         # create empty file
mkdir new_folder       # create folder
cp src dst             # copy file or folder (-r for folders)
mv src dst             # move/rename
rm file.txt            # remove file
rm -r folder            # remove folder (recursive)
```

## Viewing and searching
```bash
cat file.txt            # print file
less file.txt           # page through file (q to quit)
head -n 5 file.txt       # first 5 lines
tail -n 5 file.txt       # last 5 lines
wc -l file.txt           # line count
rg "pattern" .           # search text (ripgrep)
find . -name "*.py"      # find files by name
```

## Permissions
```bash
chmod +x script.sh       # make executable
chmod 644 file.txt       # rw for owner, r for group/others
ls -l                    # view permissions
```

## Processes and jobs
```bash
ps -u $USER              # list your processes
htop                     # interactive process viewer (if available)
ctrl+c                   # stop a running command
command &                # run in background
jobs                     # list background jobs
```

## Networking basics
```bash
ssh user@host            # connect to a remote host
scp file user@host:/path # copy file to remote host
```

## Python and environments
```bash
python3 --version          # check Python version
python3 -m venv venv       # create virtual environment
source venv/bin/activate   # activate venv
pip install matplotlib     # install matplotlib
```

## Git basics
```bash
git status            # show working tree status
git pull              # fetch + merge from remote
git add <file>        # stage changes
git commit -m "message" # create a commit
git push              # send commits to remote
```

## Tips
- Use tab to auto-complete paths and commands.
- Use the up/down arrows to navigate command history.
- Prefer `rg` over `grep` when available.
