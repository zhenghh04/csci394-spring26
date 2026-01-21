# PowerShell Basics (Review)

A quick refresher for navigating and working in Windows PowerShell.

## Navigation
```powershell
pwd                # alias for Get-Location
ls                 # alias for Get-ChildItem
ls -Force          # include hidden files
cd <dir>           # alias for Set-Location
cd ..              # go up one level
cd ~               # go to home
```

## Files and folders
```powershell
ni file.txt -ItemType File            # alias for New-Item (create empty file)
mkdir new_folder                      # alias for New-Item -ItemType Directory
cp src dst                            # alias for Copy-Item (-Recurse for folders)
mv src dst                            # alias for Move-Item
rm file.txt                           # alias for Remove-Item
rm folder -Recurse                    # remove folder (recursive)
```

## Viewing and searching
```powershell
cat file.txt                          # alias for Get-Content
gc file.txt | Select-Object -First 5  # first 5 lines
gc file.txt | Select-Object -Last 5   # last 5 lines
gc file.txt | Measure-Object -Line    # line count
Select-String -Path *.py -Pattern "pattern"   # search text
ls -Recurse -Filter "*.py"                     # find files by name
```

## Permissions
```powershell
Get-Acl file.txt                      # view permissions
# For NTFS permissions, use icacls if needed
icacls file.txt
```

## Processes and jobs
```powershell
ps                                    # alias for Get-Process
kill -Name notepad                    # alias for Stop-Process
start notepad                         # alias for Start-Process
Get-Job                               # list background jobs
```

## Networking basics
```powershell
ssh user@host                         # connect to a remote host
scp file user@host:/path              # copy file to remote host (requires OpenSSH)
```

## Python and environments
```powershell
python --version                # check Python version
python -m venv venv             # create virtual environment
.\venv\Scripts\Activate.ps1     # activate venv
pip install matplotlib          # install matplotlib
```

## Git basics
```powershell
git status            # show working tree status
git pull              # fetch + merge from remote
git add <file>        # stage changes
git commit -m "message" # create a commit
git push              # send commits to remote
```

## Tips
- Use tab to auto-complete paths and commands.
- Use the up/down arrows to navigate command history.
- Use `Get-Help <cmd>` for built-in documentation.
