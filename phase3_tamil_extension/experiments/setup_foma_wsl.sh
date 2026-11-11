# One-time setup (no sudo needed): extracts foma/flookup into ~/foma_local inside WSL.
# Run from PowerShell: wsl -d Ubuntu -- bash "/mnt/e/Project/Final One/phase3_tamil_extension/experiments/setup_foma_wsl.sh"
mkdir -p ~/foma_local
set -e
cd ~/foma_local
apt-get download foma foma-bin libfoma0t64 2>/dev/null || apt-get download foma foma-bin
apt-cache depends foma | grep Depends || true
for f in *.deb; do dpkg -x "$f" root; done
find root -type f | grep -E 'bin/|[.]so'
