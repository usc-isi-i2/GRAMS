import subprocess, os, time, sys


def snapshot(outdir: str, cwd=None):
    """Snapshot your code"""
    outdir = os.path.abspath(outdir)
    assert os.path.exists(outdir), f"Direction \"{outdir}\" does not exist"
    commit_file = os.path.join(outdir, "commit.txt")
    changes_file = os.path.join(outdir, "changes.patch")

    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))

    print("Saving your changes", end="", flush="")
    start_time = time.time()
    # save current commit id
    commit_id = subprocess.check_output("git rev-parse HEAD", shell=True, cwd=cwd)
    commit_id = commit_id.decode().strip()
    with open(commit_file, "w") as f:
        f.write(commit_id)
    print(".", end="", flush=True)

    # save current change
    stash_id = subprocess.check_output("git stash create 'save your changes'", shell=True, cwd=cwd)
    stash_id = stash_id.decode().strip()

    if stash_id == "":
        # there is no change in tracked file
        print(" No changes in tracked files", end="", flush=True)
    else:
        subprocess.check_call(f"git stash store -m 'save your changes' {stash_id}", shell=True, cwd=cwd)
        print(".", end="", flush=True)
        subprocess.check_call(f"git stash show -p --binary > {changes_file}", shell=True, cwd=cwd)
        print(".", end="", flush=True)
        subprocess.check_call(f"git stash drop -q", shell=True, cwd=cwd)

    duration = time.time() - start_time
    print(f". DONE! (Taking {duration:.2f} seconds)", flush=True)


if __name__ == "__main__":
    snapshot(sys.argv[1])