# File: summarize_logs.py

import os

def tail(filepath, n=10):
    """Read last n lines of a file"""
    with open(filepath, 'rb') as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        buffer = bytearray()
        lines = 0

        while end > 0 and lines <= n:
            end -= 1
            f.seek(end)
            char = f.read(1)
            buffer.extend(char)
            if char == b'\n':
                lines += 1

        return buffer[::-1].decode(errors='replace').strip()

def summarize_log_files(base_dir='saved_experiments', tail_lines=1):
    for root, _, files in os.walk(base_dir):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            experiment_id = os.path.relpath(root, base_dir)
            print(f"\n=== Log Summary: {experiment_id} ===")
            print(tail(log_path, n=tail_lines))

if __name__ == "__main__":
    summarize_log_files()
