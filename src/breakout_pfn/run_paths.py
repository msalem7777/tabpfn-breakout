"""Create fresh output folders without overwriting earlier experiments."""
from pathlib import Path


def open_unique_run(path):
    """Reserve the requested folder, or append _001, _002, ... if it exists.

    mkdir reserves each candidate atomically. Even an existing empty directory
    is preserved: it may belong to another process that has just started.
    """
    requested = Path(path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        candidate = requested if number == 0 else requested.with_name(f'{requested.name}_{number:03d}')
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            continue
        print(f'[run-dir] {candidate}', flush=True)
        return candidate


def smoke_test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        requested = Path(directory)/'run'
        first = open_unique_run(requested)
        (first/'keep.txt').write_text('previous result')
        second = open_unique_run(requested)
        third = open_unique_run(requested)
        assert first == requested
        assert second.name == 'run_001' and third.name == 'run_002'
        assert (first/'keep.txt').read_text() == 'previous result'
    print('Run folder smoke test passed (no model/game loaded)')


if __name__ == '__main__':
    smoke_test()
