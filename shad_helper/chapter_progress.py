"""The two terminal rows shared by the chapter workers during a full run."""

import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from io import TextIOBase
from threading import Event, RLock, Thread, get_ident

from tqdm import tqdm

_active = ContextVar("chapter_progress", default=None)


class _WorkerOutput(TextIOBase):
    """Assemble print() fragments into complete log lines, including library output."""
    def __init__(self, progress):
        self.progress = progress
        self.pending = {}

    def write(self, text):
        with self.progress.lock:
            key = get_ident()
            lines = (self.pending.pop(key, "") + text).split("\n")
            for line in lines[:-1]:
                self.progress.log(line.rstrip("\r"))
            if lines[-1]:
                self.pending[key] = lines[-1]
        return len(text)

    def flush(self):
        with self.progress.lock:
            if line := self.pending.pop(get_ident(), ""):
                self.progress.log(line)

    def finish(self):
        with self.progress.lock:
            for line in self.pending.values():
                self.progress.log(line)
            self.pending.clear()


class ChapterProgress:
    def __init__(self, stream=None):
        self.stream = sys.stderr if stream is None else stream
        self.rows = {}
        self.labels = {}
        self.lock = RLock()
        self.closed = Event()
        self.clock = None
        self.redirects = ExitStack()

    def start(self):
        if self.stream.isatty():
            try:
                columns, lines = os.get_terminal_size(self.stream.fileno())
            except (AttributeError, OSError):
                columns = lines = 0
            sized = columns > 0 and lines > 2
            for position, worker in enumerate(("Vision", "Notes")):
                self.rows[worker] = tqdm(total=None, desc=worker, position=position,
                                         dynamic_ncols=sized, ncols=None if sized else 80,
                                         nrows=None if sized else 24, file=self.stream, leave=False)

            self.clock = Thread(target=self._refresh, name="chapter-progress")
            self.clock.start()
        for redirect in (redirect_stdout, redirect_stderr):
            output = _WorkerOutput(self)
            self.redirects.callback(output.finish)
            self.redirects.enter_context(redirect(output))

    def _refresh(self):
        while not self.closed.wait(1):
            with self.lock:
                for row in self.rows.values():
                    row.refresh()

    def update(self, worker, label, phase, *, done=0, total=None, unit=""):
        with self.lock:
            state = (label, phase, total, unit)
            changed = self.labels.get(worker) != state
            self.labels[worker] = state
            if worker in self.rows:
                bar = self.rows[worker]
                if changed:
                    bar.set_description_str(f"{worker}  {label}  {phase}", refresh=False)
                    bar.bar_format = ("{desc} [{elapsed}]" if total is None else
                                      "{desc} |{bar}| {n_fmt}/{total_fmt} " + unit + " [{elapsed}]")
                    bar.total = total
                    bar.reset()
                bar.n = done
                bar.refresh()
            elif changed:
                print(f"{worker}  {label}  {phase}", file=self.stream, flush=True)

    def log(self, text):
        with self.lock:
            if self.rows:
                tqdm.write(text, file=self.stream)
            else:
                print(text, file=self.stream, flush=True)

    def close(self):
        self.redirects.close()
        self.closed.set()
        if self.clock is not None:
            self.clock.join()
        with self.lock:
            for bar in reversed(self.rows.values()):
                bar.close()
            self.rows.clear()


@contextmanager
def notes_display(progress, label, cancel):
    token = _active.set((progress, label, cancel))
    try:
        yield
    finally:
        _active.reset(token)


class ChapterCancelled(Exception):
    """Extraction was interrupted while this notes job was running."""


class NotesCancellation(Event):
    def __init__(self):
        super().__init__()
        self.lock = RLock()
        self.client = None

    def set(self):
        with self.lock:
            super().set()
            client = self.client
        if client is not None:
            client.close()


@contextmanager
def notes_connection(client):
    """Let the extraction owner close an active HTTP request during cancellation."""
    active = _active.get()
    if active is None:
        yield
        return
    cancel = active[2]
    with cancel.lock:
        if cancel.is_set():
            raise ChapterCancelled
        cancel.client = client
    try:
        yield
    finally:
        with cancel.lock:
            cancel.client = None


def check_cancelled():
    active = _active.get()
    if active and active[2].is_set():
        raise ChapterCancelled


def retry_delay(seconds, sleep):
    active = _active.get()
    if active is None:
        sleep(seconds)
    elif active[2].wait(seconds):
        raise ChapterCancelled


def notes_phase(phase, **counts):
    active = _active.get()
    if active:
        progress, label, cancel = active
        if cancel.is_set():
            raise ChapterCancelled
        progress.update("Notes", label, phase, **counts)


def progress_items(items, *, desc, unit, **kwargs):
    if _active.get() is None:
        yield from tqdm(items, desc=desc, unit=unit, **kwargs)
        return
    if not items:
        return
    notes_phase(desc, total=len(items), unit=unit)
    for index, item in enumerate(items, 1):
        yield item
        notes_phase(desc, done=index, total=len(items), unit=unit)
