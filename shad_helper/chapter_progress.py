"""The two terminal rows shared by the chapter workers during a full run."""

import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from io import TextIOBase
from threading import Event, RLock, Thread, get_ident
from time import monotonic

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


class _ChapterBar(tqdm):
    """Accumulate active time and sample both clocks on each terminal redraw."""
    timer_format = "{desc} {timers}"

    def __init__(self, **kwargs):
        self.chapter_id = None
        self.running_since = None
        self.chapter_seconds = self.total_seconds = 0.0
        self.phase = self.child = ""
        super().__init__(bar_format=self.timer_format, **kwargs)

    def _running_seconds(self, now):
        return 0.0 if self.running_since is None else now - self.running_since

    def pause(self, now):
        elapsed = self._running_seconds(now)
        self.chapter_seconds += elapsed
        self.total_seconds += elapsed
        self.running_since = None

    def set_phase(self, worker, label, phase, total, unit):
        chapter, _, part = label.partition("  part ")
        chapter_id, _, position = chapter.partition("  ")
        now = monotonic()
        self.pause(now)
        if chapter_id != self.chapter_id:
            self.chapter_id = chapter_id
            self.chapter_seconds = 0.0
        if phase not in {"waiting for vision", "complete", "failed", "interrupted", "manual handoff"}:
            self.running_since = now

        number, _, self.child = part.partition(" ")
        suffix = f"  part {number}" if number else ""
        self.desc = f"{worker}  {chapter}{suffix}"
        self.compact_desc = f"{worker} {position or chapter_id}{suffix}"
        self.phase = phase
        self.total, self.unit = total, unit
        self.bar_format = self.timer_format
        if total is not None:
            self.bar_format += " |{bar}| {n_fmt}/{total_fmt} {unit}"
        self.bar_format += " {activity}"

    @property
    def format_dict(self):
        fields = super().format_dict
        running = self._running_seconds(monotonic())
        chapter = self.format_interval(self.chapter_seconds + running)
        total = self.format_interval(self.total_seconds + running)
        timers = f"[chapter {chapter} | total {total}]"
        phase, detail = self.phase, self.child
        needed = len(f"{self.desc} {timers} {phase}")
        if self.total is not None:
            needed += len(f" | | {self.n}/{self.total} {self.unit}")
        if self.chapter_id and fields["ncols"] and needed > fields["ncols"]:
            fields["prefix"] = self.compact_desc
            phase = phase.removeprefix("Illustration ")
            detail = detail or f"({self.chapter_id})"
        fields.update(timers=timers, activity=f"{phase} {detail}" if detail else phase)
        return fields


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
                self.rows[worker] = _ChapterBar(
                    total=None, desc=worker, position=position, dynamic_ncols=sized,
                    ncols=None if sized else 80, nrows=None if sized else 24, file=self.stream, leave=False)

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
                    bar.set_phase(worker, label, phase, total, unit)
                bar.n = done
                bar.refresh()
            elif changed:
                print(f"{worker}  {label}  {phase}", file=self.stream, flush=True)

    def log(self, text):
        with self.lock:
            if self.rows:
                _ChapterBar.write(text, file=self.stream)
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
        with progress.lock:
            if (row := progress.rows.get("Notes")) is not None:
                row.pause(monotonic())


class ChapterCancelled(Exception):
    """Extraction was interrupted while this notes job was running."""


class NotesCancellation(Event):
    def __init__(self):
        super().__init__()
        self.lock = RLock()
        self.interrupt = None

    def set(self):
        with self.lock:
            super().set()
            interrupt = self.interrupt
        if interrupt is not None:
            interrupt()


def notes_connection(client):
    return cancel_on_interrupt(client.close)


@contextmanager
def cancel_on_interrupt(interrupt):
    """Let the extraction owner interrupt a blocking notes request."""
    active = _active.get()
    if active is None:
        yield
        return
    cancel = active[2]
    with cancel.lock:
        if cancel.is_set():
            raise ChapterCancelled
        cancel.interrupt = interrupt
    try:
        yield
    finally:
        with cancel.lock:
            cancel.interrupt = None


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
