"""File > Nearby send…: send OpenFlo files to, and receive them from, OpenFlo
on phones and other computers on the same local network
(docs/nearby-send.md).

Three tabs. Send: pick a file and a device found nearby (or type its
address); the first time, type the 6-digit code the other device shows.
Receive: while it is on, this computer is visible to nearby devices; it
shows the code when a new device pairs and always asks before taking a
file. Paired devices: rename this computer, forget a device.

Network work runs on worker threads. They never touch Tk: they put calls
on a queue the window drains every 50 ms (``_Bridge``), and a question to
the user waits on the worker thread for the answer.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from .async_task import run_async
from .nearby import protocol as p
from .nearby import session as sess
from .nearby.store import NearbyStore
from .tool_window import ToolWindow
from .ui_tips import auto_tip

APP = 'openflo'
# What the desktop app sends and takes: kind -> extension (the phone app's
# kinds have the same names, src/files.ts).
KINDS = {'project': '.flowproject', 'session': '.flowsession', 'fcs': '.fcs',
         'workspace': '.wsp'}
ASK_TIMEOUT = 60.0

# Why a transfer ended, in the user's words. Keys are TransferError /
# ProtocolError reasons, then peer reason codes.
_REASONS = {
    'wrong-code': "The code did not match. Nothing was paired or sent.",
    'cancelled': "Canceled.",
    'timeout': "The other device stopped answering.",
    'disconnected': "The connection was lost. Send again to resume where it stopped.",
    'unreachable': ("Could not reach that device. Both must be on the same Wi-Fi, "
                    "and some networks (guest or campus Wi-Fi) keep devices apart."),
    'file-changed': "The file changed while it was being sent. Send it again.",
    'hash-mismatch': "The file arrived damaged and was deleted. Send it again.",
    'too-much-data': "The sender sent more than it offered. The file was deleted.",
    'declined-here': "File declined.",
    'interrupted': ("Another device tried to connect while pairing, so the pairing was "
                    "stopped to be safe. Start again."),
    'not-paired-device': ("This is not the device you paired with: its key has changed. "
                          "If it was reinstalled, forget it under Paired devices and pair again."),
    'wrong-app': "That device is running a different app.",
    'unsupported-version': "The other device has a different version of Nearby send. Update both apps.",
    'bad-commit': "The connection failed a security check.",
    'bad-identity': "The connection failed a security check.",
    'bad-point': "The connection failed a security check.",
    'bad-record': "The connection failed a security check.",
}
_PEER_REASONS = {
    p.R_USER: "The other device said no.",
    p.R_TIMEOUT: "The other device timed out.",
    p.R_WRONG_CODE: "The code was typed wrong on the other device. Nothing was paired.",
    p.R_KIND: "The other device does not take this kind of file.",
    p.R_TOO_BIG: "The file is too big for the other device.",
    p.R_NO_SPACE: "The other device does not have room for this file.",
    p.R_HASH: "The file did not arrive intact. Send it again.",
    p.R_BUSY: "The other device is busy with another transfer.",
    p.R_NOT_PAIRED: "The devices disagree about their pairing. Forget and pair again.",
}


def describe_failure(exc: BaseException) -> str:
    """One sentence for the status line about why a transfer ended."""
    peer = getattr(exc, 'peer_reason', None)
    if peer in _PEER_REASONS:
        return _PEER_REASONS[peer]
    reason = getattr(exc, 'reason', '')
    if reason in _REASONS:
        return _REASONS[reason]
    if isinstance(exc, p.ProtocolError):
        return "The other device broke the transfer protocol; nothing was saved."
    if isinstance(exc, OSError):
        return f"Network error: {exc.strerror or exc}"
    return f"Stopped: {exc}"


def kind_for_path(path: str) -> str | None:
    """The kind a file is sent as, from its name; None if OpenFlo does not send it."""
    low = path.lower()
    for kind, ext in KINDS.items():
        if low.endswith(ext) and len(os.path.basename(low)) > len(ext):
            return kind
    return None


def default_receive_folder() -> str:
    """Where received files go unless the user picks a folder."""
    return os.path.join(os.path.expanduser('~'), 'Downloads', 'OpenFlo')


def size_text(n: int) -> str:
    """A file size as people say it."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.0f} KB"
    return f"{n} bytes"


class _Bridge:
    """Calls from worker threads, run on the Tk thread by the window's poll."""

    def __init__(self) -> None:
        self.q: queue.Queue[Callable[[], None]] = queue.Queue()
        self.closed = threading.Event()

    def post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the Tk thread soon; the worker does not wait."""
        if not self.closed.is_set():
            self.q.put(fn)

    def ask(self, fn: Callable[[], Any], default: Any, timeout: float = ASK_TIMEOUT) -> Any:
        """Run ``fn`` on the Tk thread and wait for its answer; ``default``
        when the window closes or nobody answers within ``timeout``."""
        box: list[Any] = []
        done = threading.Event()

        def run() -> None:
            try:
                box.append(fn())
            finally:
                done.set()
        self.post(run)
        end = threading.Event()
        waited = 0.0
        while not done.is_set() and not self.closed.is_set() and waited < timeout:
            end.wait(0.05)
            waited += 0.05
        return box[0] if box else default

    def drain(self) -> None:
        """Run what the workers posted (on the Tk thread)."""
        while True:
            try:
                fn = self.q.get_nowait()
            except queue.Empty:
                return
            fn()


class NearbyWindow(ToolWindow):
    """The Nearby send window. ``store`` and ``discovery`` are injectable for
    tests; by default the store lives in ``~/.openflo/nearby`` and discovery
    is DNS-SD (openflo.nearby.discovery)."""

    def __init__(self, parent, editor=None, store: NearbyStore | None = None,
                 discovery_factory: Callable[[], Any] | None = None,
                 receive_folder: str | None = None):
        super().__init__(parent, "Nearby send", editor=editor)
        self.editor = editor
        self.store = store or NearbyStore(
            os.path.join(os.path.expanduser('~'), '.openflo', 'nearby'), 'Computer')
        self._make_discovery = discovery_factory or _real_discovery
        self.bridge = _Bridge()
        self.folder = receive_folder or default_receive_folder()
        self._discovery: Any = None
        self._listener: Any = None
        self._found: list[Any] = []
        self._cancel = threading.Event()
        self._sending = False
        self.received: list[sess.Received] = []

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)
        self._build_send(ttk.Frame(self.nb, padding=8))
        self._build_receive(ttk.Frame(self.nb, padding=8))
        self._build_paired(ttk.Frame(self.nb, padding=8))
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind('<Escape>', lambda _e: self.close())
        self._poll()
        self._start_browsing()

    # ── plumbing ──────────────────────────────────────────────────────────
    def _poll(self) -> None:
        if self.bridge.closed.is_set():
            return
        self.bridge.drain()
        self.after(50, self._poll)

    def _discovery_or_none(self) -> Any:
        if self._discovery is None:
            try:
                self._discovery = self._make_discovery()
            except OSError as e:
                self.send_status.set(f"Cannot look for devices on this network: {e}")
                return None
        return self._discovery

    def close(self) -> None:
        """Close the window (receiving and browsing stop with it)."""
        self.destroy()

    def destroy(self) -> None:
        """Stop receiving and browsing however the window goes away, then
        close it: nothing keeps listening behind a closed window."""
        if self.bridge.closed.is_set():
            super().destroy()
            return
        self.bridge.closed.set()
        self._cancel.set()
        if self._listener is not None:
            self._listener.stop()
        if self._discovery is not None:
            try:
                self._discovery.close()
            except Exception:  # noqa: BLE001 - closing a window never fails on the network
                pass
        super().destroy()

    # ── Send tab ──────────────────────────────────────────────────────────
    def _build_send(self, f: ttk.Frame) -> None:
        self.nb.add(f, text="Send")
        ttk.Label(f, text=("Send a file to OpenFlo on a phone or another computer on this Wi-Fi.\n"
                           "To send a whole analysis with its data, send a project file "
                           "(File > Export project archive)."),
                  wraplength=520, justify='left').pack(anchor='w')
        row = ttk.Frame(f)
        row.pack(fill='x', pady=(8, 4))
        self.file_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.file_var)
        entry.pack(side='left', fill='x', expand=True)
        auto_tip(entry, "The file to send: a .flowproject, .flowsession, .fcs or .wsp file.")
        choose = ttk.Button(row, text="Choose file…", command=self._choose_file)
        choose.pack(side='left', padx=(6, 0))
        auto_tip(choose, "Pick the file to send.")
        ttk.Label(f, text="Devices nearby (open Receive on the other device):").pack(anchor='w', pady=(6, 2))
        self.devices = ttk.Treeview(f, columns=('state',), show='tree headings', height=7,
                                    selectmode='browse')
        self.devices.heading('#0', text='Device')
        self.devices.heading('state', text='')
        self.devices.column('state', width=90, anchor='w')
        self.devices.pack(fill='both', expand=True)
        btns = ttk.Frame(f)
        btns.pack(fill='x', pady=6)
        by_address = ttk.Button(btns, text="Send by address…", command=self._send_by_address)
        by_address.pack(side='left')
        auto_tip(by_address, "Send to an address the other device shows under Receive, "
                             "for networks where devices cannot find each other.")
        self.cancel_send_btn = ttk.Button(btns, text="Cancel", command=self._cancel.set,
                                          state='disabled')
        self.cancel_send_btn.pack(side='right')
        auto_tip(self.cancel_send_btn, "Stop the send now. A part already sent is kept "
                                       "on the other device for 24 hours, so sending again resumes.")
        self.send_btn = ttk.Button(btns, text="Send", command=self._send_clicked, default='active')
        self.send_btn.pack(side='right', padx=6)
        auto_tip(self.send_btn, "Send the file to the device picked in the list. The first time, "
                                "you type the code the other device shows.")
        self.send_progress = ttk.Progressbar(f, maximum=1000)
        self.send_progress.pack(fill='x')
        self.send_status = tk.StringVar(value="Looking for devices…")
        ttk.Label(f, textvariable=self.send_status, wraplength=520,
                  justify='left').pack(anchor='w', pady=(4, 0))

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Send which file?", filetypes=[
            ("OpenFlo files", ' '.join('*' + e for e in KINDS.values())), ("All files", '*')])
        if path:
            self.file_var.set(path)

    def _start_browsing(self) -> None:
        d = self._discovery_or_none()
        if d is None:
            return
        try:
            d.browse({APP}, lambda found: self.bridge.post(lambda: self._show_found(found)))
        except OSError as e:
            self.send_status.set(f"Cannot look for devices on this network: {e}")

    def _paired_by_txt(self) -> dict[str, bytes]:
        return {p.txt_id(d.static_pub): d.static_pub for d in self.store.paired()}

    def _show_found(self, found: list[Any]) -> None:
        if not self.winfo_exists():
            return
        self._found = list(found)
        keep = self.devices.selection()
        self.devices.delete(*self.devices.get_children())
        paired = self._paired_by_txt()
        # Rows are keyed by the device's service name, never by position: a
        # device appearing above the picked one must not move the pick.
        for f in self._found:
            if not self.devices.exists(f.service):
                self.devices.insert('', 'end', iid=f.service, text=f.name,
                                    values=('paired' if f.txt_id in paired else '',))
        if keep and self.devices.exists(keep[0]):
            self.devices.selection_set(keep[0])
        if not self._sending:
            self.send_status.set(f"{len(self._found)} device(s) found." if self._found
                                 else "No devices found yet. Open Receive on the other device.")

    def _send_clicked(self) -> None:
        sel = self.devices.selection()
        if not sel:
            self.send_status.set("Pick a device in the list first.")
            return
        f = next((x for x in self._found if x.service == sel[0]), None)
        if f is None:
            self.send_status.set("Pick a device in the list first.")
            return
        expected = self._paired_by_txt().get(f.txt_id)
        self._start_send(list(f.addresses), f.port, f.name, expected)

    def _send_by_address(self) -> None:
        text = simpledialog.askstring("Send by address",
                                      "The address the other device shows under Receive "
                                      "(for example 192.168.1.20:52000):", parent=self)
        if not text:
            return
        host, _, port = text.strip().rpartition(':')
        host = host.strip('[]')
        if not port.isdigit() or not p.is_lan_address(host):
            self.send_status.set("That is not a local-network address and port.")
            return
        self._start_send([host], int(port), host, None)

    def _start_send(self, addresses: list[str], port: int, label: str,
                    expected: bytes | None) -> None:
        path = self.file_var.get().strip()
        kind = kind_for_path(path)
        if not path or not os.path.isfile(path):
            self.send_status.set("Choose a file to send first.")
            return
        if kind is None:
            self.send_status.set("OpenFlo sends .flowproject, .flowsession, .fcs and .wsp files.")
            return
        if self._sending:
            return
        self._sending = True
        self._cancel.clear()
        self.send_btn.configure(state='disabled')
        self.cancel_send_btn.configure(state='normal')
        self.send_progress['value'] = 0
        self.send_status.set(f"Connecting to {label}…")
        ui = _TkSendUI(self)
        out = sess.Outgoing(path, kind)

        def work() -> None:
            from .nearby.transport import connect
            try:
                with connect(addresses, port) as sock:
                    peer = sess.send_file(sock, APP, self.store, out, ui, expected_pub=expected)
                msg = f"Sent to {peer.name}."
                if peer.newly_paired:
                    msg += " The devices are now paired; next time no code is needed."
            except Exception as e:  # noqa: BLE001 - every outcome is told to the user
                msg = describe_failure(e)
            # Through the window's queue like every other worker message, so
            # the answer arrives in order after the progress posted before it.
            self.bridge.post(lambda: self._send_finished(msg))
        run_async(self, work)

    def _send_finished(self, msg: str) -> None:
        self._sending = False
        self.send_btn.configure(state='normal')
        self.cancel_send_btn.configure(state='disabled')
        self.send_status.set(msg)
        self._refresh_paired()

    def ask_code(self, peer_name: str) -> str | None:
        """Ask for the code the receiving device shows (Tk thread)."""
        return simpledialog.askstring(
            "Pair with " + peer_name,
            f"First time sending to {peer_name}.\nType the 6-digit code it shows:",
            parent=self)

    # ── Receive tab ───────────────────────────────────────────────────────
    def _build_receive(self, f: ttk.Frame) -> None:
        self.nb.add(f, text="Receive")
        ttk.Label(f, text=("While receiving is on, devices on this network can see this computer "
                           "and offer it files. You are always asked before a file is taken.\n"
                           "Windows may ask to allow OpenFlo through its firewall; if this Wi-Fi "
                           "is set as a Public network, receiving may stay blocked. Sending from "
                           "this computer always works."),
                  wraplength=520, justify='left').pack(anchor='w')
        row = ttk.Frame(f)
        row.pack(fill='x', pady=8)
        self.receive_btn = ttk.Button(row, text="Start receiving", command=self.toggle_receiving)
        self.receive_btn.pack(side='left')
        auto_tip(self.receive_btn, "Let devices on this network see this computer and offer it "
                                   "files. Off when this window closes.")
        folder_btn = ttk.Button(row, text="Save files to…", command=self._choose_folder)
        folder_btn.pack(side='left', padx=6)
        auto_tip(folder_btn, "Choose the folder received files are saved in. "
                             "A file never replaces one already there.")
        self.folder_var = tk.StringVar(value=self.folder)
        ttk.Label(f, textvariable=self.folder_var, wraplength=520).pack(anchor='w')
        self.code_var = tk.StringVar()
        ttk.Label(f, textvariable=self.code_var, font='OpenFloTitle').pack(pady=8)
        self.receive_progress = ttk.Progressbar(f, maximum=1000)
        self.receive_progress.pack(fill='x')
        self.receive_status = tk.StringVar(value="Not receiving.")
        ttk.Label(f, textvariable=self.receive_status, wraplength=520,
                  justify='left').pack(anchor='w', pady=(4, 0))
        self.open_btn = ttk.Button(f, text="Open received file", command=self._open_received,
                                   state='disabled')
        self.open_btn.pack(anchor='w', pady=6)
        auto_tip(self.open_btn, "Open the last received file in OpenFlo.")

    def _choose_folder(self) -> None:
        d = filedialog.askdirectory(parent=self, title="Save received files in", mustexist=False)
        if d:
            self.folder = d
            self.folder_var.set(d)

    @property
    def receiving(self) -> bool:
        """Whether this computer is listening for senders now."""
        return self._listener is not None and self._listener.running

    def toggle_receiving(self) -> None:
        """Start or stop listening and announcing."""
        if self.receiving:
            self._stop_receiving("Not receiving.")
            return
        if self._listener is not None and self._listener.busy:
            # Stopping cut the last connection; it is still winding down. A
            # second listener now would let that one and a new one overlap.
            self.receive_status.set("Still ending the last connection. Try again in a moment.")
            return
        from .nearby.transport import Listener
        self._listener = Listener(self._serve_one,
                                  on_error=lambda e: self.bridge.post(
                                      lambda: self._receive_failed(e)),
                                  on_lockout=lambda: self.bridge.post(self._locked_out))
        port = self._listener.start()
        name = self.store.device_name()
        d = self._discovery_or_none()
        where = ''
        if d is not None:
            try:
                d.announce(APP, self.store.public_key(), name, port)
            except Exception as e:  # noqa: BLE001 - zeroconf has its own errors; receiving by address still works
                where = f" (not announced: {e})"
        from .nearby.discovery import local_addresses
        addrs = ', '.join(f"{a}:{port}" if ':' not in a else f"[{a}]:{port}"
                          for a in local_addresses()) or f"port {port}"
        self.receive_btn.configure(text="Stop receiving")
        self.receive_status.set(f"Visible to nearby devices as “{name}”{where}.\n"
                                f"Address for Send by address: {addrs}")

    def _stop_receiving(self, status: str) -> None:
        if self._listener is not None:
            self._listener.stop()
        if self._discovery is not None:
            self._discovery.stop_announcing()
        self.receive_btn.configure(text="Start receiving")
        self.code_var.set('')
        self.receive_status.set(status)

    def _locked_out(self) -> None:
        self._stop_receiving("Stopped receiving after several wrong codes. "
                             "Start again when you are ready.")

    def _serve_one(self, sock, host: str, pairing: threading.Event) -> None:
        """One sender's connection (listener's worker thread)."""
        ui = _TkReceiveUI(self)
        got = sess.receive_one(sock, APP, self.store, KINDS, self.folder, ui, pairing=pairing)
        self.bridge.post(lambda: self._received(got))

    def _received(self, got: sess.Received) -> None:
        self.received.append(got)
        self.code_var.set('')
        self.receive_status.set(f"Received {os.path.basename(got.path)} from {got.peer.name}, "
                                f"saved in {os.path.dirname(got.path)}.")
        self.open_btn.configure(state='normal')
        self._refresh_paired()

    def _receive_failed(self, exc: BaseException) -> None:
        self.code_var.set('')
        if self.winfo_exists():
            self.receive_status.set(describe_failure(exc)
                                    + (" Still receiving." if self.receiving else ""))

    def _open_received(self) -> None:
        if not self.received or self.editor is None:
            return
        got = self.received[-1]
        if got.kind in ('project', 'session'):
            self.editor._load_session_path(got.path)
        else:
            self.editor._import_dropped_paths([got.path])

    def ask_yes_no(self, title: str, text: str) -> bool:
        """A yes/no question on top of this window (Tk thread)."""
        return bool(messagebox.askyesno(title, text, parent=self))

    # ── Paired devices tab ────────────────────────────────────────────────
    def _build_paired(self, f: ttk.Frame) -> None:
        self.nb.add(f, text="Paired devices")
        row = ttk.Frame(f)
        row.pack(fill='x')
        self.name_var = tk.StringVar(value=self.store.device_name())
        ttk.Label(row, text="This computer's name:").pack(side='left')
        name_entry = ttk.Entry(row, textvariable=self.name_var, width=28)
        name_entry.pack(side='left', padx=6)
        auto_tip(name_entry, "The name other devices see for this computer. "
                             "Leave it blank for the default.")
        save_btn = ttk.Button(row, text="Save", command=self._save_name)
        save_btn.pack(side='left')
        auto_tip(save_btn, "Save this computer's name.")
        ttk.Label(f, text="Devices that can send and receive without a code:").pack(anchor='w', pady=(10, 2))
        self.paired_list = ttk.Treeview(f, show='tree', height=8, selectmode='browse')
        self.paired_list.pack(fill='both', expand=True)
        forget = ttk.Button(f, text="Forget", command=self._forget)
        forget.pack(anchor='w', pady=6)
        auto_tip(forget, "Forget the picked device: the next send between the two "
                         "asks for a code again.")
        self._refresh_paired()

    def _save_name(self) -> None:
        self.store.set_device_name(self.name_var.get())
        self.name_var.set(self.store.device_name())

    def _refresh_paired(self) -> None:
        if not self.winfo_exists():
            return
        self.paired_list.delete(*self.paired_list.get_children())
        for d in self.store.paired():
            self.paired_list.insert('', 'end', iid=d.static_pub.hex(), text=d.name)

    def _forget(self) -> None:
        sel = self.paired_list.selection()
        if sel:
            self.store.forget(bytes.fromhex(sel[0]))
            self._refresh_paired()


class _TkSendUI(sess.SendUI):
    def __init__(self, w: NearbyWindow):
        self.w = w
        self._last = -1

    def enter_code(self, peer_name: str) -> str | None:
        return self.w.bridge.ask(lambda: self.w.ask_code(peer_name), None)

    def waiting_for_peer_confirm(self, peer_name: str) -> None:
        self.w.bridge.post(lambda: self.w.send_status.set(
            f"Code correct. Tap Confirm on {peer_name} to finish pairing."))

    def progress(self, done: int, total: int) -> None:
        step = int(1000 * done / total) if total else 1000
        if step != self._last:
            self._last = step

            def show() -> None:
                self.w.send_progress['value'] = step
                self.w.send_status.set(f"Sending… {size_text(done)} of {size_text(total)}")
            self.w.bridge.post(show)

    def cancelled(self) -> bool:
        return self.w._cancel.is_set()


class _TkReceiveUI(sess.ReceiveUI):
    def __init__(self, w: NearbyWindow):
        self.w = w
        self._last = -1

    def show_code(self, code: str, peer_name: str) -> None:
        def show() -> None:
            self.w.code_var.set(f"{code[:3]} {code[3:]}")
            self.w.receive_status.set(f"{peer_name} wants to pair. Type this code on {peer_name}.")
        self.w.bridge.post(show)

    def confirm_pairing(self, code: str, peer_name: str) -> bool:
        return self.w.bridge.ask(lambda: self.w.ask_yes_no(
            "Pair", f"Code {code[:3]} {code[3:]} was typed on \u201c{peer_name}\u201d.\n\n"
                    "Confirm only if the device in your hand now says \u201cCode correct\u201d. "
                    "If it does not, another device may be trying to pair: choose No."), False)

    def accept(self, name: str, size: int, kind: str, peer_name: str) -> bool:
        self.w.bridge.post(lambda: self.w.code_var.set(''))
        return self.w.bridge.ask(lambda: self.w.ask_yes_no(
            "Receive file", f"Accept {name} ({size_text(size)}) from {peer_name}?"), False)

    def progress(self, done: int, total: int) -> None:
        step = int(1000 * done / total) if total else 1000
        if step != self._last:
            self._last = step

            def show() -> None:
                self.w.receive_progress['value'] = step
                self.w.receive_status.set(f"Receiving… {size_text(done)} of {size_text(total)}")
            self.w.bridge.post(show)

    def cancelled(self) -> bool:
        return self.w.bridge.closed.is_set()


def _real_discovery() -> Any:
    from .nearby.discovery import Discovery
    return Discovery()
