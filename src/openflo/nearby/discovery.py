"""Finding Nearby send devices on the local network (DNS-SD over mDNS, the
`zeroconf` package): the receiver announces itself while its Receive window
is open; the sender lists what it finds.

TXT record: ``v`` (protocol version), ``app``, ``id`` (protocol.txt_id) and
``n`` (the device's display name). Only entries with a version and app this
side speaks are listed.
"""
from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import protocol as p


@dataclass(frozen=True)
class Found:
    """A receiver seen on the network."""
    service: str            # the DNS-SD instance name (unique on the network)
    name: str
    txt_id: str
    app: str
    addresses: tuple[str, ...]
    port: int


def _txt(props: dict[Any, Any], key: str) -> str:
    v = props.get(key.encode(), props.get(key))
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    return v if isinstance(v, str) else ''


def found_from(service: str, props: dict[Any, Any], addresses: list[str], port: int,
               apps: set[str]) -> Found | None:
    """The listed entry for one resolved service, or None when its version,
    app or details are not ones this side takes."""
    if _txt(props, 'v') != str(p.VERSION):
        return None
    app = _txt(props, 'app')
    if app not in apps or not 0 < port < 65536:
        return None
    lan = tuple(a for a in addresses if p.is_lan_address(a))
    if not lan:
        return None
    return Found(service, p.safe_device_name(_txt(props, 'n')), _txt(props, 'id')[:16],
                 app, lan, port)


def txt_record(app: str, static_pub: bytes, name: str) -> dict[str, str]:
    """The TXT record this device announces."""
    return {'v': str(p.VERSION), 'app': app, 'id': p.txt_id(static_pub),
            'n': name[:p.MAX_DEVICE_NAME]}


def instance_name(name: str, static_pub: bytes) -> str:
    """The DNS-SD instance name: the display name made unique by the id."""
    clean = ''.join(c for c in name if c.isprintable() and c != '.')[:40] or 'Device'
    return f'{clean} {p.txt_id(static_pub)[:6]}.{p.SERVICE_TYPE}'


class Discovery:
    """One zeroconf instance for announcing and browsing.

    ``announce`` and ``browse`` may be used together; ``close`` ends both.
    ``on_change(list of Found)`` is called from zeroconf's thread."""

    def __init__(self) -> None:
        from zeroconf import IPVersion, Zeroconf
        try:
            self._zc = Zeroconf(ip_version=IPVersion.All)
        except OSError:                       # no IPv6 on this machine
            self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        self._info: Any = None
        self._browser: Any = None
        self._found: dict[str, Found] = {}
        self._lock = threading.Lock()

    def announce(self, app: str, static_pub: bytes, name: str, port: int) -> None:
        """Announce this device until ``stop_announcing`` or ``close``."""
        from zeroconf import ServiceInfo
        addrs = [socket.inet_pton(socket.AF_INET6 if ':' in a else socket.AF_INET, a)
                 for a in local_addresses()]
        self._info = ServiceInfo(p.SERVICE_TYPE, instance_name(name, static_pub),
                                 port=port, properties=txt_record(app, static_pub, name),
                                 addresses=addrs)
        self._zc.register_service(self._info, allow_name_change=True)

    def stop_announcing(self) -> None:
        """Withdraw the announcement (other devices drop it from their list)."""
        if self._info is not None:
            try:
                self._zc.unregister_service(self._info)
            except Exception:  # noqa: BLE001 - best effort on the way out
                pass
            self._info = None

    def browse(self, apps: set[str], on_change: Callable[[list[Found]], None]) -> None:
        """List receivers of ``apps`` as they appear and go."""
        from zeroconf import ServiceBrowser, ServiceStateChange

        def handler(zeroconf, service_type, name, state_change):
            if state_change is ServiceStateChange.Removed:
                with self._lock:
                    self._found.pop(name, None)
            else:
                info = zeroconf.get_service_info(service_type, name, timeout=3000)
                if info is None:
                    return
                f = found_from(name, info.properties or {}, info.parsed_scoped_addresses(),
                               info.port or 0, apps)
                with self._lock:
                    if f is None:
                        self._found.pop(name, None)
                    else:
                        self._found[name] = f
            with self._lock:
                listed = sorted(self._found.values(), key=lambda f: f.name.lower())
            on_change(listed)

        self._browser = ServiceBrowser(self._zc, p.SERVICE_TYPE, handlers=[handler])

    def close(self) -> None:
        """Stop announcing and browsing."""
        self.stop_announcing()
        if self._browser is not None:
            try:
                self._browser.cancel()
            except Exception:  # noqa: BLE001 - best effort on the way out
                pass
        self._zc.close()


def local_addresses() -> list[str]:
    """This machine's local-network addresses to announce, via ifaddr
    (IPv6 link-local left out: it needs a zone the other side cannot use)."""
    import ifaddr
    out = []
    for adapter in ifaddr.get_adapters():
        for ip in adapter.ips:
            a = ip.ip if isinstance(ip.ip, str) else ip.ip[0]
            if p.is_lan_address(a) and not a.startswith(('127.', 'fe80')) and a != '::1':
                out.append(a)
    return out
