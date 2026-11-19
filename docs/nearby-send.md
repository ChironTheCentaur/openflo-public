# Nearby send

Status: protocol core (`src/openflo/nearby/protocol.py`) and its test
vectors (`src/openflo/nearby/vectors_v1.json`) are in; desktop, Android and
iOS follow in separate changes.

Goal: send OpenFlo files (.flowproject, .flowsession, .fcs, .wsp) between the
desktop app (Python/Tk), Android and iPhone on the same local network, with
automatic discovery, a one-time short-code pairing, Wi-Fi speed, and no
internet relay. The same core must later drop into Pocket Lab gel and plasmid
(phone-only Capacitor apps), so nothing in the core knows about OpenFlo.

Out of scope for v1: Bluetooth/NFC (pairing or small files, maybe later),
sending to a device on another network, background receiving.

## 1. User flow

1. Receiver opens **Nearby send > Receive**. While that screen is open (and
   only then) it announces itself on the local network.
2. Sender opens **Nearby send > Send file**, picks a file (or "this
   project"), and sees the receivers found nearby, by name.
3. First time between two devices: the receiver shows a 6-digit code, the
   user types it on the sender, and then confirms on the receiver once the
   sender says the code was right (section 3, Pairing). The devices
   remember each other ("paired"); later sends skip the code.
4. Receiver is always asked "Accept FILE (284 MB) from DEVICE?" (safe
   default; no auto-accept in v1).
5. Progress on both sides; the file is checked by SHA-256; if the link drops,
   sending again resumes where it stopped.
6. Received file: phone hands it to the same intake as "Open with" (kind
   check, size cap) and offers to open it; desktop saves it to a chosen folder
   (default `Downloads/OpenFlo`), never over an existing file, and offers to
   open it.

Paired devices are listed in Settings with "Forget".

## 2. Discovery

- DNS-SD service type `_pl-send._tcp` (one type for all Pocket Lab apps), TXT
  record: `v=1`, `app=openflo`, `id=` first 8 bytes of the device's identity
  fingerprint (base32), `n=` display name.
- Display name default is generic: "Windows PC", "Android phone", "iPhone"
  plus a 4-character suffix from the fingerprint; user can rename it. No
  account name, no device owner name.
- Sender lists only services with `v=1` and an `app` it accepts. A gel app can
  see an OpenFlo receiver but is told "OpenFlo cannot take this file" before
  any connection.
- Fallback when multicast is blocked (some campus Wi-Fi): "Connect by
  address" shows `IP:port` on the receiver; the sender types it. Same
  handshake.
- Platform APIs: desktop `zeroconf` (pure Python, LGPL-2.1); Android
  `NsdManager`; iOS `NWListener`/`NWBrowser` (Network.framework).

## 3. Protocol (OFNS v1, over one TCP connection)

Crypto suite, chosen because every platform has it built in: P-256 ECDH and
ECDSA, HKDF-SHA256, AES-256-GCM, SHA-256. (Android: JCA, minSdk 24 has all of
it; iOS: CryptoKit; desktop: `cryptography`.) No X25519 because Android only
has it from API 31+ in JCA.

Each install has a long-term identity key pair (P-256), made on first use.
Fingerprint = SHA-256 of the uncompressed public point. Stored: Android
AndroidKeyStore (non-exportable, sign only; paired list and partials excluded
from backup), iOS Keychain (ThisDeviceOnly), desktop user config dir (0600 on
POSIX; profile ACL on Windows). The identity key only signs, so it never
needs ECDH (Keystore ECDH is API 31+).
Privacy note: R sends its public identity key to anyone who connects, and
the TXT `id` is stable while Receive is open. Accepted and documented.
HKDF on Android is written over HmacSHA256 (RFC 5869 vectors); CryptoKit HKDF
needs iOS 14+.

### Handshake (commitment + short authentication string)

I = sender (connects), R = receiver (listens). Every message has a 4-byte BE
length prefix; byte layouts are fixed in the spec file. Points are 65-byte
uncompressed P-256; every received point is checked on-curve and not
infinity (all three platforms, with a negative vector). ECDSA signatures are
DER, SHA-256.

```
M1 I->R: "OFNS" | ver | app | C = SHA-256("pl-send v1 commit" || eI || nI)
M2 R->I: eR | nR (32 random bytes)        fresh for EVERY connection
M3 I->R: eI | nI (32 random bytes)        R checks C; both validate points
h    = SHA-256(M1 || M2 || M3)            with their length prefixes
ss   = ECDH x-coordinate (32 bytes)
k_IR | k_RI = HKDF-SHA256(ikm=ss, salt=h, info="pl-send v1 keys", 64)
SAS  = BE uint32(HKDF-SHA256(ikm=ss, salt=h, info="pl-send v1 sas", 4)) mod 10^6
-- encrypted records from here --
both: Identity{ static_pub, name, ECDSA(static_priv, "pl-send v1 id" || role || h) }
both: PairState{ known }   known = "I have this peer's static key stored"
both known      -> Offer
otherwise       -> pairing (below); save the peer only after sending AND
                   receiving Confirm
```

Invariants (write into the spec, test each):
- eR, nR, eI, nI are fresh per connection, never cached (reuse lets an
  attacker grind the code offline).
- The SAS depends only on M1-M3 and ss. Nothing sent after M3 (static keys,
  names) is ever mixed into it.
- The commit check, point checks and signature checks happen before any
  state changes.
- An unsupported `ver` gets an explicit `Unsupported{max_ver}` reply so the
  UI can say "update the other app". Refusals are not authenticated, so a
  future version must never fall back to an older one because of one: it
  says "update the other app" and stops.
- A Responder that refused (or failed to parse) M1 takes nothing more.

Pairing (first time between two devices):
- R shows the 6-digit code; the sender's user types it on I; I compares
  locally (no blind "codes match" tap). On a match I sends `Confirm` and shows
  "Code correct, tap Confirm on the other device"; R's user then taps
  Confirm, and R sends `Confirm`. A wrong code, Cancel or 60 s timeout:
  `Abort`, close, nothing saved.
- The code proves R to I, not I to R: whoever is at the other end of the
  connection also knows the code. So (a) while R is pairing, any second
  connection makes R abort the pairing and say "another device tried to
  connect; start again", and (b) R's Confirm question repeats the code and
  the sender's name and says to confirm only if the device in your hand
  shows "Code correct".
- Names shown to the user go through `safe_device_name` (control, format,
  surrogate and private-use characters removed, at most 64 UTF-8 bytes).
  For a paired device the UI shows the label stored when it paired, never
  the name the peer sends now.
- The user picked a device from the paired list but its static key is not
  the stored one: hard fail "this is not your paired device", never a new
  pairing prompt. Paired devices are identified by stored key, never by name.
- Receiving always asks "Accept ...?", paired or not.

### Records

`len (4 bytes BE, plaintext length+16, also the AAD) || AES-256-GCM(ct||tag)`.
Nonce = 4 zero bytes || 8-byte BE counter, a separate counter per direction,
starting at 0; a record that fails to decrypt, or a counter reaching 2^32,
ends the connection. Max plaintext 64 KiB; a length above 65,552 is refused
before any allocation. Unknown message types and messages out of state end
the connection. First plaintext byte = message
type. An explicit `Close` record ends a session (a bare TCP close mid-file is
an interrupted transfer, never a finished one).

Records can arrive in the same socket read as the last handshake message (a
sender writes M3 and its Identity back to back). Read the handshake one
message at a time and hand every byte after M2 (at I) or M3 (at R) to the
record layer unparsed: a handshake parser that splits the whole read would
take those records for handshake messages, and refuse them as too long or
out of order.

### Transfer messages

```
Offer   { transfer_id (16 random), name, size, sha256, kind }
Accept  { offset }            receiver resumes from a kept partial file
Decline { reason }            user said no / kind / too big / no space
Data    { bytes }             in order from offset
Done    { }                   sender: all bytes sent
Result  { ok, reason }        receiver: SHA-256 checked
```

- Resume: the receiver keeps `partial/<peer fp>/<sha256>.part` for 24 h. A new
  Offer with the same sha256 and size from the same paired peer gets
  `Accept{offset = part length}`; the receiver re-hashes the kept part first.
- Name is reduced to a base name (`safe_file_name`): control, format (bidi
  override, zero width), surrogate and private-use characters removed
  (unassigned code points are kept, so platforms on different Unicode
  versions agree); `\/:*?"<>|` become `_`; Windows reserved names (CON,
  PRN, AUX, NUL, CONIN$, CONOUT$, COM1-9, LPT1-9, with any extension) get a
  "received " prefix; leading and trailing dots and spaces are removed; the
  whole name is at most 200 UTF-8 bytes, cut on a code-point boundary;
  the extension comes from the validated kind, not the sender's name.
  The receiver picks the final file name (never overwrites).
- Resume safety: a failed Result deletes the .part; the sender refuses
  offset > size; the sender hashes and sends from a stable snapshot (a copy
  or a size+mtime check before and after); partial files have a total cap
  and a free-space check.
- A received file is hostile input: the project/session loaders it reaches
  are checked for zip-slip and unsafe deserialisation in PR 1.
- Kind and size are checked against the receiver app's own list and cap
  before the user is asked, and again while bytes arrive.

## 4. Limits and abuse

- Listen only while Receive is open; accept peers from private, link-local
  and loopback addresses only (RFC 1918, 169.254/16, 127/8, fc00::/7,
  fe80::/10, ::1); refuse others. Global IPv6 addresses, common on home
  networks, are refused too: the sender tries the receiver's IPv4 (or
  ULA / link-local) address instead.
- Timeouts: handshake 10 s, code confirm 60 s, 30 s idle mid-transfer.
- One incoming connection at a time, with per-address backoff (a peer that
  fails or aborts waits longer each time); after 3 failed pairings in 5
  minutes, stop announcing until the user reopens Receive.
- Addresses: IPv4-mapped IPv6 normalised before the private-range check;
  fe80 scope ids kept; listen dual-stack; the sender tries every resolved
  address. Wi-Fi with client isolation gets a clear "devices on this network
  can't reach each other" message.
- Nothing goes to any server; no analytics event for transfers.

## 5. Platform notes

- **Desktop**: new deps `cryptography` (Apache-2.0/BSD) and `zeroconf`
  (LGPL-2.1; add to third-party notices). Windows shows a firewall prompt on
  the first Receive; allowing needs admin and many home Wi-Fi networks are
  classed "Public", so the Receive screen explains this and says sending from
  the PC (outbound) always works. macOS builds get their own local network
  prompt. Network work on a
  background thread; Tk updated via the existing async task helper.
- **Android**: `NsdManager` for announce/discover (resolves queued one at a
  time on old APIs); `ServerSocket`/`Socket` bound to the Wi-Fi `Network` so
  an active VPN does not swallow them; wake lock + screen on during a
  transfer. Manifest: INTERNET (already). Local network permission: opt-in at
  targetSdk 36, enforced when targeting 37 (permission name to verify on the
  current Android docs in PR 3). Multicast lock believed unneeded with
  NsdManager; verify. Kotlin protocol core has no Android imports so CI runs
  it on the JVM; one emulator run on API 24 for Conscrypt.
- **iOS**: Info.plist `NSLocalNetworkUsageDescription` ("OpenFlo finds your
  other devices on this Wi-Fi to send files to them.") and
  `NSBonjourServices = [_pl-send._tcp]`. iOS asks the user once, the first
  time Receive or Send opens. No multicast entitlement needed for Bonjour.
  Transfers stop if the app goes to the background (v1: warn "keep OpenFlo
  open"); resume covers it. A denied prompt fails silently: detect it and
  point to Settings.
- **Store labels**: no transfer data goes to us or a third party, so
  Nearby send adds nothing to the App Store privacy label or Play Data
  safety answers (the app's other rows, ads and opt-in usage counts, are
  in mobile/store/listing.md; re-check against the current wording when
  filling them in).
  Export compliance: only OS-provided standard crypto (CryptoKit), believed
  exempt, so `ITSAppUsesNonExemptEncryption=false` stands; confirm on Apple's
  export compliance page before submission.

## 6. Code layout (reuse in gel/plasmid)

- **Python** `src/openflo/nearby/`: `protocol.py` (sans-IO: framing,
  crypto, the handshake, and `Session`, the message-order state machine; no
  OpenFlo imports, copyable as is), `session.py` (one connection end to end
  over a socket), `transport.py` (listener, connect), `discovery.py`
  (zeroconf), `store.py` (identity, paired devices, partials); GUI in
  `ui_nearby.py`.
- **Phones**: a local Capacitor plugin package
  `mobile/plugins/pocketlab-nearby/` (TypeScript definitions, Kotlin, Swift)
  that takes `app`, accepted kinds and size cap at start. Protocol, hashing
  and file I/O are native, so file bytes never cross the JS bridge. Gel and
  plasmid copy the package.
- **Conformance**: `src/openflo/nearby/vectors_v1.json` (fixed keys and randoms ->
  expected h, keys, SAS, record bytes) generated by the Python reference;
  Kotlin JVM tests and Swift tests read the same file. CI runs a Python <->
  Kotlin loopback transfer. Swift tests need macOS (Codemagic) since iOS tests
  run nowhere automatically today.

## 7. Changes

1. Spec + Python protocol core + vectors + tests (no UI).
2. Desktop: discovery, transport, Tk screens, paired-device settings.
3. Android: plugin (Kotlin), phone screens (TS), manifest.
4. iOS: plugin (Swift), Info.plist keys.
5. Later: the same plugin in the other Pocket Lab apps.

## Byte layouts (version 1)

Integers are big-endian. `str` = u16 length + UTF-8 bytes. Points are 65
bytes (0x04 | x | y). The vectors file has an example of every message.

Handshake bodies (each sent as u32 length + body; h hashes them WITH their
length prefixes):

| Message | Body |
|---|---|
| M1 | "OFNS" · u8 version · u8 app length · app (ASCII `[a-z0-9-]`, ≤32) · commit (32) |
| M2 | u8 status 0 · eR (65) · nR (32) |
| M2 refusal | u8 status 1 · u8 highest version, or u8 status 2 (wrong app); nothing after |
| M3 | eI (65) · nI (32) |

Handshake bodies above 256 bytes are refused from the length alone.

Record messages (first byte = type):

| Type | Message | Rest |
|---|---|---|
| 0x01 | Identity | static pub (65) · str name (≤64 bytes) · u16 length · DER signature (≤80) |
| 0x02 | PairState | u8 known (0/1) |
| 0x03 | Confirm | |
| 0x04 | Abort | u8 reason |
| 0x10 | Offer | transfer id (16) · u64 size (≤ 2^63−1) · sha256 (32) · str kind (≤32) · str name (≤1024) |
| 0x11 | Accept | u64 offset (≤ 2^63−1) |
| 0x12 | Decline | u8 reason |
| 0x13 | Data | bytes to the end of the record (1 to 65,535; empty is refused) |
| 0x14 | Done | |
| 0x15 | Result | u8 ok (0/1) · u8 reason |
| 0x7F | Close | |

Reasons: 1 user, 2 timeout, 3 wrong code, 4 kind, 5 too big, 6 no space,
7 hash mismatch, 8 busy, 9 not the paired device. A flag other than 0/1,
an unknown type, a short body or extra bytes end the connection. Sizes and
offsets of 2^63 or more are refused (they must fit a signed 64-bit integer
on every platform).

Message order (`Session`): Identity, then PairState, once each. When
pairing, I's Confirm comes first and R's after it, and nothing else passes
until both have; Confirm is refused when no pairing is needed. Then one
Offer; Accept (offset ≤ size) or Decline; after Accept, Data totalling
exactly size − offset, then Done; then R's Result and I's Close. A Close
before the Result (or Decline) is refused, so the transfer counts as
interrupted. Abort ends the session anywhere; nothing is taken after it.

Identity signature: ECDSA P-256 SHA-256, DER, over
`"pl-send v1 id" | role ("I" or "R", the signer's) | h`.

## Test vectors

`python -m openflo.nearby.vectors` checks `vectors_v1.json` against the
Python reference; `--write` rewrites it after a deliberate change. Every
implementation must: replay the handshake from the listed scalars and
nonces and match M1-M3, h, ss, the keys and the code; decrypt each record
list with its key in order and get the listed plaintexts; verify both
identity signatures (and fail identity R checked as role I); match the
RFC 5869 HKDF case; match every file-name, device-name and address row;
refuse every `negative` entry; and run every `sessions` scenario through
its Session (`send` steps recorded, `recv` steps checked), failing at the
listed step (or not at all when it is null). ECDSA signatures are random, so other platforms
only verify the listed ones.

## Why this design

The commit-then-reveal handshake with a 6-digit code is the pattern of
Bluetooth numeric comparison and ZRTP: a man in the middle gets one
one-in-a-million guess per attempt. Noise was not used because P-256 is
not a standard Noise curve and a code over its handshake hash would need the
same commitment; SPAKE2/CPace need elliptic-curve arithmetic that neither
Android's JCA nor iOS's CryptoKit exposes. Each platform implements the
protocol natively (Python, Kotlin, Swift) against the shared vectors, so
file bytes never cross the phone app's JavaScript bridge.
