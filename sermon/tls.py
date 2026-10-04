"""Self-signed TLS cert for the local Wi-Fi server.

Browsers refuse the microphone API on plain HTTP outside localhost, so the
speaker page can't get mic access unless the origin is HTTPS. On a LAN there
is no real CA to issue us a cert; we generate our own once and cache it in
`certs/`. Phones will show a "not secure" warning on first visit - the person
holding the phone taps through it once, and normal HTTPS behaviour applies
after that.

Cert covers:
* every non-loopback IPv4 address of the host at generation time
* `localhost`

If the host's IP changes (new Wi-Fi, hotspot toggled), regenerate:
    (delete certs/server.crt and certs/server.key, restart)
"""

from __future__ import annotations

import datetime
import ipaddress
import ssl
from pathlib import Path


CERT_DIR = Path(__file__).resolve().parent.parent / "certs"
CERT_PATH = CERT_DIR / "server.crt"
KEY_PATH = CERT_DIR / "server.key"


def _generate(hosts: list[str]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for h in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san.append(x509.DNSName(h))

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "neo-language-box.local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Neo Language Box"),
    ])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365 * 5))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    KEY_PATH.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def ensure_cert(hosts: list[str]) -> tuple[Path, Path]:
    """Return (cert_path, key_path), regenerating if the cert doesn't yet
    cover the given hosts."""
    need = not (CERT_PATH.exists() and KEY_PATH.exists())
    if not need:
        # Check whether existing cert covers each of these hosts
        try:
            from cryptography import x509
            existing = x509.load_pem_x509_certificate(CERT_PATH.read_bytes())
            san_ext = existing.extensions.get_extension_for_class(
                x509.SubjectAlternativeName).value
            covered: set[str] = set()
            for name in san_ext:
                if isinstance(name, x509.IPAddress):
                    covered.add(str(name.value))
                elif isinstance(name, x509.DNSName):
                    covered.add(name.value)
            missing = [h for h in hosts if h not in covered]
            if missing:
                need = True
        except Exception:
            need = True
    if need:
        _generate(hosts)
    return CERT_PATH, KEY_PATH


def ssl_context(hosts: list[str]) -> ssl.SSLContext:
    cert, key = ensure_cert(hosts)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    return ctx
