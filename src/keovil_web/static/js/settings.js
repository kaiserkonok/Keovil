async function deactivateNode() {
    if (!confirm("CRITICAL: This will unbind the Master Key from this hardware. You will be locked out until re-activation. Proceed?")) return;

    try {
        const res = await fetch('/api/logout', { method: 'POST' });
        if (res.ok) {
            window.location.href = '/activate';
        }
    } catch (err) {
        alert("Logout failed: " + err);
    }
}
